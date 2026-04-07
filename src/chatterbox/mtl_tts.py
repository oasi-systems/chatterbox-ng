from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config, BPE_VOCAB_SIZE
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str, language_id: str = None) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset.
        Supports language-specific rules when language_id is provided.
    """
    if len(text) == 0:
        if language_id == 'it':
            return "Devi aggiungere del testo da leggere."
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("\u201c", "\""),
        ("\u201d", "\""),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ]

    # Italian-specific: guillemets to double quotes
    if language_id == 'it':
        punc_to_replace.extend([
            ("«", "\""),
            ("»", "\""),
        ])

    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


def _load_bpe_vocab(ckpt_dir: Path = None) -> dict:
    """Load BPE vocab dict {token_str: token_id} from tokenizer JSON.

    Searches: ckpt_dir, package dir, HF cache.
    """
    import json
    import glob

    candidates = []
    if ckpt_dir:
        candidates.append(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")

    _pkg_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(_pkg_dir / "models" / "tokenizers" / "grapheme_mtl_merged_expanded_v1.json")

    # HF cache
    for path in glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--ResembleAI--chatterbox/*/grapheme_mtl_merged_expanded_v1.json")):
        candidates.append(Path(path))

    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if "model" in data and "vocab" in data["model"]:
                vocab_data = data["model"]["vocab"]
                if isinstance(vocab_data, dict):
                    return vocab_data
                elif isinstance(vocab_data, list):
                    return {tok: i for i, (tok, _score) in enumerate(vocab_data)}

    return {}


def _extend_t3_state_with_phonemes(state_dict: dict, new_vocab_size: int,
                                    ckpt_dir: Path = None) -> dict:
    """Extend T3 state dict from BPE-only (2454) to BPE+phoneme vocab.

    Adds new rows to text_emb.weight and text_head.weight.
    New embedding rows initialized from related grapheme embeddings
    (smart init from phoneme_tokens.py). Original weights untouched.

    Args:
        state_dict: original T3 state dict with [2454, dim] text weights
        new_vocab_size: target vocab size (2454 + N_phoneme_tokens)
        ckpt_dir: directory to search for vocab JSON

    Returns:
        Modified state dict with extended text_emb and text_head.
    """
    import logging
    from .phoneme_tokens import initialize_phoneme_embeddings

    _logger = logging.getLogger(__name__)

    old_emb = state_dict["text_emb.weight"]  # [2454, 1024]
    old_head = state_dict["text_head.weight"]  # [2454, 1024]
    old_size, dim = old_emb.shape
    n_new = new_vocab_size - old_size

    assert n_new > 0, f"new_vocab_size ({new_vocab_size}) must be > old ({old_size})"

    # Load BPE vocab for grapheme→ID mapping
    bpe_vocab = _load_bpe_vocab(ckpt_dir)
    if bpe_vocab:
        _logger.info(f"Loaded BPE vocab ({len(bpe_vocab)} tokens) for phoneme init")
    else:
        _logger.warning("BPE vocab not found — phoneme embeddings will use random init")

    # text_emb: smart init from grapheme mappings
    new_emb_rows = initialize_phoneme_embeddings(old_emb, bpe_vocab)  # [n_new, dim]
    state_dict["text_emb.weight"] = torch.cat([old_emb, new_emb_rows], dim=0)

    # text_head: random init with matching std
    head_std = old_head.std().item()
    new_head_rows = torch.zeros(n_new, dim)
    new_head_rows.normal_(0, head_std)
    state_dict["text_head.weight"] = torch.cat([old_head, new_head_rows], dim=0)

    _logger.info(
        f"Extended T3: text_emb {old_size}→{new_vocab_size}, "
        f"text_head {old_size}→{new_vocab_size} "
        f"({n_new} phoneme tokens added)"
    )

    return state_dict


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device, meanflow=False, meanflow_ckpt_dir=None,
                   phoneme_mode=False) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
        )
        ve.to(device).eval()

        # --- T3 with optional phoneme embedding extension ---
        if phoneme_mode:
            t3_config = T3Config.multilingual_phoneme()
        else:
            t3_config = T3Config.multilingual()

        t3 = T3(t3_config)

        # Load pretrained weights
        t3_ckpt = ckpt_dir / "t3_phoneme.safetensors"
        if not t3_ckpt.exists():
            t3_ckpt = ckpt_dir / "t3_mtl23ls_v2.safetensors"

        t3_state = load_safetensors(t3_ckpt)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]

        if phoneme_mode and t3_state["text_emb.weight"].shape[0] == BPE_VOCAB_SIZE:
            # Old checkpoint (2454 tokens) → extend with phoneme embeddings
            import logging
            _logger = logging.getLogger(__name__)
            _logger.info("Extending T3 with phoneme embeddings (2454 → %d tokens)",
                         t3_config.text_tokens_dict_size)
            t3_state = _extend_t3_state_with_phonemes(
                t3_state, t3_config.text_tokens_dict_size, ckpt_dir=ckpt_dir,
            )
            t3.load_state_dict(t3_state, strict=True)
        else:
            # Either not phoneme mode, or checkpoint already has extended vocab
            t3.load_state_dict(t3_state, strict=True)

        t3.to(device).eval()

        s3gen = S3Gen(meanflow=meanflow)
        if meanflow and meanflow_ckpt_dir:
            meanflow_ckpt_dir = Path(meanflow_ckpt_dir)
            s3gen.load_state_dict(
                load_safetensors(meanflow_ckpt_dir / "s3gen_meanflow.safetensors"),
                strict=True,
            )
        else:
            s3gen.load_state_dict(
                torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True)
            )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"),
            phoneme_mode=phoneme_mode,
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device, meanflow: bool = False,
                        phoneme_mode: bool = False) -> 'ChatterboxMultilingualTTS':
        """Load pretrained multilingual ChatterBox model.

        Args:
            device: torch device
            meanflow: if True, load meanflow S3Gen weights from the turbo repo.
                Uses 2 ODE steps instead of 10 + no CFG batch doubling = ~5-7x CFM speedup.
                Quality may differ — the meanflow weights were distilled from English data.
            phoneme_mode: if True, extend vocab with IPA phoneme tokens for
                EU languages (IT/EN/FR/DE/ES/PT). Original weights preserved.
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )

        meanflow_ckpt_dir = None
        if meanflow:
            meanflow_ckpt_dir = Path(
                snapshot_download(
                    repo_id=TURBO_REPO_ID,
                    repo_type="model",
                    revision="main",
                    allow_patterns=["s3gen_meanflow.safetensors"],
                    token=os.getenv("HF_TOKEN"),
                )
            )

        return cls.from_local(ckpt_dir, device, meanflow=meanflow,
                              meanflow_ckpt_dir=meanflow_ckpt_dir,
                              phoneme_mode=phoneme_mode)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text, language_id=language_id.lower() if language_id else None)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
