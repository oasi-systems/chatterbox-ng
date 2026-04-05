# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional, Tuple

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .transformer.upsample_encoder import EncoderCaches
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """
    S3Gen's CFM decoder maps S3 speech tokens to mel-spectrograms.

    TODO: make these modules configurable?
    """
    def __init__(self, meanflow=False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus(
            # NOTE: This doesn't affect inference. It turns off activation checkpointing
            # (a training optimization), which causes a crazy DDP error with accelerate
            memory_efficient=False,
        )
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
            meanflow=self.meanflow,
        )
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    @property
    def dtype(self):
        params = self.flow.parameters()
        return next(params).dtype

    def _autocast_ctx(self):
        """Return autocast context if configured, else nullcontext."""
        dtype = getattr(self, '_autocast_dtype', None)
        if dtype and self.device != 'cpu' and torch.cuda.is_available():
            return torch.autocast('cuda', dtype=dtype)
        return nullcontext()

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: s3gen received ref longer than 10s")

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        # mel_extractor uses torch.stft (FFT) — must stay fp32
        ref_wav_24 = ref_wav_24.to(device=device, dtype=torch.float32)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)
        ref_mels_24_len = None

        # Resample to 16kHz
        ref_wav_16 = ref_wav
        if ref_sr != S3_SR:
            ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav)

        # Speaker embedding — always fp32 (fbank uses FFT, not supported in BF16)
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=torch.float32))

        # Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
        noised_mels=None,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.
        - This function is designed for batch_size=1 only.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        - `finalize`: whether streaming is finished or not. Note that if False, the last 3 tokens will be ignored.
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(device=self.device, dtype=self.dtype)

        speech_tokens = torch.atleast_2d(speech_tokens)

        # backcompat
        if speech_token_lens is None:
            speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            noised_mels=noised_mels,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of S3Gen is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.

    TODO: make these modules configurable?
    """

    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow=False):
        super().__init__(meanflow)

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False) # (buffers get automatic device casting)
        self.estimator_dtype = "fp32"

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens=None,
        skip_vocoder=False,
        n_cfm_timesteps=None,
        noised_mels=None,

    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.
        NOTE: used for sync synthesis only. Please use `S3GenStreamer` for streaming synthesis.
        """
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav,
            ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps, noised_mels=noised_mels,
        )

        if skip_vocoder:
            return output_mels

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
    ):
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        noise = None
        if self.meanflow:
            # When finalize=False, the encoder truncates by pre_lookahead_len * token_mel_ratio frames.
            # We must match the noise size to the actual mel output, not the full token count.
            mel_len = speech_tokens.size(-1) * 2
            if not finalize:
                mel_len -= self.flow.pre_lookahead_len * self.flow.token_mel_ratio
            noise = torch.randn(1, 80, mel_len, device=self.device)
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps, finalize=finalize, noised_mels=noise,
        )
        return output_mels

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        # left as a kwarg because this can change input/output size ratio
        drop_invalid_tokens=True,
        n_cfm_timesteps=None,
        speech_token_lens=None,
    ):
        # hallucination prevention, drop special tokens
        # if drop_invalid_tokens:
        #     speech_tokens, speech_token_lens = drop_invalid(speech_tokens, pad=S3_QUIET_PAD)

        output_mels = self.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )
        output_mels = output_mels.to(dtype=self.dtype) # FIXME (fp16 mode) is this still needed?
        output_wavs, output_sources = self.hift_inference(output_mels, None)

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources

    # ---- Streaming support ----

    @dataclass
    class StreamState:
        """Mutable state for streaming inference."""
        hifi_cache_source: torch.Tensor = None  # HiFiGAN source cache for continuity
        prev_stable_mel_len: int = 0  # number of stable mel frames already emitted
        is_first_chunk: bool = True
        # Encoder KV-cache
        encoder_caches: EncoderCaches = None
        # Cached outputs for context window
        cached_encoder_output: torch.Tensor = None  # (B, accumulated_mel_frames, 80) projected
        cached_mel: torch.Tensor = None             # (B, 80, accumulated_mel_frames) generated mel
        # Projected speaker embedding (cached after first chunk)
        spk_embedding_proj: torch.Tensor = None     # (B, 80) projected

    def init_streaming(self, ref_dict: dict = None) -> 'S3Token2Wav.StreamState':
        """Initialize streaming state. Call once before streaming tokens.

        Args:
            ref_dict: reference audio embeddings from embed_ref(). If provided,
                seeds the HiFiGAN cache from the reference audio tail to eliminate
                cold-start artifacts on the first chunk.
        """
        # HiFiGAN cache is always fp32 (vocoder runs in fp32)
        hifi_cache = torch.zeros(1, 1, 0, device=self.device, dtype=torch.float32)

        # Seed cache from reference audio to warm HiFiGAN conv layers
        if ref_dict is not None and 'prompt_feat' in ref_dict:
            prompt_feat = ref_dict['prompt_feat']  # (B, mel_len, 80)
            warmup_frames = min(5, prompt_feat.shape[1])
            warmup_mel = prompt_feat[:, -warmup_frames:].transpose(1, 2)  # (B, 80, frames)
            with torch.inference_mode():
                _, warmup_source = self.mel2wav.inference(
                    speech_feat=warmup_mel.float(),
                    cache_source=torch.zeros(1, 1, 0, device=self.device),
                )
            cache_len = min(warmup_source.shape[2], S3GEN_SR // 10)
            hifi_cache = warmup_source[:, :, -cache_len:]

        return S3Token2Wav.StreamState(
            hifi_cache_source=hifi_cache,
            prev_stable_mel_len=0,
            is_first_chunk=True,
            encoder_caches=self.flow.encoder.init_caches(self.device, torch.float32),
        )

    @torch.inference_mode()
    def streaming_step(
        self,
        all_tokens: torch.Tensor,
        ref_dict: dict,
        state: 'S3Token2Wav.StreamState',
        finalize: bool = False,
        n_cfm_timesteps: int = None,
    ) -> Tuple[Optional[torch.Tensor], 'S3Token2Wav.StreamState']:
        """Process accumulated tokens and return NEW audio (only the delta).

        Args:
            all_tokens: ALL speech tokens generated so far (1D or 2D tensor)
            ref_dict: reference audio embeddings from embed_ref()
            state: streaming state from init_streaming()
            finalize: True when EOS reached (includes lookahead frames)
            n_cfm_timesteps: CFM ODE steps (default: 2 for meanflow, 10 otherwise)

        Returns:
            (audio_chunk or None, updated_state)
            audio_chunk is None if no new stable frames are available yet.
        """
        all_tokens = torch.atleast_2d(all_tokens)

        # --- Flow inference in fp32 (NOT autocast) ---
        # Autocast causes mel output in bf16 (7-bit mantissa ≈ 42dB dynamic range).
        # Mel spectrograms need ~80dB range → bf16 quantization creates metallic
        # harmonics. The monolithic path runs flow_inference in fp32; streaming
        # must do the same for quality parity.
        # Speed comes from meanflow (2 ODE steps), not from reduced precision.
        output_mels = self.flow_inference(
            speech_tokens=all_tokens,
            ref_dict=ref_dict,
            finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps,
        )

        total_mel_len = output_mels.shape[2]

        # How many NEW stable mel frames do we have?
        new_mel_len = total_mel_len - state.prev_stable_mel_len
        if new_mel_len <= 0:
            return None, state

        # Extract only the NEW mel frames
        new_mels = output_mels[:, :, state.prev_stable_mel_len:]

        # Run HiFiGAN on new mel frames with cache for continuity.
        # HiFiGAN runs in fp32 (STFT/phase precision).
        #
        # Source signal length = mel_frames × f0_upsamp_scale.
        # f0_upsamp_scale = prod(upsample_rates) × istft_hop = 120 × 4 = 480.
        SOURCE_PER_MEL = 480
        estimated_source_len = new_mels.shape[2] * SOURCE_PER_MEL
        cache = state.hifi_cache_source
        if cache.shape[2] > estimated_source_len:
            cache = cache[:, :, -estimated_source_len:]

        audio_chunk, new_source = self.mel2wav.inference(
            speech_feat=new_mels.float(),
            cache_source=cache.float(),
        )

        # Apply trim fade only on first chunk to reduce reference spillover
        if state.is_first_chunk:
            trim_len = min(len(self.trim_fade), audio_chunk.shape[1])
            audio_chunk[:, :trim_len] *= self.trim_fade[:trim_len]

        # Update state
        # Keep full source signal for cache — ensures SineGen phase continuity
        # across chunk boundaries. Previous 100ms cap caused phase jumps at
        # the cache→fresh boundary, producing metallic artifacts.
        state.hifi_cache_source = new_source
        state.prev_stable_mel_len = total_mel_len
        state.is_first_chunk = False

        return audio_chunk, state

    @torch.inference_mode()
    def streaming_step_windowed(
        self,
        all_tokens: torch.Tensor,
        ref_dict: dict,
        state: 'S3Token2Wav.StreamState',
        finalize: bool = False,
        n_cfm_timesteps: int = None,
        context_frames: int = 20,
    ) -> Tuple[Optional[torch.Tensor], 'S3Token2Wav.StreamState']:
        """Hybrid streaming: full encoder + CFM on full [prompt|speech] sequence.

        Mirrors the standard inference() path exactly:
        - Encoder: re-processes ALL tokens (correct for bidirectional encoder)
        - CFM: runs on FULL [prompt_mel | speech_mel] with prompt_feat conditioning,
          freezing prompt + previously generated speech frames
        - HiFiGAN: uses waveform cache for continuity

        This produces identical quality to monolithic generate() because the CFM
        decoder sees the same inputs: full encoder mu, prompt_feat as conditioning,
        and speaker embedding.

        Args:
            all_tokens: ALL speech tokens so far
            ref_dict: reference audio embeddings
            state: streaming state
            finalize: True when EOS
            n_cfm_timesteps: ODE steps (default: 2 for meanflow, 10 otherwise)
            context_frames: (unused, kept for API compat — freeze is now prompt + prev speech)
        """
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        all_tokens = torch.atleast_2d(all_tokens)

        # Prepare ref_dict tensors
        prompt_token = ref_dict['prompt_token'].to(self.device, dtype=torch.long)
        prompt_token_len = ref_dict['prompt_token_len'].to(self.device)
        prompt_feat = ref_dict['prompt_feat'].to(self.device, dtype=self.dtype)
        embedding = ref_dict['embedding'].to(self.device, dtype=self.dtype)

        speech_token_lens = torch.LongTensor([all_tokens.size(-1)]).to(self.device)

        # --- Step 1: Full encoder (correct bidirectional) ---
        # Returns FULL [prompt | speech] mu, not just speech
        full_mu, spk_emb, prompt_mel_len = self.flow.encode_only(
            token=all_tokens,
            token_len=speech_token_lens,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            embedding=embedding,
            finalize=finalize,
        )
        # full_mu: (B, 80, prompt_mel + speech_mel)

        total_speech_mel = full_mu.size(2) - prompt_mel_len
        new_mel_frames = total_speech_mel - state.prev_stable_mel_len

        if new_mel_frames <= 0:
            return None, state

        # --- Step 2: CFM on full [prompt | speech] with freeze ---
        # This matches standard inference() exactly:
        # - mu = full encoder output [prompt | speech]
        # - cond = [prompt_feat | zeros] (voice conditioning)
        # - freeze_len = prompt_mel + previously generated speech mel
        new_mel = self.flow.inference_windowed(
            full_mu=full_mu,
            prompt_feat=prompt_feat,
            embedding=spk_emb,
            prompt_mel_len=prompt_mel_len,
            prev_speech_mel_len=state.prev_stable_mel_len,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            cached_mel=state.cached_mel,
        )
        new_mel = new_mel.to(dtype=self.dtype)

        # Accumulate cached mel (full sequence: prompt + all speech)
        if state.cached_mel is None:
            # First chunk: build [prompt_mel | new_speech_mel]
            # Initialize prompt portion from z_init (will be frozen in subsequent calls)
            prompt_cached = torch.zeros(1, 80, prompt_mel_len, device=self.device, dtype=self.dtype)
            if prompt_feat.size(1) > 0:
                prompt_cached = prompt_feat.transpose(1, 2).contiguous().to(self.dtype)
            state.cached_mel = torch.cat([prompt_cached, new_mel], dim=2)
        else:
            state.cached_mel = torch.cat([state.cached_mel, new_mel], dim=2)

        # --- Step 3: HiFiGAN vocoding ---
        estimated_source_len = new_mel.shape[2] * 120
        cache = state.hifi_cache_source
        if cache.shape[2] > estimated_source_len:
            cache = cache[:, :, -estimated_source_len:]

        audio_chunk, new_source = self.mel2wav.inference(
            speech_feat=new_mel,
            cache_source=cache,
        )

        # Apply trim fade on first chunk
        if state.is_first_chunk:
            trim_len = min(len(self.trim_fade), audio_chunk.shape[1])
            audio_chunk[:, :trim_len] *= self.trim_fade[:trim_len]

        # Update state
        cache_len = min(new_source.shape[2], S3GEN_SR // 10)
        state.hifi_cache_source = new_source[:, :, -cache_len:]
        state.prev_stable_mel_len = total_speech_mel
        state.is_first_chunk = False

        return audio_chunk, state

    @torch.inference_mode()
    def streaming_step_cached(
        self,
        all_tokens: torch.Tensor,
        ref_dict: dict,
        state: 'S3Token2Wav.StreamState',
        finalize: bool = False,
        n_cfm_timesteps: int = None,
        context_frames: int = 20,
    ) -> Tuple[Optional[torch.Tensor], 'S3Token2Wav.StreamState']:
        """Process accumulated tokens with encoder KV-cache and CFM context window.

        Instead of re-encoding all tokens and running CFM on the full sequence,
        this method:
        1. Encoder: only processes NEW tokens via KV-cache (saves encoder recomputation)
        2. CFM: runs ODE on [context_window | new_frames] with frozen context (saves CFM cost)

        Args:
            all_tokens: ALL speech tokens generated so far (1D or 2D tensor)
            ref_dict: reference audio embeddings from embed_ref()
            state: streaming state from init_streaming()
            finalize: True when EOS reached
            n_cfm_timesteps: CFM ODE steps
            context_frames: number of mel frames to use as frozen context for CFM

        Returns:
            (audio_chunk or None, updated_state)
        """
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        all_tokens = torch.atleast_2d(all_tokens)

        # Prepare prompt tokens and speaker embedding (from ref_dict)
        prompt_token = ref_dict['prompt_token'].to(self.device, dtype=torch.long)
        prompt_token_len = ref_dict['prompt_token_len'].to(self.device)
        embedding = ref_dict['embedding'].to(self.device, dtype=self.dtype)

        # Project speaker embedding once and cache it
        if state.spk_embedding_proj is None:
            embedding = torch.atleast_2d(embedding)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            state.spk_embedding_proj = self.flow.spk_embed_affine_layer(embedding)

        speech_token_lens = torch.LongTensor([all_tokens.size(-1)]).to(self.device)

        # --- Step 1: Encoder with KV-cache ---
        new_h_proj, state.encoder_caches = self.flow.inference_cached(
            token=all_tokens,
            token_len=speech_token_lens,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            embedding=state.spk_embedding_proj,
            finalize=finalize,
            encoder_caches=state.encoder_caches,
        )

        if new_h_proj.size(1) == 0:
            return None, state

        new_h_proj = new_h_proj.to(dtype=self.dtype)

        # Accumulate encoder output
        if state.cached_encoder_output is not None:
            state.cached_encoder_output = torch.cat(
                [state.cached_encoder_output, new_h_proj], dim=1
            )
        else:
            state.cached_encoder_output = new_h_proj

        # --- Step 2: CFM with context window ---
        new_mel_frames = new_h_proj.size(1)
        actual_context = 0

        if state.cached_mel is not None and context_frames > 0:
            actual_context = min(context_frames, state.cached_mel.size(2))
            context_mel = state.cached_mel[:, :, -actual_context:]
            context_mu = state.cached_encoder_output[:, -(new_mel_frames + actual_context):-new_mel_frames]
            context_mu = context_mu.transpose(1, 2).contiguous()  # (B, 80, ctx)
        else:
            context_mel = None
            context_mu = None

        new_mu = new_h_proj.transpose(1, 2).contiguous()  # (B, 80, new_frames)

        if context_mu is not None:
            mu_windowed = torch.cat([context_mu, new_mu], dim=2)
        else:
            mu_windowed = new_mu
            context_mel = torch.zeros(1, 80, 0, device=self.device, dtype=self.dtype)

        # For meanflow, generate matching noise
        noise = None
        if self.meanflow:
            noise = torch.randn(1, 80, mu_windowed.size(2), dtype=self.dtype, device=self.device)
            if actual_context > 0:
                noise[:, :, :actual_context] = context_mel

        new_mel = self.flow.inference_windowed(
            mu_windowed=mu_windowed,
            context_mel=context_mel if actual_context > 0 else torch.zeros(1, 80, 0, device=self.device, dtype=self.dtype),
            embedding=state.spk_embedding_proj,
            context_frames=actual_context,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
        )

        new_mel = new_mel.to(dtype=self.dtype)

        # Accumulate generated mel
        if state.cached_mel is not None:
            state.cached_mel = torch.cat([state.cached_mel, new_mel], dim=2)
        else:
            state.cached_mel = new_mel

        # --- Step 3: HiFiGAN vocoding ---
        estimated_source_len = new_mel.shape[2] * 120
        cache = state.hifi_cache_source
        if cache.shape[2] > estimated_source_len:
            cache = cache[:, :, -estimated_source_len:]

        audio_chunk, new_source = self.mel2wav.inference(
            speech_feat=new_mel,
            cache_source=cache,
        )

        # Apply trim fade on first chunk
        if state.is_first_chunk:
            trim_len = min(len(self.trim_fade), audio_chunk.shape[1])
            audio_chunk[:, :trim_len] *= self.trim_fade[:trim_len]

        # Update state
        cache_len = min(new_source.shape[2], S3GEN_SR // 10)
        state.hifi_cache_source = new_source[:, :, -cache_len:]
        state.prev_stable_mel_len += new_mel.size(2)
        state.is_first_chunk = False

        return audio_chunk, state
