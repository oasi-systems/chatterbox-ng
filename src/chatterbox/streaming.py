"""
Streaming TTS orchestrator for ChatterBox.

Yields audio chunks as speech tokens are generated, providing real-time audio output
instead of waiting for the full utterance to complete.

Architecture:
    T3 (yields tokens) → Token buffer → S3Gen (growing-sequence mel) → HiFiGAN (cached vocoding) → Audio chunks

The encoder+CFM process the growing token sequence with `finalize=False`,
which guarantees that emitted mel frames are stable (last 6 frames are trimmed
as they depend on future context via PreLookaheadLayer).
HiFiGAN's `cache_source` maintains waveform continuity between chunks.

Sentence pipelining mode:
    Text is split into sentences. Each sentence gets its own T3 generation,
    but audio is emitted continuously with HiFiGAN cache maintaining waveform continuity.
"""
import logging
import os
import re as _re
from typing import Generator, Tuple, Union, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import perth

from math import gcd

from .models.s3tokenizer import drop_invalid_tokens, SPEECH_VOCAB_SIZE
from .models.s3gen import S3GEN_SR

# Built-in breath templates directory
_BREATH_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'breath_templates')

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at natural boundaries.

    Handles Italian and English punctuation, keeping the delimiter with the sentence.
    """
    # Split on sentence-ending punctuation followed by space or end of string
    # Keep the punctuation with the sentence
    parts = _re.split(r'(?<=[.!?…])\s+', text.strip())
    # Filter empty strings and strip whitespace
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences if sentences else [text]


class StreamingResampler:
    """Streaming audio resampler for converting between sample rates without boundary artifacts.

    Accumulates input and re-resamples the full signal each call, returning only
    new output samples. This guarantees zero boundary artifacts (bit-exact with
    offline resampling). The O(N²) total cost is negligible for speech-length signals
    (~68ms for a 30s utterance).

    Usage:
        resampler = StreamingResampler(24000, 16000)
        for chunk_24k in audio_chunks:
            chunk_16k = resampler.process(chunk_24k)
            send_to_asterisk(chunk_16k)
    """

    def __init__(self, orig_sr: int = S3GEN_SR, target_sr: int = 16000):
        from scipy.signal import resample_poly as _resample_poly
        self._resample_poly = _resample_poly
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        g = gcd(orig_sr, target_sr)
        self._up = target_sr // g
        self._down = orig_sr // g
        self._all_input = np.array([], dtype=np.float32)
        self._output_emitted = 0

    def process(self, chunk: np.ndarray, finalize: bool = False) -> np.ndarray:
        """Resample a chunk and return new output samples.

        Args:
            chunk: input audio at orig_sr
            finalize: if True, flush all remaining samples (last chunk)

        Returns:
            Resampled audio at target_sr. May be empty if not enough input yet.
        """
        self._all_input = np.concatenate([self._all_input, chunk])
        all_output = self._resample_poly(self._all_input, self._up, self._down)

        if finalize:
            new = all_output[self._output_emitted:]
            self._output_emitted = len(all_output)
            return new.astype(np.float32)

        # Hold back last few samples to avoid shutdown transient
        # Filter half_len=10, max(up,down) taps → ~20 output samples affected
        holdback = 20
        safe_end = len(all_output) - holdback
        if safe_end <= self._output_emitted:
            return np.array([], dtype=np.float32)

        new = all_output[self._output_emitted:safe_end]
        self._output_emitted = safe_end
        return new.astype(np.float32)

    def reset(self):
        """Reset state for a new utterance."""
        self._all_input = np.array([], dtype=np.float32)
        self._output_emitted = 0


class ChatterboxStreamingTTS:
    """Streaming wrapper for ChatterboxMultilingualTTS.

    Usage:
        model = ChatterboxMultilingualTTS.from_pretrained(device)
        streamer = ChatterboxStreamingTTS(model)

        for audio_chunk in streamer.generate_stream("Hello world", audio_prompt_path="ref.wav"):
            play_audio(audio_chunk, sample_rate=streamer.sample_rate)

        # Optional: get full watermarked audio after streaming
        full_audio = streamer.get_full_watermarked()
    """

    def __init__(
        self,
        model,
        chunk_tokens: int = 25,
        min_initial_tokens: int = 15,
        output_sample_rate: int = 16000,
        # Adaptive chunking: start with small chunks for low FCL, grow for quality
        adaptive_chunking: bool = True,
        adaptive_schedule: tuple = None,  # e.g. (8, 15, 25) — token thresholds per chunk
        # Initial breath: emit a short breath before the first speech chunk
        # to mask cold-start artifacts (double consonant "BBuongiorno" etc.)
        initial_breath: bool = False,
        initial_breath_duration_ms: int = 120,
        # Voice humanization: insert breaths between sentences in real-time
        humanize: bool = False,
        humanizer_reference: str = None,  # path to reference audio for breath adaptation
        humanizer = None,  # pre-built VoiceHumanizer instance (reuse across calls)
        # Deprecated — kept for API compat, ignored
        streaming_cfm_steps: int = None,
        use_cfm_windowing: bool = False,
        cfm_context_frames: int = 30,
        use_kv_cache: bool = False,
        efficient_streaming: bool = True,
    ):
        """
        Args:
            model: ChatterboxMultilingualTTS instance
            chunk_tokens: number of speech tokens to buffer before emitting audio (~40ms per token)
            min_initial_tokens: minimum tokens before first audio emission (higher = better first-chunk quality)
            output_sample_rate: resample output to this rate. Default 16000 (telephony/Asterisk).
                Uses streaming polyphase resampler with zero boundary artifacts.
                Set to 24000 for native quality without resampling.
            adaptive_chunking: if True, use progressive chunk sizes — small first chunk
                for low latency, growing chunks for better quality.
            adaptive_schedule: tuple of token counts for each chunk. After the schedule
                is exhausted, chunk_tokens is used. Default: (5, 10, 20, chunk_tokens).
            humanize: if True, insert natural breaths in silence gaps during streaming.
                Requires humanizer_reference or humanizer to be set.
            humanizer_reference: path to reference audio for breath adaptation.
                If set, automatically enables humanize=True.
            humanizer: pre-built VoiceHumanizer instance. Use this to avoid
                re-extracting breath profiles on every call.

        Note:
            Streaming always uses full ODE steps (same as monolithic generate()).
            For speed, use meanflow=True when loading the model — this uses 2 ODE steps
            by design, without quality degradation. Never reduce ODE steps arbitrarily.
        """
        self.model = model
        self.chunk_tokens = chunk_tokens
        self.min_initial_tokens = min_initial_tokens
        self.adaptive_chunking = adaptive_chunking
        if adaptive_schedule is not None:
            self.adaptive_schedule = tuple(adaptive_schedule)
        else:
            # Default: aggressive ramp — 5→10→20→chunk_tokens
            # Benchmarked: -53% to -59% FCL vs fixed 15-token first chunk,
            # with equal or better perceived audio quality.
            self.adaptive_schedule = (5, 10, 20, chunk_tokens)
        # Always use full ODE steps — quality must match monolithic.
        # Speed comes from meanflow (2 steps by design), not from reducing steps.
        self.streaming_cfm_steps = None
        self.use_cfm_windowing = False
        self.cfm_context_frames = cfm_context_frames
        self.use_kv_cache = False

        # Efficient streaming: windowed CFM decoder (O(1) per step instead of O(N²)).
        # Uses full encoder (correct bidirectional) + CFM on [context + new] frames only.
        # First chunk uses full CFM for voice identity, subsequent chunks use windowed.
        self.efficient_streaming = efficient_streaming

        if output_sample_rate and output_sample_rate != S3GEN_SR:
            self._resampler = StreamingResampler(S3GEN_SR, output_sample_rate)
            self.sample_rate = output_sample_rate
        else:
            self._resampler = None
            self.sample_rate = S3GEN_SR
        self._all_chunks = []
        if self._resampler is not None:
            self._resampler.reset()
        self._watermarker = None  # lazy init — only created when get_full_watermarked() is called

        # Humanizer setup
        if humanizer is not None:
            self._humanizer = humanizer
        elif humanizer_reference or humanize:
            from .humanizer import VoiceHumanizer
            ref_path = humanizer_reference
            if ref_path is None and hasattr(model, '_last_audio_prompt_path'):
                ref_path = model._last_audio_prompt_path
            if ref_path:
                self._humanizer = VoiceHumanizer.from_reference(ref_path)
            else:
                self._humanizer = None
                logger.warning("humanize=True but no reference audio provided. "
                               "Pass humanizer_reference or call prepare_conditionals() first.")
        else:
            self._humanizer = None

        # Initial breath config
        self._initial_breath = initial_breath
        self._initial_breath_duration_ms = initial_breath_duration_ms
        self._breath_templates_cache = None  # lazy-loaded

        # Crossfade buffer for seamless chunk boundaries
        # HiFiGAN conv layers produce edge artifacts at chunk boundaries.
        # We hold back a small overlap from each chunk and crossfade it with
        # the beginning of the next chunk to eliminate clicks.
        self._crossfade_samples = S3GEN_SR // 50  # 20ms = 480 samples at 24kHz
        self._crossfade_buffer = None  # held-back tail of previous chunk

        # Humanizer streaming state
        self._cumulative_speech_s = 0.0
        self._last_breath_time_s = -999.0
        self._audio_emitted_s = 0.0
        self._rms_sum_sq = 0.0  # running sum of squared samples for overall RMS
        self._rms_n_samples = 0  # total samples counted

    def _is_multilingual(self):
        return hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'cangjie_converter')

    def _load_breath_templates(self):
        """Lazy-load breath templates from the built-in directory."""
        if self._breath_templates_cache is not None:
            return self._breath_templates_cache

        import librosa
        templates = []
        if os.path.isdir(_BREATH_TEMPLATES_DIR):
            for fname in sorted(os.listdir(_BREATH_TEMPLATES_DIR)):
                if fname.endswith('.wav') and 'breath' in fname.lower():
                    path = os.path.join(_BREATH_TEMPLATES_DIR, fname)
                    audio, _ = librosa.load(path, sr=S3GEN_SR)
                    templates.append(audio)
        if templates:
            logger.info(f"Loaded {len(templates)} breath templates for initial breath")
        else:
            logger.warning("No breath templates found — initial breath disabled")
        self._breath_templates_cache = templates
        return templates

    def _get_initial_breath(self) -> Optional[np.ndarray]:
        """Get a short breath to emit before the first speech chunk.

        Returns breath audio at output_sample_rate, or None if unavailable.
        """
        if not self._initial_breath:
            return None

        templates = self._load_breath_templates()
        if not templates:
            return None

        # Pick a random template
        rng = np.random.default_rng()
        idx = rng.integers(0, len(templates))
        breath = templates[idx].copy()

        # Trim to target duration
        target_samples = int(self._initial_breath_duration_ms / 1000 * S3GEN_SR)
        if len(breath) > target_samples:
            breath = breath[:target_samples]

        # Gentle fade in/out (5ms) to avoid clicks
        fade = int(0.005 * S3GEN_SR)
        if len(breath) > 2 * fade:
            breath[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            breath[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        # Scale to a soft level (breaths are quiet)
        rms = np.sqrt(np.mean(breath ** 2))
        if rms > 0:
            breath *= 0.03 / rms  # target RMS ~0.03 (soft breath)

        # Resample to output rate if needed
        if self._resampler is not None:
            # Use a one-shot resample (not the streaming resampler state)
            import torchaudio
            breath_t = torch.from_numpy(breath).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(S3GEN_SR, self.sample_rate)
            breath_t = resampler(breath_t)
            breath = breath_t.squeeze(0).numpy()

        return breath.astype(np.float32)

    def _tokenize_text(self, text: str, language_id: Optional[str], device):
        """Normalize and tokenize text, adding SOT/EOT."""
        is_multilingual = self._is_multilingual()

        if is_multilingual:
            from .mtl_tts import punc_norm
            text = punc_norm(text, language_id=language_id.lower() if language_id else None)
            text_tokens = self.model.tokenizer.text_to_tokens(
                text, language_id=language_id.lower() if language_id else None
            ).to(device)
        else:
            from .tts import punc_norm
            text = punc_norm(text)
            text_tokens = self.model.tokenizer.text_to_tokens(text).to(device)

        sot = self.model.t3.hp.start_text_token
        eot = self.model.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        return text_tokens

    def _start_t3_stream(self, text_tokens, cfg_weight, temperature,
                         repetition_penalty, min_p, top_p):
        """Start T3 token generator for given text tokens."""
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        return self.model.t3.inference_streaming(
            t3_cond=self.model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = None,
        # T3 params
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 0.95,
        cfg_weight: float = 0.0,
        exaggeration: float = 0.7,
        # S3Gen params
        n_cfm_timesteps: Optional[int] = None,
        # Pipelining splits text into sentences with independent T3 passes.
        # Disabled by default to preserve prosody continuity.
        sentence_pipelining: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks as speech tokens are generated.

        Supports SSML input: if the text contains SSML tags, it is automatically
        parsed into segments. Each segment is generated with its own prosody
        parameters, and <break> tags produce silence intervals.

        Args:
            sentence_pipelining: if True, split text into sentences and process
                each independently through T3 while maintaining audio continuity.
                Reduces latency for long texts and improves stability.

        Yields:
            np.ndarray: audio chunk (1D float array at self.sample_rate Hz).
                        Chunks are NOT watermarked — call get_full_watermarked() after streaming.
        """
        # --- Auto-detect SSML ---
        from .ssml import is_ssml
        if is_ssml(text):
            yield from self._generate_stream_ssml(
                text=text, audio_prompt_path=audio_prompt_path, language_id=language_id,
                temperature=temperature, repetition_penalty=repetition_penalty,
                min_p=min_p, top_p=top_p, cfg_weight=cfg_weight, exaggeration=exaggeration,
                n_cfm_timesteps=n_cfm_timesteps,
            )
            return

        if sentence_pipelining:
            yield from self._generate_stream_pipelined(
                text=text, audio_prompt_path=audio_prompt_path, language_id=language_id,
                temperature=temperature, repetition_penalty=repetition_penalty,
                min_p=min_p, top_p=top_p, cfg_weight=cfg_weight, exaggeration=exaggeration,
                n_cfm_timesteps=n_cfm_timesteps,
            )
            return

        self._all_chunks = []
        # Reset humanizer state for new utterance
        self._cumulative_speech_s = 0.0
        self._last_breath_time_s = -999.0
        self._audio_emitted_s = 0.0
        self._rms_sum_sq = 0.0
        self._rms_n_samples = 0
        self._crossfade_buffer = None
        if self._resampler is not None:
            self._resampler.reset()
        device = self.model.device

        # --- Prepare conditionals ---
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.model.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        # --- Tokenize text ---
        text_tokens = self._tokenize_text(text, language_id, device)

        # --- Initialize streaming state (seed HiFiGAN from reference audio) ---
        stream_state = self.model.s3gen.init_streaming(ref_dict=self.model.conds.gen)
        accumulated_tokens = []

        # --- Emit initial breath (masks cold-start artifacts like "BBuongiorno") ---
        initial_breath = self._get_initial_breath()
        if initial_breath is not None:
            yield initial_breath

        # --- Start T3 streaming ---
        token_gen = self._start_t3_stream(
            text_tokens, cfg_weight, temperature,
            repetition_penalty, min_p, top_p,
        )

        # --- Stream tokens and emit audio chunks ---
        tokens_since_last_emit = 0
        chunk_index = 0  # tracks which chunk we're building (for adaptive schedule)

        for token in token_gen:
            token_val = token.view(-1)
            if token_val.item() < SPEECH_VOCAB_SIZE:
                accumulated_tokens.append(token_val)
            tokens_since_last_emit += 1

            threshold = self._get_chunk_threshold(chunk_index, stream_state.is_first_chunk)
            if tokens_since_last_emit >= threshold and len(accumulated_tokens) > 0:
                audio_chunk = self._emit_chunk(accumulated_tokens, stream_state, finalize=False, n_cfm_timesteps=n_cfm_timesteps)
                if audio_chunk is not None:
                    yield audio_chunk
                tokens_since_last_emit = 0
                chunk_index += 1

        # --- Final chunk with finalize=True ---
        if len(accumulated_tokens) > 0:
            audio_chunk = self._emit_chunk(accumulated_tokens, stream_state, finalize=True, n_cfm_timesteps=n_cfm_timesteps)
            if audio_chunk is not None:
                yield audio_chunk

    def _generate_stream_pipelined(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        language_id: Optional[str],
        temperature: float,
        repetition_penalty: float,
        min_p: float,
        top_p: float,
        cfg_weight: float,
        exaggeration: float,
        n_cfm_timesteps: Optional[int],
    ) -> Generator[np.ndarray, None, None]:
        """Sentence-level pipelining: process each sentence through T3 independently
        while maintaining continuous audio output via shared S3Gen state.

        Benefits:
        - T3 context doesn't grow unbounded for long texts
        - More stable generation (each sentence starts fresh)
        - Natural sentence boundaries produce cleaner prosody
        """
        self._all_chunks = []
        # Reset humanizer state for new utterance
        self._cumulative_speech_s = 0.0
        self._last_breath_time_s = -999.0
        self._audio_emitted_s = 0.0
        self._rms_sum_sq = 0.0
        self._rms_n_samples = 0
        self._crossfade_buffer = None
        if self._resampler is not None:
            self._resampler.reset()
        device = self.model.device

        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.model.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        sentences = _split_sentences(text)
        logger.info(f"Sentence pipelining: {len(sentences)} sentence(s)")

        # Shared S3Gen state across all sentences for audio continuity
        stream_state = self.model.s3gen.init_streaming(ref_dict=self.model.conds.gen)

        # --- Emit initial breath before first sentence ---
        initial_breath = self._get_initial_breath()
        if initial_breath is not None:
            yield initial_breath

        for sent_idx, sentence in enumerate(sentences):
            is_last_sentence = (sent_idx == len(sentences) - 1)

            # Tokenize this sentence
            text_tokens = self._tokenize_text(sentence, language_id, device)

            # Start T3 for this sentence
            token_gen = self._start_t3_stream(
                text_tokens, cfg_weight, temperature,
                repetition_penalty, min_p, top_p,
            )

            # Collect tokens for this sentence (fresh accumulation per sentence)
            sentence_tokens = []
            tokens_since_last_emit = 0

            for token in token_gen:
                token_val = token.view(-1)
                if token_val.item() < SPEECH_VOCAB_SIZE:
                    sentence_tokens.append(token_val)
                tokens_since_last_emit += 1

                threshold = self.min_initial_tokens if stream_state.is_first_chunk else self.chunk_tokens
                if tokens_since_last_emit >= threshold and len(sentence_tokens) > 0:
                    audio_chunk = self._emit_chunk(
                        sentence_tokens, stream_state,
                        finalize=False, n_cfm_timesteps=n_cfm_timesteps,
                    )
                    if audio_chunk is not None:
                        yield audio_chunk
                    tokens_since_last_emit = 0

            # Finalize this sentence's tokens
            if len(sentence_tokens) > 0:
                finalize = is_last_sentence
                audio_chunk = self._emit_chunk(
                    sentence_tokens, stream_state,
                    finalize=finalize, n_cfm_timesteps=n_cfm_timesteps,
                )
                if audio_chunk is not None:
                    yield audio_chunk

            # Reset S3Gen mel tracking for next sentence but keep HiFiGAN cache
            # This way: new sentence starts fresh mel generation,
            # but audio waveform stays continuous via HiFiGAN cache
            if not is_last_sentence:
                stream_state.prev_stable_mel_len = 0
                # Reset encoder caches and cached outputs for fresh sentence encoding
                if hasattr(stream_state, 'encoder_caches') and stream_state.encoder_caches is not None:
                    stream_state.encoder_caches = self.model.s3gen.flow.encoder.init_caches(
                        self.model.device, next(self.model.s3gen.parameters()).dtype
                    )
                    stream_state.cached_encoder_output = None
                    stream_state.cached_mel = None

    def _generate_stream_ssml(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        language_id: Optional[str],
        temperature: float,
        repetition_penalty: float,
        min_p: float,
        top_p: float,
        cfg_weight: float,
        exaggeration: float,
        n_cfm_timesteps: Optional[int],
    ) -> Generator[np.ndarray, None, None]:
        """Process SSML markup: generate audio for text segments, silence for breaks.

        Each text segment uses its own exaggeration/cfg_weight from SSML tags.
        Audio continuity is maintained across segments via shared S3Gen state.
        """
        from .ssml import parse_ssml

        self._all_chunks = []
        self._cumulative_speech_s = 0.0
        self._last_breath_time_s = -999.0
        self._audio_emitted_s = 0.0
        self._rms_sum_sq = 0.0
        self._rms_n_samples = 0
        self._crossfade_buffer = None
        if self._resampler is not None:
            self._resampler.reset()
        device = self.model.device

        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.model.conds is not None, "Call prepare_conditionals() first"

        segments = parse_ssml(text, default_language=language_id)
        logger.info(f"SSML: {len(segments)} segments")

        # Shared S3Gen streaming state across all segments
        stream_state = self.model.s3gen.init_streaming(ref_dict=self.model.conds.gen)

        # Emit initial breath
        initial_breath = self._get_initial_breath()
        if initial_breath is not None:
            yield initial_breath

        for seg_idx, seg in enumerate(segments):
            is_last = (seg_idx == len(segments) - 1)

            # --- Break: emit silence ---
            if seg.is_break:
                silence = self._make_silence(seg.break_duration_ms)
                if silence is not None and len(silence) > 0:
                    self._all_chunks.append(silence)
                    # Resample silence to output rate
                    if self._resampler is not None:
                        silence = self._resampler.process(silence, finalize=False)
                    if len(silence) > 0:
                        yield silence
                continue

            # --- Text segment ---
            if not seg.text.strip():
                continue

            # Use segment-specific parameters from SSML tags
            seg_lang = seg.language_id or language_id
            seg_exag = seg.exaggeration
            seg_cfg = seg.cfg_weight

            # Update exaggeration if different from current
            from .models.t3.modules.cond_enc import T3Cond
            if float(seg_exag) != float(self.model.conds.t3.emotion_adv[0, 0, 0].item()):
                _cond = self.model.conds.t3
                self.model.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=seg_exag * torch.ones(1, 1, 1),
                ).to(device=device)

            # Tokenize segment text
            text_tokens = self._tokenize_text(seg.text, seg_lang, device)

            # Start T3 for this segment
            token_gen = self._start_t3_stream(
                text_tokens, seg_cfg, temperature,
                repetition_penalty, min_p, top_p,
            )

            # Collect and emit tokens
            segment_tokens = []
            tokens_since_last_emit = 0
            chunk_index = 0

            for token in token_gen:
                token_val = token.view(-1)
                if token_val.item() < SPEECH_VOCAB_SIZE:
                    segment_tokens.append(token_val)
                tokens_since_last_emit += 1

                threshold = self._get_chunk_threshold(chunk_index, stream_state.is_first_chunk)
                if tokens_since_last_emit >= threshold and len(segment_tokens) > 0:
                    audio_chunk = self._emit_chunk(
                        segment_tokens, stream_state,
                        finalize=False, n_cfm_timesteps=n_cfm_timesteps,
                    )
                    if audio_chunk is not None:
                        yield audio_chunk
                    tokens_since_last_emit = 0
                    chunk_index += 1

            # Finalize this segment
            if len(segment_tokens) > 0:
                audio_chunk = self._emit_chunk(
                    segment_tokens, stream_state,
                    finalize=is_last, n_cfm_timesteps=n_cfm_timesteps,
                )
                if audio_chunk is not None:
                    yield audio_chunk

            # Reset mel tracking for next segment (keep HiFiGAN cache)
            if not is_last:
                stream_state.prev_stable_mel_len = 0
                if hasattr(stream_state, 'encoder_caches') and stream_state.encoder_caches is not None:
                    stream_state.encoder_caches = self.model.s3gen.flow.encoder.init_caches(
                        self.model.device, next(self.model.s3gen.parameters()).dtype
                    )
                    stream_state.cached_encoder_output = None
                    stream_state.cached_mel = None

    def _make_silence(self, duration_ms: float) -> Optional[np.ndarray]:
        """Generate silence at native sample rate (24kHz)."""
        if duration_ms <= 0:
            return None
        n_samples = int(duration_ms / 1000.0 * S3GEN_SR)
        return np.zeros(n_samples, dtype=np.float32)

    def _get_chunk_threshold(self, chunk_index: int, is_first_chunk: bool) -> int:
        """Get token threshold for the current chunk.

        With adaptive_chunking=False (default): uses min_initial_tokens for the
        first chunk, then chunk_tokens for all subsequent chunks.

        With adaptive_chunking=True: follows adaptive_schedule for progressive
        chunk sizes, then falls back to chunk_tokens.
        """
        if not self.adaptive_chunking:
            return self.min_initial_tokens if is_first_chunk else self.chunk_tokens

        # Adaptive: follow schedule, then use chunk_tokens
        if chunk_index < len(self.adaptive_schedule):
            return self.adaptive_schedule[chunk_index]
        return self.chunk_tokens

    def _emit_chunk(
        self,
        accumulated_tokens: list,
        stream_state,
        finalize: bool,
        n_cfm_timesteps: Optional[int],
    ) -> Optional[np.ndarray]:
        """Run S3Gen on accumulated tokens and return new audio.

        By default uses efficient streaming: full encoder + windowed CFM on
        [context + new] frames only (O(1) per step). Falls back to full
        reprocessing if efficient_streaming is disabled.

        Speed comes from meanflow (2 ODE steps by design) + windowed CFM.
        """
        all_tokens = torch.cat(accumulated_tokens, dim=0).unsqueeze(0).to(self.model.device)

        if self.efficient_streaming:
            audio_chunk, updated_state = self.model.s3gen.streaming_step_efficient(
                all_tokens=all_tokens,
                ref_dict=self.model.conds.gen,
                state=stream_state,
                finalize=finalize,
                n_cfm_timesteps=n_cfm_timesteps,
                cfm_context_frames=self.cfm_context_frames,
            )
        else:
            audio_chunk, updated_state = self.model.s3gen.streaming_step(
                all_tokens=all_tokens,
                ref_dict=self.model.conds.gen,
                state=stream_state,
                finalize=finalize,
                n_cfm_timesteps=n_cfm_timesteps,
            )

        # Update state in-place (copy all fields from updated state)
        for attr in vars(updated_state):
            setattr(stream_state, attr, getattr(updated_state, attr))

        if audio_chunk is None:
            return None

        chunk_np = audio_chunk.squeeze(0).detach().cpu().numpy()

        # Crossfade with previous chunk's tail to eliminate boundary clicks
        cf = self._crossfade_samples
        if self._crossfade_buffer is not None and len(chunk_np) > cf:
            fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
            fade_out = 1.0 - fade_in
            chunk_np[:cf] = self._crossfade_buffer * fade_out + chunk_np[:cf] * fade_in

        # Hold back tail for next crossfade (unless final chunk)
        if not finalize and len(chunk_np) > cf:
            self._crossfade_buffer = chunk_np[-cf:].copy()
            chunk_np = chunk_np[:-cf]
        else:
            # Final chunk: flush everything, prepend any held-back buffer
            self._crossfade_buffer = None

        self._all_chunks.append(chunk_np)

        # Insert breaths in real-time if humanizer is active
        if self._humanizer is not None:
            chunk_np = self._humanize_chunk(chunk_np, is_final=finalize)

        # Resample if output_sample_rate differs from native 24kHz
        if self._resampler is not None:
            chunk_np = self._resampler.process(chunk_np, finalize=finalize)
            if len(chunk_np) == 0:
                return None

        return chunk_np

    def _humanize_chunk(self, chunk: np.ndarray, is_final: bool = False) -> np.ndarray:
        """Insert breaths into silence gaps within this chunk.

        Analyzes the chunk for silence gaps (≥200ms) and inserts a breath
        if enough speech has accumulated and enough time has passed since
        the last breath. Modifies chunk in-place.

        This runs at 24kHz (before resampling) so it adds negligible latency.

        Args:
            chunk: audio chunk at 24kHz
            is_final: if True, this is the last chunk — don't insert breaths
                in trailing silence (end of utterance, not a pause between phrases)
        """
        import librosa

        sr = S3GEN_SR
        cfg = self._humanizer.config
        chunk_dur = len(chunk) / sr

        # Find speech/silence segments in this chunk
        intervals = librosa.effects.split(chunk, top_db=cfg.silence_top_db)

        if len(intervals) == 0:
            # Entire chunk is silence
            return chunk

        result = chunk.copy()
        chunk_time_start = self._audio_emitted_s

        for i in range(len(intervals)):
            seg_start, seg_end = intervals[i]
            seg_dur = (seg_end - seg_start) / sr
            self._cumulative_speech_s += seg_dur

            # Check gap AFTER this segment
            if i < len(intervals) - 1:
                # Gap between two speech segments within this chunk
                gap_start = seg_end
                gap_end = intervals[i + 1][0]
            else:
                # Last segment in chunk — trailing silence
                # Never insert breath at the end of the final chunk
                # (that's end of utterance, not a pause between phrases)
                if is_final or seg_end >= len(chunk):
                    continue
                gap_start = seg_end
                gap_end = len(chunk)

            gap_dur = (gap_end - gap_start) / sr
            gap_time = chunk_time_start + gap_start / sr

            if gap_dur < cfg.min_gap_s:
                continue

            # Check if gap already has content (T3 natural breath)
            gap_audio = chunk[gap_start:gap_end]
            gap_rms = np.sqrt(np.mean(gap_audio ** 2))
            # Use accumulated RMS across all chunks (not just this chunk)
            # to avoid threshold instability from quiet/loud chunk variation
            if self._rms_n_samples > 0:
                overall_rms = np.sqrt(self._rms_sum_sq / self._rms_n_samples)
            else:
                overall_rms = np.sqrt(np.mean(chunk ** 2))
            if gap_rms > overall_rms * cfg.existing_sound_threshold:
                continue

            # Enough speech before?
            if self._cumulative_speech_s < cfg.min_speech_before_s:
                continue

            # Enough time since last breath?
            if gap_time - self._last_breath_time_s < cfg.min_breath_spacing_s:
                continue

            # Insert breath
            breath_ms = self._humanizer._get_breath_duration_ms(self._cumulative_speech_s)
            breath_samples = int(breath_ms / 1000 * sr)
            padding = int(cfg.breath_padding_ms / 1000 * sr)
            available = gap_end - gap_start - 2 * padding

            if available < breath_samples:
                breath_samples = max(available, int(0.05 * sr))
            if breath_samples <= 0:
                continue

            # Local speech RMS for volume
            ctx_start = max(0, gap_start - int(0.5 * sr))
            local_speech = chunk[ctx_start:gap_start]
            local_rms = np.sqrt(np.mean(local_speech ** 2)) if len(local_speech) > 0 else 0.05
            target_rms = local_rms * cfg.breath_volume_ratio

            breath = self._humanizer.breaths.get_breath(
                duration_ms=int(breath_samples / sr * 1000),
                volume_rms=target_rms,
            )

            # Apply fade
            fade = int(cfg.breath_fade_ms / 1000 * sr)
            if len(breath) > 2 * fade:
                breath[:fade] *= np.linspace(0, 1, fade)
                breath[-fade:] *= np.linspace(1, 0, fade)

            # Place centered in gap
            gap_center = (gap_start + gap_end) // 2
            b_start = gap_center - len(breath) // 2
            b_start = max(gap_start + padding, b_start)
            b_end = b_start + len(breath)
            if b_end > gap_end - padding:
                b_end = gap_end - padding
                b_start = b_end - len(breath)
                if b_start < gap_start + padding:
                    continue

            result[b_start:b_end] = breath[:b_end - b_start]
            self._last_breath_time_s = gap_time
            logger.debug(f"Streaming breath @{gap_time:.2f}s: {breath_ms}ms, "
                         f"after {self._cumulative_speech_s:.1f}s speech")

        # Accumulate RMS across chunks for stable threshold
        self._rms_sum_sq += float(np.sum(chunk ** 2))
        self._rms_n_samples += len(chunk)

        self._audio_emitted_s += chunk_dur
        return result

    def get_full_watermarked(self) -> np.ndarray:
        """Get the full watermarked audio after streaming is complete.

        Call this after the generate_stream() generator is exhausted.
        Watermark is applied at native 24kHz, then resampled to output_sample_rate.

        Returns:
            np.ndarray: full watermarked audio at self.sample_rate Hz
        """
        if not self._all_chunks:
            raise RuntimeError("No audio chunks generated. Call generate_stream() first.")

        # _all_chunks are always at native 24kHz (before resampling)
        full_audio_24k = np.concatenate(self._all_chunks, axis=0)

        # Watermark at native sample rate (24kHz)
        if self._watermarker is None:
            self._watermarker = perth.PerthImplicitWatermarker()
        watermarked_24k = self._watermarker.apply_watermark(full_audio_24k, sample_rate=S3GEN_SR)

        # Resample to output rate if needed
        if self._resampler is not None:
            from scipy.signal import resample_poly
            from math import gcd as _gcd
            g = _gcd(S3GEN_SR, self.sample_rate)
            up = self.sample_rate // g
            down = S3GEN_SR // g
            watermarked = resample_poly(watermarked_24k, up, down).astype(np.float32)
            return watermarked

        return watermarked_24k
