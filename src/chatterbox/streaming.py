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
import re as _re
from typing import Generator, Tuple, Union, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import perth

from math import gcd

from .models.s3tokenizer import drop_invalid_tokens, SPEECH_VOCAB_SIZE
from .models.s3gen import S3GEN_SR
from .models.t3.modules.cond_enc import T3Cond

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
    """Streaming wrapper for ChatterboxTTS or ChatterboxMultilingualTTS.

    Usage:
        model = ChatterboxTTS.from_pretrained(device)
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
        streaming_cfm_steps: int = 4,
        use_cfm_windowing: bool = True,
        cfm_context_frames: int = 20,
        use_kv_cache: bool = False,
        output_sample_rate: int = None,
    ):
        """
        Args:
            model: ChatterboxTTS, ChatterboxMultilingualTTS, or ChatterboxTurboTTS instance
            chunk_tokens: number of speech tokens to buffer before emitting audio (~40ms per token)
            min_initial_tokens: minimum tokens before first audio emission (higher = better first-chunk quality)
            streaming_cfm_steps: CFM ODE steps for intermediate chunks (fewer = faster, lower quality).
                Final chunk always uses full steps. Set to None to use model default.
            use_cfm_windowing: if True, use CFM context window optimization.
                Re-processes encoder fully (correct for bidirectional) but runs CFM only on
                [context | new] frames with frozen context. Saves ~60-80% CFM cost.
            cfm_context_frames: number of mel frames to freeze as CFM context (default: 20)
            use_kv_cache: if True, use encoder KV-cache + CFM context window.
                Deprecated — encoder KV-cache degrades quality with bidirectional encoder.
                Use use_cfm_windowing instead.
            output_sample_rate: if set, resample output to this rate (e.g. 16000 for Asterisk).
                Uses streaming polyphase resampler with zero boundary artifacts.
                Default None = output at native 24kHz.
        """
        self.model = model
        self.chunk_tokens = chunk_tokens
        self.min_initial_tokens = min_initial_tokens
        # With meanflow (2 ODE steps), intermediate chunks don't need reduced steps
        if streaming_cfm_steps is not None and hasattr(model, 's3gen') and model.s3gen.meanflow:
            self.streaming_cfm_steps = None  # let flow_inference use its 2-step default
        else:
            self.streaming_cfm_steps = streaming_cfm_steps
        self.use_cfm_windowing = use_cfm_windowing
        self.cfm_context_frames = cfm_context_frames
        self.use_kv_cache = use_kv_cache

        if output_sample_rate and output_sample_rate != S3GEN_SR:
            self._resampler = StreamingResampler(S3GEN_SR, output_sample_rate)
            self.sample_rate = output_sample_rate
        else:
            self._resampler = None
            self.sample_rate = S3GEN_SR
        self._all_chunks = []
        if self._resampler is not None:
            self._resampler.reset()
        self._watermarker = perth.PerthImplicitWatermarker()

    def _is_multilingual(self):
        return hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'cangjie_converter')

    def _is_turbo(self):
        return hasattr(self.model, 't3') and self.model.t3.is_gpt

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

    def _start_t3_stream(self, text_tokens, is_turbo, cfg_weight, temperature,
                         repetition_penalty, min_p, top_p, top_k):
        """Start T3 token generator for given text tokens."""
        if not is_turbo and cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        if is_turbo:
            return self.model.t3.inference_turbo_streaming(
                t3_cond=self.model.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
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
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        # Turbo-specific params
        top_k: int = 1000,
        # S3Gen params
        n_cfm_timesteps: Optional[int] = None,
        # Pipelining splits text into sentences with independent T3 passes.
        # Disabled by default to preserve prosody continuity.
        sentence_pipelining: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks as speech tokens are generated.

        Args:
            sentence_pipelining: if True, split text into sentences and process
                each independently through T3 while maintaining audio continuity.
                Reduces latency for long texts and improves stability.

        Yields:
            np.ndarray: audio chunk (1D float array at self.sample_rate Hz).
                        Chunks are NOT watermarked — call get_full_watermarked() after streaming.
        """
        if sentence_pipelining:
            yield from self._generate_stream_pipelined(
                text=text, audio_prompt_path=audio_prompt_path, language_id=language_id,
                temperature=temperature, repetition_penalty=repetition_penalty,
                min_p=min_p, top_p=top_p, cfg_weight=cfg_weight, exaggeration=exaggeration,
                top_k=top_k, n_cfm_timesteps=n_cfm_timesteps,
            )
            return

        self._all_chunks = []
        if self._resampler is not None:
            self._resampler.reset()
        is_turbo = self._is_turbo()
        device = self.model.device

        # --- Prepare conditionals ---
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.model.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        # --- Tokenize text ---
        text_tokens = self._tokenize_text(text, language_id, device)

        # --- Initialize streaming state ---
        stream_state = self.model.s3gen.init_streaming()
        accumulated_tokens = []

        # --- Start T3 streaming ---
        token_gen = self._start_t3_stream(
            text_tokens, is_turbo, cfg_weight, temperature,
            repetition_penalty, min_p, top_p, top_k,
        )

        # --- Stream tokens and emit audio chunks ---
        tokens_since_last_emit = 0

        for token in token_gen:
            token_val = token.view(-1)
            if token_val.item() < SPEECH_VOCAB_SIZE:
                accumulated_tokens.append(token_val)
            tokens_since_last_emit += 1

            threshold = self.min_initial_tokens if stream_state.is_first_chunk else self.chunk_tokens
            if tokens_since_last_emit >= threshold and len(accumulated_tokens) > 0:
                audio_chunk = self._emit_chunk(accumulated_tokens, stream_state, finalize=False, n_cfm_timesteps=n_cfm_timesteps)
                if audio_chunk is not None:
                    yield audio_chunk
                tokens_since_last_emit = 0

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
        top_k: int,
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
        if self._resampler is not None:
            self._resampler.reset()
        is_turbo = self._is_turbo()
        device = self.model.device

        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.model.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        sentences = _split_sentences(text)
        logger.info(f"Sentence pipelining: {len(sentences)} sentence(s)")

        # Shared S3Gen state across all sentences for audio continuity
        stream_state = self.model.s3gen.init_streaming()

        for sent_idx, sentence in enumerate(sentences):
            is_last_sentence = (sent_idx == len(sentences) - 1)

            # Tokenize this sentence
            text_tokens = self._tokenize_text(sentence, language_id, device)

            # Start T3 for this sentence
            token_gen = self._start_t3_stream(
                text_tokens, is_turbo, cfg_weight, temperature,
                repetition_penalty, min_p, top_p, top_k,
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

    def _emit_chunk(
        self,
        accumulated_tokens: list,
        stream_state,
        finalize: bool,
        n_cfm_timesteps: Optional[int],
    ) -> Optional[np.ndarray]:
        """Run S3Gen on accumulated tokens and return new audio.

        Three modes (selected by constructor flags):
        1. use_cfm_windowing=True (default): Full encoder + CFM context window.
           Correct bidirectional encoder, saves ~60-80% CFM cost.
        2. use_kv_cache=True: Encoder KV-cache + CFM context window.
           Fastest but degrades quality (stale KV from bidirectional encoder).
        3. Neither: Full reprocessing (O(N²) encoder + O(N) CFM). Slowest but safest.

        Uses reduced ODE steps for intermediate chunks (streaming_cfm_steps)
        and full steps for the final chunk, balancing latency and quality.
        """
        all_tokens = torch.cat(accumulated_tokens, dim=0).unsqueeze(0).to(self.model.device)

        # Use fewer ODE steps for intermediate chunks to reduce latency
        effective_steps = n_cfm_timesteps
        if effective_steps is None and not finalize and self.streaming_cfm_steps is not None:
            effective_steps = self.streaming_cfm_steps

        if self.use_kv_cache and hasattr(stream_state, 'encoder_caches') and stream_state.encoder_caches is not None:
            # Legacy: encoder KV-cache + CFM window (deprecated, quality issues)
            audio_chunk, updated_state = self.model.s3gen.streaming_step_cached(
                all_tokens=all_tokens,
                ref_dict=self.model.conds.gen,
                state=stream_state,
                finalize=finalize,
                n_cfm_timesteps=effective_steps,
                context_frames=self.cfm_context_frames,
            )
        elif self.use_cfm_windowing:
            # Hybrid: full encoder + CFM context window (recommended)
            audio_chunk, updated_state = self.model.s3gen.streaming_step_windowed(
                all_tokens=all_tokens,
                ref_dict=self.model.conds.gen,
                state=stream_state,
                finalize=finalize,
                n_cfm_timesteps=effective_steps,
                context_frames=self.cfm_context_frames,
            )
        else:
            # Fallback: full reprocessing (slowest, highest quality)
            audio_chunk, updated_state = self.model.s3gen.streaming_step(
                all_tokens=all_tokens,
                ref_dict=self.model.conds.gen,
                state=stream_state,
                finalize=finalize,
                n_cfm_timesteps=effective_steps,
            )

        # Update state in-place (copy all fields from updated state)
        for attr in vars(updated_state):
            setattr(stream_state, attr, getattr(updated_state, attr))

        if audio_chunk is None:
            return None

        chunk_np = audio_chunk.squeeze(0).detach().cpu().numpy()
        self._all_chunks.append(chunk_np)

        # Resample if output_sample_rate differs from native 24kHz
        if self._resampler is not None:
            chunk_np = self._resampler.process(chunk_np, finalize=finalize)
            if len(chunk_np) == 0:
                return None

        return chunk_np

    def get_full_watermarked(self) -> np.ndarray:
        """Get the full watermarked audio after streaming is complete.

        Call this after the generate_stream() generator is exhausted.

        Returns:
            np.ndarray: full watermarked audio at self.sample_rate Hz
        """
        if not self._all_chunks:
            raise RuntimeError("No audio chunks generated. Call generate_stream() first.")

        full_audio = np.concatenate(self._all_chunks, axis=0)
        watermarked = self._watermarker.apply_watermark(full_audio, sample_rate=self.sample_rate)
        return watermarked
