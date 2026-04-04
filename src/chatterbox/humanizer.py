"""
Voice Humanizer — inject life into synthesized speech.

Inserts real breath sounds into silence gaps between speech segments,
making TTS output sound like a living human being.

Principles:
    - Breaths go ONLY inside existing silence gaps (never cut speech)
    - Breath duration is proportional to preceding speech length
    - Breath volume is ~8% of surrounding speech RMS
    - No breath at the start, no breath mid-word

Usage:
    humanizer = VoiceHumanizer.from_reference("reference_voice.wav")
    humanized = humanizer.process(tts_audio)

    # Or in streaming:
    streamer = ChatterboxStreamingTTS(model)
    for chunk in streamer.generate_stream(text="...", language_id="it"):
        send(chunk)
    full_audio = np.concatenate(streamer._all_chunks)
    humanized = humanizer.process(full_audio)
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)

DEFAULT_SR = 24000


@dataclass
class HumanizerConfig:
    """Configuration for voice humanization."""
    # Breath volume as fraction of local speech RMS
    breath_volume_ratio: float = 0.08

    # Minimum gap duration (seconds) to consider for breath insertion
    # 200ms is the minimum for an audible breath with padding on both sides
    min_gap_s: float = 0.20

    # Minimum total speech before a gap to insert a breath (seconds)
    # Prevents breathing after very short phrases
    min_speech_before_s: float = 1.5

    # Breath duration rules (ms) based on speech length before the gap
    # (max_speech_s, breath_ms_min, breath_ms_max)
    breath_duration_rules: tuple = (
        (2.0, 80, 120),    # <2s speech → short breath
        (4.0, 130, 180),   # 2-4s speech → medium breath
        (999, 180, 280),   # >4s speech → full breath
    )

    # Maximum breaths to insert (prevents over-breathing)
    max_breaths: int = 5

    # Minimum seconds between two breaths
    min_breath_spacing_s: float = 2.0

    # If a gap's RMS exceeds this fraction of overall speech RMS,
    # assume the T3 model already generated a natural breath there — skip it.
    # 0.10 = only skip gaps with significant content (>10% of speech energy)
    existing_sound_threshold: float = 0.10

    # Silence detection threshold (dB below peak)
    silence_top_db: int = 25

    # Fade durations for smooth insertion
    breath_fade_ms: float = 8.0

    # Padding silence around breath (ms)
    breath_padding_ms: float = 15.0

    sample_rate: int = DEFAULT_SR


class BreathLibrary:
    """Breath samples adapted to a target speaker's vocal profile.

    Uses real breath templates (curated, shipped with the package) and adapts
    their spectral profile to match the target speaker. This way:
    - Breath structure (timing, dynamics) comes from real recordings
    - Spectral color (timbre) matches the target voice
    - Works with any voice without re-extracting breaths
    """

    # Default breath templates directory (shipped with package)
    _TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'breath_templates')

    def __init__(self, sample_rate: int = DEFAULT_SR):
        self.breaths: List[np.ndarray] = []
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng()

    @classmethod
    def from_reference(cls, reference_path: str, templates_dir: str = None,
                       sample_rate: int = DEFAULT_SR) -> 'BreathLibrary':
        """Create breath library adapted to a speaker's voice.

        Loads breath templates and adapts their spectral profile to match
        the reference audio's vocal characteristics.

        Args:
            reference_path: path to reference voice audio (same used for TTS cloning)
            templates_dir: directory with breath template WAVs. If None, uses
                built-in templates shipped with the package.
            sample_rate: target sample rate
        """
        lib = cls(sample_rate=sample_rate)
        templates_dir = templates_dir or cls._TEMPLATES_DIR

        # Load reference and extract speaker spectral profile
        ref_audio, sr = librosa.load(reference_path, sr=sample_rate)
        speaker_profile = cls._extract_speaker_profile(ref_audio, sr)

        # Load and adapt templates
        if os.path.isdir(templates_dir):
            for fname in sorted(os.listdir(templates_dir)):
                if fname.endswith('.wav') and 'breath' in fname.lower():
                    path = os.path.join(templates_dir, fname)
                    template, _ = librosa.load(path, sr=sample_rate)
                    adapted = cls._adapt_breath(template, speaker_profile, sr)
                    lib.breaths.append(adapted)
            logger.info(f"Loaded {len(lib.breaths)} breath templates from {templates_dir}, "
                        f"adapted to {reference_path}")
        else:
            logger.warning(f"Templates directory not found: {templates_dir}. "
                           f"Falling back to extraction from reference.")
            lib = cls.from_audio(reference_path, sample_rate=sample_rate)

        return lib

    @classmethod
    def from_audio(cls, audio_path: str, sample_rate: int = DEFAULT_SR) -> 'BreathLibrary':
        """Extract breath samples directly from an audio file.

        Fallback method when templates are not available. Uses strict filtering:
        - Spectral centroid > 2000Hz (breath = high-frequency air noise)
        - Zero voicing (no pitched content)
        - Zero crossing rate > 0.08 (air turbulence oscillations)
        """
        lib = cls(sample_rate=sample_rate)
        y, sr = librosa.load(audio_path, sr=sample_rate)
        logger.info(f"Extracting breaths from {audio_path} ({len(y)/sr:.1f}s)")

        intervals = librosa.effects.split(y, top_db=20)

        candidates = []
        for i in range(1, len(intervals)):
            gap_start = intervals[i - 1][1]
            gap_end = intervals[i][0]
            gap_dur = (gap_end - gap_start) / sr

            if not (0.12 < gap_dur < 0.6):
                continue

            gap_audio = y[gap_start:gap_end]
            rms = np.sqrt(np.mean(gap_audio ** 2))
            if rms < 0.003:
                continue

            centroid = librosa.feature.spectral_centroid(y=gap_audio, sr=sr)[0]
            mean_centroid = np.mean(centroid)
            if mean_centroid < 2000:
                continue

            f0, _, _ = librosa.pyin(gap_audio, fmin=60, fmax=400, sr=sr)
            voiced_ratio = np.sum(~np.isnan(f0)) / max(len(f0), 1)
            if voiced_ratio > 0.1:
                continue

            zcr = np.mean(librosa.feature.zero_crossing_rate(gap_audio)[0])
            if zcr < 0.08:
                continue

            fade = int(0.005 * sr)
            audio = gap_audio.copy()
            if len(audio) > 2 * fade:
                audio[:fade] *= np.linspace(0, 1, fade)
                audio[-fade:] *= np.linspace(1, 0, fade)

            candidates.append({
                'audio': audio,
                'rms': rms,
                'centroid': mean_centroid,
            })

        candidates.sort(key=lambda c: c['centroid'], reverse=True)
        lib.breaths = [c['audio'] for c in candidates[:20]]
        logger.info(f"Extracted {len(lib.breaths)} breath samples")
        return lib

    @classmethod
    def from_directory(cls, breath_dir: str, sample_rate: int = DEFAULT_SR) -> 'BreathLibrary':
        """Load pre-extracted breath samples from a directory (no adaptation)."""
        lib = cls(sample_rate=sample_rate)
        for fname in sorted(os.listdir(breath_dir)):
            if fname.endswith('.wav') and 'breath' in fname.lower():
                path = os.path.join(breath_dir, fname)
                audio, _ = librosa.load(path, sr=sample_rate)
                lib.breaths.append(audio)
        logger.info(f"Loaded {len(lib.breaths)} breaths from {breath_dir}")
        return lib

    @staticmethod
    def _extract_speaker_profile(audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract average spectral profile from speech segments."""
        intervals = librosa.effects.split(audio, top_db=25)
        if len(intervals) == 0:
            return None
        speech = np.concatenate([audio[s:e] for s, e in intervals])
        speech_stft = librosa.stft(speech, n_fft=1024)
        return np.mean(np.abs(speech_stft), axis=1, keepdims=True)

    @staticmethod
    def _adapt_breath(template: np.ndarray, speaker_profile: np.ndarray,
                      sr: int, blend: float = 0.3) -> np.ndarray:
        """Adapt a breath template to a speaker's spectral profile.

        Keeps the temporal structure of the breath (real dynamics) but
        shifts spectral coloring toward the target speaker's voice.

        Args:
            template: breath audio template
            speaker_profile: speaker's average spectral magnitude (from _extract_speaker_profile)
            sr: sample rate
            blend: how much speaker coloring to apply (0=none, 1=full). Default 0.3.
        """
        if speaker_profile is None:
            return template.copy()

        from scipy.ndimage import gaussian_filter1d

        breath_stft = librosa.stft(template, n_fft=1024)
        breath_mag = np.abs(breath_stft)
        breath_phase = np.angle(breath_stft)

        # Spectral ratio: where speaker has more energy, boost breath
        breath_profile = np.mean(breath_mag, axis=1, keepdims=True)
        ratio = speaker_profile / (breath_profile + 1e-10)
        ratio = np.clip(ratio, 0.3, 3.0)
        ratio_smooth = gaussian_filter1d(ratio.squeeze(), sigma=5).reshape(-1, 1)

        # Blend original breath with speaker-colored version
        adapted_mag = breath_mag * (1 - blend + blend * ratio_smooth)

        # Reconstruct with original phase
        adapted_stft = adapted_mag * np.exp(1j * breath_phase)
        adapted = librosa.istft(adapted_stft, length=len(template))

        # Preserve original RMS
        rms_orig = np.sqrt(np.mean(template ** 2))
        rms_new = np.sqrt(np.mean(adapted ** 2))
        if rms_new > 0:
            adapted *= rms_orig / rms_new

        return adapted.astype(np.float32)

    def get_breath(self, duration_ms: int, volume_rms: float) -> np.ndarray:
        """Get a breath sample trimmed to target duration and normalized to target RMS.

        Args:
            duration_ms: target breath duration in milliseconds
            volume_rms: target RMS level for the breath

        Returns:
            Breath audio array at the correct duration and volume.
        """
        if not self.breaths:
            logger.warning("No breath samples available")
            return np.array([], dtype=np.float32)

        idx = self._rng.integers(0, len(self.breaths))
        breath = self.breaths[idx].copy()

        # Trim to target duration
        target_samples = int(duration_ms / 1000 * self.sample_rate)
        if len(breath) > target_samples:
            breath = breath[:target_samples]
        elif len(breath) < target_samples:
            # Pad with silence if breath is shorter
            breath = np.pad(breath, (0, target_samples - len(breath)))

        # Normalize volume
        current_rms = np.sqrt(np.mean(breath ** 2))
        if current_rms > 0:
            breath *= volume_rms / current_rms

        return breath.astype(np.float32)


class VoiceHumanizer:
    """Post-processor that adds breathing to synthesized speech.

    Finds real silence gaps in TTS audio and inserts breath sounds
    inside them, without altering speech timing.
    """

    def __init__(self, breath_library: BreathLibrary, config: HumanizerConfig = None):
        self.breaths = breath_library
        self.config = config or HumanizerConfig()
        self._rng = np.random.default_rng()

    @classmethod
    def from_reference(cls, audio_path: str, config: HumanizerConfig = None,
                       templates_dir: str = None) -> 'VoiceHumanizer':
        """Create humanizer adapted to a speaker's voice.

        Uses breath templates adapted to the speaker's spectral profile.
        If templates are not available, falls back to extraction from reference.

        Args:
            audio_path: reference voice audio (same used for TTS cloning)
            config: humanizer configuration
            templates_dir: directory with breath template WAVs (optional)
        """
        cfg = config or HumanizerConfig()
        lib = BreathLibrary.from_reference(audio_path, templates_dir=templates_dir,
                                           sample_rate=cfg.sample_rate)
        return cls(lib, cfg)

    @classmethod
    def from_breath_dir(cls, breath_dir: str, config: HumanizerConfig = None) -> 'VoiceHumanizer':
        """Create humanizer from pre-extracted breath samples."""
        cfg = config or HumanizerConfig()
        lib = BreathLibrary.from_directory(breath_dir, sample_rate=cfg.sample_rate)
        return cls(lib, cfg)

    def process(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Humanize synthesized audio by inserting breaths in silence gaps.

        The output has the same duration as the input — breaths are placed
        inside existing silence, never extending or cutting speech.

        Args:
            audio: 1D float numpy array of synthesized speech
            sample_rate: audio sample rate (default: config.sample_rate)

        Returns:
            Humanized audio (same length as input)
        """
        sr = sample_rate or self.config.sample_rate
        cfg = self.config

        if not self.breaths.breaths:
            logger.warning("No breath samples loaded — returning original audio")
            return audio

        # Step 1: Find silence gaps between speech segments
        gaps = self._find_gaps(audio, sr)
        if not gaps:
            logger.debug("No gaps found for breath insertion")
            return audio

        # Step 2: Select which gaps get breaths
        selected = self._select_breath_points(gaps, audio, sr)
        if not selected:
            logger.debug("No eligible gaps for breath insertion")
            return audio

        # Step 3: Insert breaths inside selected gaps
        result = audio.copy()
        for gap in selected:
            self._insert_breath_in_gap(result, gap, sr)

        logger.info(f"Inserted {len(selected)} breath(s)")
        return result

    def _find_gaps(self, audio: np.ndarray, sr: int) -> List[dict]:
        """Find silence gaps between speech segments.

        Each gap is tagged with its RMS energy so we can detect whether
        the T3 model already generated a natural breath sound there.
        """
        intervals = librosa.effects.split(audio, top_db=self.config.silence_top_db)

        gaps = []
        cumulative_speech = 0.0

        for i in range(1, len(intervals)):
            # Speech duration of the segment that just ended
            seg_dur = (intervals[i - 1][1] - intervals[i - 1][0]) / sr
            cumulative_speech += seg_dur

            gap_start = intervals[i - 1][1]
            gap_end = intervals[i][0]
            gap_dur = (gap_end - gap_start) / sr

            if gap_dur >= self.config.min_gap_s:
                # Measure energy inside gap to detect existing sounds
                gap_audio = audio[gap_start:gap_end]
                gap_rms = np.sqrt(np.mean(gap_audio ** 2)) if len(gap_audio) > 0 else 0.0

                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_dur,
                    'speech_before': cumulative_speech,
                    'time': gap_start / sr,
                    'gap_rms': gap_rms,
                })

        return gaps

    def _select_breath_points(self, gaps: List[dict], audio: np.ndarray, sr: int) -> List[dict]:
        """Select which gaps should receive a breath.

        Skips gaps where the T3 model already generated a natural breath
        or other sound (detected by RMS energy above threshold).
        """
        cfg = self.config
        selected = []
        last_breath_time = -999.0

        # Compute reference RMS over full audio (including silence).
        # This gives a lower threshold that produces more natural, subtle breaths.
        speech_rms = np.sqrt(np.mean(audio ** 2))

        for gap in gaps:
            # Skip if gap already has content (T3 generated a natural breath)
            # Threshold: if gap RMS > 3% of overall speech RMS, something is there
            if gap['gap_rms'] > speech_rms * cfg.existing_sound_threshold:
                logger.debug(
                    f"Gap @{gap['time']:.2f}s already has sound "
                    f"(RMS={gap['gap_rms']:.4f} > {speech_rms * cfg.existing_sound_threshold:.4f}), skipping"
                )
                continue

            # Skip if not enough speech before
            if gap['speech_before'] < cfg.min_speech_before_s:
                continue

            # Skip if too close to last breath
            if gap['time'] - last_breath_time < cfg.min_breath_spacing_s:
                continue

            # Skip if we've reached max breaths
            if len(selected) >= cfg.max_breaths:
                break

            selected.append(gap)
            last_breath_time = gap['time']

        return selected

    def _get_breath_duration_ms(self, speech_before_s: float) -> int:
        """Get breath duration based on preceding speech length."""
        for max_speech, ms_min, ms_max in self.config.breath_duration_rules:
            if speech_before_s < max_speech:
                return self._rng.integers(ms_min, ms_max + 1)
        # Fallback
        return self._rng.integers(180, 280)

    def _insert_breath_in_gap(self, audio: np.ndarray, gap: dict, sr: int):
        """Insert a breath sound inside an existing silence gap (in-place).

        The breath is placed centered in the gap, with padding on both sides.
        """
        cfg = self.config
        fade = int(cfg.breath_fade_ms / 1000 * sr)
        padding = int(cfg.breath_padding_ms / 1000 * sr)

        # Determine breath duration
        breath_ms = self._get_breath_duration_ms(gap['speech_before'])
        breath_samples = int(breath_ms / 1000 * sr)

        # Available space in gap (minus padding on both sides)
        available = gap['end'] - gap['start'] - 2 * padding
        if available < breath_samples:
            # Trim breath to fit
            breath_samples = max(available, int(0.05 * sr))

        # Get local speech RMS for volume normalization
        ctx_start = max(0, gap['start'] - int(0.5 * sr))
        local_speech = audio[ctx_start:gap['start']]
        local_rms = np.sqrt(np.mean(local_speech ** 2)) if len(local_speech) > 0 else 0.05
        target_rms = local_rms * cfg.breath_volume_ratio

        # Get breath
        breath = self.breaths.get_breath(
            duration_ms=int(breath_samples / sr * 1000),
            volume_rms=target_rms,
        )

        # Apply fade in/out
        if len(breath) > 2 * fade:
            breath[:fade] *= np.linspace(0, 1, fade)
            breath[-fade:] *= np.linspace(1, 0, fade)

        # Place centered in gap
        gap_center = (gap['start'] + gap['end']) // 2
        b_start = gap_center - len(breath) // 2

        # Ensure we stay inside the gap with padding
        b_start = max(gap['start'] + padding, b_start)
        b_end = b_start + len(breath)
        if b_end > gap['end'] - padding:
            b_end = gap['end'] - padding
            b_start = b_end - len(breath)
            if b_start < gap['start'] + padding:
                # Gap too small, skip
                logger.debug(f"Gap @{gap['time']:.2f}s too small for breath")
                return

        # Insert breath (overwrite silence with breath sound)
        audio[b_start:b_end] = breath[:b_end - b_start]

        logger.debug(
            f"Breath @{b_start/sr:.2f}s: {breath_ms}ms, "
            f"after {gap['speech_before']:.1f}s speech, "
            f"RMS={target_rms:.4f}"
        )
