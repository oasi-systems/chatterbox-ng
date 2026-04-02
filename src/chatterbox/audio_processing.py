"""
Audio post-processing for ChatterBox TTS.

Provides LUFS loudness normalization, de-essing, and room tone matching
to produce broadcast-quality, human-indistinguishable speech audio.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def lufs_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Normalize audio loudness to a target LUFS level.

    Args:
        audio: 1D float audio array
        sample_rate: sample rate in Hz
        target_lufs: target loudness in LUFS (default -16.0, broadcast standard for speech)

    Returns:
        Loudness-normalized audio array
    """
    if audio.ndim != 1:
        audio = audio.squeeze()

    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("pyloudnorm not available - loudness normalization skipped")
        return audio

    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio)

    if np.isinf(current_lufs) or np.isnan(current_lufs):
        logger.warning("Could not measure loudness (silent audio?) - skipping normalization")
        return audio

    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    return normalized


def de_ess(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -20.0,
    frequency_range: Tuple[int, int] = (4000, 9000),
    reduction_db: float = 6.0,
) -> np.ndarray:
    """Reduce sibilance ('s', 'sh', 'z' sounds) for more natural speech.

    Uses frequency-targeted dynamic gain reduction: detects energy in the
    sibilance band and applies gain reduction only when it exceeds the threshold.

    Args:
        audio: 1D float audio array
        sample_rate: sample rate in Hz
        threshold_db: sibilance detection threshold in dB (relative to peak)
        frequency_range: (low_hz, high_hz) sibilance frequency band
        reduction_db: maximum gain reduction in dB when sibilance is detected

    Returns:
        De-essed audio array
    """
    if len(audio) < 512:
        return audio

    # Work in short frames for time-domain processing
    frame_size = int(sample_rate * 0.01)  # 10ms frames
    hop_size = frame_size // 2  # 50% overlap

    output = np.copy(audio)
    low_hz, high_hz = frequency_range

    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start:start + frame_size]

        # FFT analysis of this frame
        spectrum = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)

        # Compute energy in sibilance band
        sib_mask = (freqs >= low_hz) & (freqs <= high_hz)
        total_energy = np.sum(np.abs(spectrum) ** 2) + 1e-10
        sib_energy = np.sum(np.abs(spectrum[sib_mask]) ** 2)
        sib_ratio = sib_energy / total_energy

        # Convert to dB and check threshold
        sib_db = 10 * np.log10(sib_ratio + 1e-10)
        if sib_db > threshold_db:
            # Apply proportional reduction
            overshoot = sib_db - threshold_db
            gain_reduction = min(overshoot, reduction_db)
            gain = 10 ** (-gain_reduction / 20.0)

            # Apply gain reduction only in the sibilance band (frequency-selective)
            spectrum_modified = spectrum.copy()
            spectrum_modified[sib_mask] *= gain
            modified_frame = np.fft.irfft(spectrum_modified, n=frame_size)

            # Crossfade with overlap-add
            window = np.hanning(frame_size)
            output[start:start + frame_size] = (
                output[start:start + frame_size] * (1 - window) +
                modified_frame * window
            )

    return output


def match_room_tone(
    audio: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    blend: float = 0.3,
) -> np.ndarray:
    """Match the spectral envelope (room tone) of generated audio to a reference.

    Applies gentle spectral shaping so the generated speech sounds like it was
    recorded in the same acoustic environment as the reference.

    Args:
        audio: 1D float generated audio
        reference: 1D float reference audio (from voice prompt)
        sample_rate: sample rate in Hz
        blend: how much to apply the correction (0.0 = none, 1.0 = full)

    Returns:
        Spectrally shaped audio
    """
    if len(audio) < 1024 or len(reference) < 1024:
        return audio

    # Compute average spectral envelope of reference and generated audio
    fft_size = 2048
    hop = fft_size // 2

    def avg_spectrum(signal):
        n_frames = max(1, (len(signal) - fft_size) // hop)
        window = np.hanning(fft_size)
        acc = np.zeros(fft_size // 2 + 1)
        for i in range(n_frames):
            frame = signal[i * hop:i * hop + fft_size]
            if len(frame) < fft_size:
                frame = np.pad(frame, (0, fft_size - len(frame)))
            acc += np.abs(np.fft.rfft(frame * window)) ** 2
        return np.sqrt(acc / n_frames + 1e-10)

    ref_spectrum = avg_spectrum(reference)
    gen_spectrum = avg_spectrum(audio)

    # Compute correction filter (ratio), clamped to avoid extreme boosts
    correction = ref_spectrum / (gen_spectrum + 1e-10)
    correction = np.clip(correction, 0.5, 2.0)  # max ±6dB correction

    # Smooth the correction filter to avoid artifacts
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    correction = np.convolve(correction, kernel, mode='same')

    # Blend with unity
    correction = 1.0 + blend * (correction - 1.0)

    # Apply via STFT
    window = np.hanning(fft_size)
    output = np.zeros_like(audio)
    norm = np.zeros_like(audio)

    for start in range(0, len(audio) - fft_size, hop):
        frame = audio[start:start + fft_size]
        spectrum = np.fft.rfft(frame * window)
        spectrum *= correction
        modified = np.fft.irfft(spectrum, n=fft_size)
        output[start:start + fft_size] += modified * window
        norm[start:start + fft_size] += window ** 2

    # Normalize overlap-add
    norm = np.maximum(norm, 1e-10)
    output /= norm

    # Handle edges that weren't fully covered
    edge = hop
    if edge < len(audio):
        output[:edge] = audio[:edge]
        output[-edge:] = audio[-edge:]

    return output


def post_process(
    audio: np.ndarray,
    sample_rate: int,
    reference: Optional[np.ndarray] = None,
    target_lufs: float = -16.0,
    de_ess_enabled: bool = True,
    room_tone_enabled: bool = True,
    room_tone_blend: float = 0.3,
) -> np.ndarray:
    """Full post-processing pipeline for TTS output.

    Applies in order:
    1. De-essing (sibilance reduction)
    2. Room tone matching (if reference provided)
    3. LUFS loudness normalization

    Args:
        audio: 1D float audio array
        sample_rate: sample rate in Hz
        reference: optional reference audio for room tone matching
        target_lufs: target loudness in LUFS
        de_ess_enabled: whether to apply de-essing
        room_tone_enabled: whether to apply room tone matching
        room_tone_blend: room tone correction strength (0-1)

    Returns:
        Post-processed audio array
    """
    if audio.ndim != 1:
        audio = audio.squeeze()

    # 1. De-essing first (before any gain changes)
    if de_ess_enabled:
        audio = de_ess(audio, sample_rate)

    # 2. Room tone matching (before loudness normalization)
    if room_tone_enabled and reference is not None:
        if reference.ndim != 1:
            reference = reference.squeeze()
        audio = match_room_tone(audio, reference, sample_rate, blend=room_tone_blend)

    # 3. LUFS normalization last (sets final loudness)
    audio = lufs_normalize(audio, sample_rate, target_lufs=target_lufs)

    return audio
