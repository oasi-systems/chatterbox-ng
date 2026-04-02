"""Tests for audio post-processing pipeline."""
import numpy as np
import pytest
import importlib.util
import os

# Direct import to avoid full package init
_ap_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "chatterbox", "audio_processing.py"
)
spec = importlib.util.spec_from_file_location("audio_processing", os.path.abspath(_ap_path))
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)


SR = 24000


def _make_speech_signal(duration=1.0, sr=SR):
    """Create a synthetic speech-like signal."""
    np.random.seed(42)
    t = np.linspace(0, duration, int(sr * duration))
    # Mix of frequencies simulating speech formants + sibilance
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +   # F1-like
        0.15 * np.sin(2 * np.pi * 500 * t) +   # F2-like
        0.08 * np.sin(2 * np.pi * 2500 * t) +  # F3-like
        0.1 * np.sin(2 * np.pi * 6000 * t) +   # sibilance
        0.02 * np.random.randn(len(t))          # noise floor
    )
    return signal.astype(np.float64)


class TestDeEss:
    def test_output_shape(self):
        audio = _make_speech_signal()
        result = ap.de_ess(audio, SR)
        assert result.shape == audio.shape

    def test_short_audio(self):
        # Should handle very short audio gracefully
        short = np.random.randn(100).astype(np.float64)
        result = ap.de_ess(short, SR)
        np.testing.assert_array_equal(result, short)

    def test_reduces_sibilance(self):
        # Create signal with strong sibilance
        t = np.linspace(0, 0.5, SR // 2)
        sibilant = np.sin(2 * np.pi * 6000 * t) * 0.5
        result = ap.de_ess(sibilant, SR, threshold_db=-30.0)
        # RMS of result should be less than input (sibilance reduced)
        assert np.sqrt(np.mean(result**2)) <= np.sqrt(np.mean(sibilant**2))

    def test_preserves_low_freq(self):
        # Pure low-frequency signal should be mostly unchanged
        t = np.linspace(0, 1.0, SR)
        low_freq = np.sin(2 * np.pi * 200 * t) * 0.3
        result = ap.de_ess(low_freq, SR)
        # Correlation should be very high
        corr = np.corrcoef(low_freq, result)[0, 1]
        assert corr > 0.95


class TestMatchRoomTone:
    def test_output_shape(self):
        audio = _make_speech_signal()
        ref = _make_speech_signal()
        result = ap.match_room_tone(audio, ref, SR)
        assert result.shape == audio.shape

    def test_short_audio_passthrough(self):
        short = np.random.randn(500).astype(np.float64)
        ref = np.random.randn(500).astype(np.float64)
        result = ap.match_room_tone(short, ref, SR)
        np.testing.assert_array_equal(result, short)

    def test_identical_signals(self):
        audio = _make_speech_signal()
        result = ap.match_room_tone(audio, audio, SR, blend=1.0)
        # Should be very similar when matching to itself
        corr = np.corrcoef(audio[:len(result)], result[:len(audio)])[0, 1]
        assert corr > 0.9

    def test_blend_zero(self):
        audio = _make_speech_signal()
        ref = _make_speech_signal(duration=0.5)
        result = ap.match_room_tone(audio, ref, SR, blend=0.0)
        # With zero blend, output should be nearly identical to input
        corr = np.corrcoef(audio, result)[0, 1]
        assert corr > 0.99


class TestLufsNormalize:
    def test_output_shape(self):
        audio = _make_speech_signal()
        result = ap.lufs_normalize(audio, SR)
        assert result.shape == audio.shape

    def test_silent_audio(self):
        silent = np.zeros(SR)
        result = ap.lufs_normalize(silent, SR)
        np.testing.assert_array_equal(result, silent)

    def test_2d_input(self):
        audio = _make_speech_signal().reshape(1, -1)
        result = ap.lufs_normalize(audio, SR)
        assert result.ndim == 1


class TestPostProcess:
    def test_full_pipeline(self):
        audio = _make_speech_signal()
        ref = _make_speech_signal(duration=0.5)
        result = ap.post_process(audio, SR, reference=ref)
        assert result.shape == audio.shape

    def test_no_reference(self):
        audio = _make_speech_signal()
        result = ap.post_process(audio, SR)
        assert result.shape == audio.shape

    def test_disabled_steps(self):
        audio = _make_speech_signal()
        result = ap.post_process(
            audio, SR,
            de_ess_enabled=False,
            room_tone_enabled=False,
        )
        assert result.shape == audio.shape

    def test_2d_input(self):
        audio = _make_speech_signal().reshape(1, -1)
        result = ap.post_process(audio, SR)
        assert result.ndim == 1
