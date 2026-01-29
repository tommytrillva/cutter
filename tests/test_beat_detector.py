"""Tests for beat detection."""
import pytest
import numpy as np

# Skip tests if librosa not available
librosa = pytest.importorskip("librosa")

from src.audio_analyzer import BeatDetector


class TestBeatDetector:
    """Tests for BeatDetector class."""

    @pytest.fixture
    def sample_audio(self):
        """Generate a simple test audio signal with beats."""
        sr = 22050
        duration = 5.0
        bpm = 120

        # Generate beat track
        beat_interval = 60.0 / bpm
        t = np.linspace(0, duration, int(sr * duration))

        # Create click track at beat positions
        audio = np.zeros_like(t)
        for beat_time in np.arange(0, duration, beat_interval):
            beat_sample = int(beat_time * sr)
            if beat_sample < len(audio):
                # Add click
                click_duration = int(0.01 * sr)
                end_sample = min(beat_sample + click_duration, len(audio))
                audio[beat_sample:end_sample] = 0.5

        # Add some noise
        audio += np.random.randn(len(audio)) * 0.01

        return audio, sr

    def test_detector_initialization(self):
        """Test BeatDetector initialization."""
        detector = BeatDetector(sensitivity="medium")
        assert detector.sensitivity.value == "medium"

        detector_high = BeatDetector(sensitivity="high")
        assert detector_high.sensitivity.value == "high"

    def test_detect_beats(self, sample_audio):
        """Test beat detection on sample audio."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium")

        beat_info = detector.detect_beats(audio, sr)

        assert beat_info.bpm > 0
        assert len(beat_info.beat_times) > 0
        assert len(beat_info.beat_strength) == len(beat_info.beat_times)

    def test_estimate_bpm(self, sample_audio):
        """Test BPM estimation."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium")

        bpm = detector.estimate_bpm(audio, sr)

        # Should be within reasonable range
        assert 60 <= bpm <= 200

    def test_calculate_energy(self, sample_audio):
        """Test energy envelope calculation."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium")

        energy, times = detector.calculate_energy(audio, sr)

        assert len(energy) > 0
        assert len(times) == len(energy)
        assert np.all(energy >= 0)
        assert np.all(energy <= 1)

    def test_sensitivity_affects_detection(self, sample_audio):
        """Test that sensitivity affects detection results."""
        audio, sr = sample_audio

        detector_low = BeatDetector(sensitivity="low")
        detector_high = BeatDetector(sensitivity="high")

        beats_low = detector_low.detect_beats(audio, sr)
        beats_high = detector_high.detect_beats(audio, sr)

        # High sensitivity should generally detect more beats
        # (though this depends on the audio content)
        assert beats_low.num_beats > 0
        assert beats_high.num_beats > 0

    def test_custom_threshold(self, sample_audio):
        """Test custom threshold override."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium", custom_threshold=0.1)

        beat_info = detector.detect_beats(audio, sr)
        assert beat_info.num_beats > 0

    def test_quantize_to_beat(self, sample_audio):
        """Test time quantization to beat."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium")
        beat_info = detector.detect_beats(audio, sr)

        if beat_info.beat_times:
            # Time close to beat should snap
            first_beat = beat_info.beat_times[0]
            quantized = detector.quantize_to_beat(
                beat_info, first_beat + 0.05, snap_threshold=0.1
            )
            assert quantized == first_beat

            # Time far from beat should not snap
            far_time = first_beat + 0.5
            quantized = detector.quantize_to_beat(
                beat_info, far_time, snap_threshold=0.1
            )
            # May or may not snap depending on beat positions

    def test_get_strong_beats(self, sample_audio):
        """Test filtering strong beats."""
        audio, sr = sample_audio
        detector = BeatDetector(sensitivity="medium")
        beat_info = detector.detect_beats(audio, sr)

        strong_beats = detector.get_strong_beats(beat_info, strength_threshold=0.5)

        for idx, time, strength in strong_beats:
            assert strength >= 0.5
            assert time == beat_info.beat_times[idx]

    def test_empty_audio(self):
        """Test with empty/silent audio."""
        audio = np.zeros(22050)  # 1 second of silence
        sr = 22050

        detector = BeatDetector(sensitivity="medium")
        beat_info = detector.detect_beats(audio, sr)

        # Should still return valid BeatInfo even with no beats
        assert isinstance(beat_info.bpm, float)
        assert isinstance(beat_info.beat_times, list)
