"""Tests for shot quality scoring."""
import pytest
import numpy as np

from src.shot_quality import QualityScorer, MotionDetector, CompositionAnalyzer
from src.data_structures import FrameQuality, Shot


class TestMotionDetector:
    """Tests for MotionDetector class."""

    def test_initialization(self):
        """Test MotionDetector initialization."""
        detector = MotionDetector()
        assert detector.flow_method == "farneback"

        detector_lk = MotionDetector(flow_method="lucas_kanade")
        assert detector_lk.flow_method == "lucas_kanade"

    def test_no_motion(self):
        """Test motion detection with identical frames."""
        detector = MotionDetector()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        motion_score = detector.calculate_motion(frame, frame)

        # Identical frames should have very low motion
        assert motion_score < 0.1

    def test_high_motion(self):
        """Test motion detection with different frames."""
        detector = MotionDetector()

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255

        motion_score = detector.calculate_motion(frame2, frame1)

        # Very different frames should have high motion
        assert motion_score > 0

    def test_scene_change_detection(self):
        """Test scene change detection."""
        detector = MotionDetector()

        # Same scene
        frame1 = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        frame2 = frame1 + np.random.randint(0, 10, frame1.shape, dtype=np.uint8)
        frame2 = np.clip(frame2, 0, 255).astype(np.uint8)

        is_change, diff = detector.detect_scene_change(frame2, frame1)
        assert not is_change

        # Different scene
        frame3 = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)
        is_change, diff = detector.detect_scene_change(frame3, frame1)
        assert is_change or diff > 0.3


class TestCompositionAnalyzer:
    """Tests for CompositionAnalyzer class."""

    def test_initialization(self):
        """Test CompositionAnalyzer initialization."""
        analyzer = CompositionAnalyzer()
        assert analyzer is not None

    def test_analyze_composition(self):
        """Test composition analysis."""
        analyzer = CompositionAnalyzer()

        # Create test frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        score = analyzer.analyze_composition(frame)

        assert 0 <= score <= 1

    def test_analyze_contrast(self):
        """Test contrast analysis."""
        analyzer = CompositionAnalyzer()

        # Low contrast frame
        low_contrast = np.ones((480, 640, 3), dtype=np.uint8) * 128
        score_low = analyzer.analyze_contrast(low_contrast)

        # High contrast frame
        high_contrast = np.zeros((480, 640, 3), dtype=np.uint8)
        high_contrast[:240, :, :] = 255
        score_high = analyzer.analyze_contrast(high_contrast)

        assert score_high > score_low

    def test_analyze_lighting(self):
        """Test lighting analysis."""
        analyzer = CompositionAnalyzer()

        # Well-lit frame
        well_lit = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        score_good = analyzer.analyze_lighting(well_lit)

        # Underexposed frame
        underexposed = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
        score_dark = analyzer.analyze_lighting(underexposed)

        # Overexposed frame
        overexposed = np.random.randint(225, 255, (480, 640, 3), dtype=np.uint8)
        score_bright = analyzer.analyze_lighting(overexposed)

        assert score_good > score_dark
        assert score_good > score_bright


class TestQualityScorer:
    """Tests for QualityScorer class."""

    def test_initialization(self):
        """Test QualityScorer initialization."""
        scorer = QualityScorer()
        assert scorer is not None

        # Check weights are normalized
        total = (
            scorer.motion_weight +
            scorer.face_weight +
            scorer.composition_weight +
            scorer.contrast_weight +
            scorer.lighting_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_score_frame(self):
        """Test single frame scoring."""
        scorer = QualityScorer()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        quality = scorer.score_frame(frame, frame_index=0, timestamp=0.0)

        assert isinstance(quality, FrameQuality)
        assert 0 <= quality.composite_score <= 1
        assert quality.frame_index == 0

        scorer.close()

    def test_score_with_motion(self):
        """Test scoring with motion between frames."""
        scorer = QualityScorer()

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        quality1 = scorer.score_frame(frame1, frame_index=0)
        quality2 = scorer.score_frame(frame2, prev_frame=frame1, frame_index=1)

        # Second frame should have motion score
        assert quality2.motion_score > 0

        scorer.close()

    def test_detect_shots(self):
        """Test shot detection from frame qualities."""
        scorer = QualityScorer()

        # Create mock frame qualities with clear shot boundary
        frame_qualities = []
        for i in range(50):
            # First 25 frames - high quality
            if i < 25:
                quality = 0.8 + np.random.random() * 0.1
            # Next 25 frames - different quality
            else:
                quality = 0.4 + np.random.random() * 0.1

            fq = FrameQuality(
                frame_index=i,
                timestamp=i / 30.0,
                motion_score=0.5,
                face_score=0.5,
                composition_score=0.5,
                contrast_score=0.5,
                lighting_score=0.5,
                composite_score=quality,
            )
            frame_qualities.append(fq)

        shots = scorer.detect_shots(frame_qualities, scene_threshold=0.3)

        assert len(shots) >= 1
        for shot in shots:
            assert isinstance(shot, Shot)
            assert shot.start_frame < shot.end_frame

        scorer.close()

    def test_get_best_shots(self):
        """Test filtering best shots by quality."""
        scorer = QualityScorer()

        shots = [
            Shot(0, 30, 0.0, 1.0, 0.9, 0.95, 15),
            Shot(30, 60, 1.0, 2.0, 0.3, 0.4, 45),
            Shot(60, 90, 2.0, 3.0, 0.7, 0.8, 75),
        ]

        best = scorer.get_best_shots(shots, min_quality=0.5)

        assert len(best) == 2
        assert best[0].avg_quality >= best[1].avg_quality

        scorer.close()

    def test_quality_summary(self):
        """Test quality summary statistics."""
        scorer = QualityScorer()

        frame_qualities = [
            FrameQuality(i, i/30, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 + i*0.01)
            for i in range(10)
        ]

        summary = scorer.get_quality_summary(frame_qualities)

        assert "total_frames" in summary
        assert summary["total_frames"] == 10
        assert "composite" in summary
        assert "mean" in summary["composite"]

        scorer.close()
