"""Tests for data structures."""
import pytest
from src.data_structures import (
    BeatInfo, FrameQuality, Shot, Cut, Edit, Transition,
    TextAnimation, ParticleEffect, ColorGrade, Timeline,
    VideoMetadata, AudioMetadata
)


class TestBeatInfo:
    """Tests for BeatInfo class."""

    def test_basic_creation(self):
        """Test basic BeatInfo creation."""
        beat_info = BeatInfo(
            beat_times=[0.0, 0.5, 1.0, 1.5],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.9, 0.7],
            energy_peaks=[0.5, 1.5],
        )

        assert beat_info.num_beats == 4
        assert beat_info.bpm == 120.0
        assert len(beat_info.energy_peaks) == 2

    def test_get_nearest_beat(self):
        """Test finding nearest beat."""
        beat_info = BeatInfo(
            beat_times=[0.0, 1.0, 2.0, 3.0],
            bpm=60.0,
            beat_strength=[0.8, 0.8, 0.8, 0.8],
            energy_peaks=[],
        )

        # Exactly on beat
        idx, time, dist = beat_info.get_nearest_beat(1.0)
        assert idx == 1
        assert time == 1.0
        assert dist == 0.0

        # Between beats
        idx, time, dist = beat_info.get_nearest_beat(1.3)
        assert idx == 1
        assert time == 1.0
        assert abs(dist - 0.3) < 0.001

        idx, time, dist = beat_info.get_nearest_beat(1.7)
        assert idx == 2
        assert time == 2.0
        assert abs(dist - 0.3) < 0.001

    def test_empty_beats(self):
        """Test with no beats."""
        beat_info = BeatInfo(
            beat_times=[],
            bpm=0.0,
            beat_strength=[],
            energy_peaks=[],
        )

        assert beat_info.num_beats == 0
        idx, time, dist = beat_info.get_nearest_beat(1.0)
        assert idx == -1
        assert dist == float('inf')


class TestCut:
    """Tests for Cut class."""

    def test_cut_creation(self):
        """Test Cut creation and properties."""
        cut = Cut(
            source_start=5.0,
            source_end=8.0,
            timeline_start=0.0,
            timeline_end=3.0,
            quality_score=0.85,
            beat_alignment="strong_beat",
            beat_index=2,
            shot_index=0,
        )

        assert cut.duration == 3.0
        assert cut.source_duration == 3.0
        assert cut.quality_score == 0.85

    def test_cut_to_dict(self):
        """Test Cut serialization."""
        cut = Cut(
            source_start=0.0,
            source_end=2.0,
            timeline_start=0.0,
            timeline_end=2.0,
            quality_score=0.7,
            beat_alignment="beat",
            beat_index=0,
            shot_index=0,
        )

        data = cut.to_dict()
        assert data["type"] == "video_clip"
        assert data["source_start"] == 0.0
        assert data["timeline_end"] == 2.0


class TestShot:
    """Tests for Shot class."""

    def test_shot_creation(self):
        """Test Shot creation and properties."""
        shot = Shot(
            start_frame=0,
            end_frame=90,
            start_time=0.0,
            end_time=3.0,
            avg_quality=0.75,
            peak_quality=0.9,
            peak_frame=45,
        )

        assert shot.duration == 3.0
        assert shot.frame_count == 91
        assert shot.avg_quality == 0.75


class TestEdit:
    """Tests for Edit class."""

    def test_edit_creation(self):
        """Test Edit creation."""
        cuts = [
            Cut(0, 2, 0, 2, 0.8, "beat", 0, 0),
            Cut(5, 7, 2, 4, 0.7, "beat", 1, 1),
        ]

        edit = Edit(
            timeline=cuts,
            transitions=[],
            text_animations=[],
            particle_effects=[],
            color_grade=ColorGrade(),
            total_duration=4.0,
            generation_params={"sensitivity": "medium"},
            quality_score=0.75,
        )

        assert edit.num_cuts == 2
        assert edit.total_duration == 4.0

    def test_edit_to_dict(self):
        """Test Edit serialization."""
        edit = Edit(
            timeline=[],
            transitions=[],
            text_animations=[],
            particle_effects=[],
            color_grade=ColorGrade(preset="cinematic"),
            total_duration=60.0,
            generation_params={},
            quality_score=0.8,
        )

        data = edit.to_dict()
        assert "timeline" in data
        assert "color_grading" in data
        assert data["total_duration"] == 60.0


class TestVideoMetadata:
    """Tests for VideoMetadata class."""

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        metadata = VideoMetadata(
            path="/test.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=1000000,
        )

        assert abs(metadata.aspect_ratio - 1.778) < 0.01
        assert not metadata.is_portrait

    def test_portrait_video(self):
        """Test portrait video detection."""
        metadata = VideoMetadata(
            path="/test.mp4",
            width=1080,
            height=1920,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=1000000,
        )

        assert metadata.is_portrait
        assert abs(metadata.aspect_ratio - 0.5625) < 0.01
