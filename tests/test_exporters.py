"""Tests for project exporters."""
import pytest
import json
import os
import tempfile

from src.project_exporter import TimelineSerializer, CapCutExporter, FCPExporter
from src.data_structures import (
    Edit, Cut, Transition, TextAnimation, ParticleEffect,
    ColorGrade, BeatInfo, VideoMetadata, Timeline
)


class TestTimelineSerializer:
    """Tests for TimelineSerializer class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "project_name": "test_project",
            "export": {
                "video": {"fps": 30},
            },
        }

    @pytest.fixture
    def sample_edit(self):
        """Create sample edit."""
        cuts = [
            Cut(0.0, 2.0, 0.0, 2.0, 0.8, "beat", 0, 0),
            Cut(5.0, 7.0, 2.0, 4.0, 0.7, "beat", 4, 1),
        ]
        return Edit(
            timeline=cuts,
            transitions=[],
            text_animations=[],
            particle_effects=[],
            color_grade=ColorGrade(preset="cinematic"),
            total_duration=4.0,
            generation_params={"sensitivity": "medium"},
            quality_score=0.75,
        )

    @pytest.fixture
    def sample_beat_info(self):
        """Create sample beat info."""
        return BeatInfo(
            beat_times=[0.0, 0.5, 1.0, 1.5, 2.0],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.8, 0.6, 0.8],
            energy_peaks=[1.0],
        )

    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            path="/test/video.mp4",
            width=1080,
            height=1920,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=100000000,
        )

    def test_edit_to_timeline(
        self, sample_config, sample_edit, sample_beat_info, sample_video_metadata
    ):
        """Test converting edit to timeline."""
        serializer = TimelineSerializer(sample_config)

        timeline = serializer.edit_to_timeline(
            sample_edit,
            sample_beat_info,
            sample_video_metadata,
            "/test/audio.mp3",
        )

        assert isinstance(timeline, Timeline)
        assert timeline.metadata["project_name"] == "test_project"
        assert timeline.edit == sample_edit

    def test_timeline_to_json(
        self, sample_config, sample_edit, sample_beat_info, sample_video_metadata
    ):
        """Test exporting timeline to JSON."""
        serializer = TimelineSerializer(sample_config)

        timeline = serializer.edit_to_timeline(
            sample_edit,
            sample_beat_info,
            sample_video_metadata,
            "/test/audio.mp3",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timeline.json")
            result = serializer.timeline_to_json(timeline, output_path)

            assert os.path.exists(result)

            with open(result, 'r') as f:
                data = json.load(f)

            assert "metadata" in data
            assert "timeline" in data
            assert "beat_info" in data

    def test_load_timeline(self, sample_config):
        """Test loading timeline from JSON."""
        serializer = TimelineSerializer(sample_config)

        # Create a test JSON file
        test_data = {
            "metadata": {"project_name": "test"},
            "timeline": [],
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(test_data, f)
            json_path = f.name

        try:
            loaded = serializer.load_timeline(json_path)
            assert loaded["metadata"]["project_name"] == "test"
        finally:
            os.unlink(json_path)


class TestCapCutExporter:
    """Tests for CapCutExporter class."""

    @pytest.fixture
    def sample_edit(self):
        """Create sample edit."""
        cuts = [
            Cut(0.0, 2.0, 0.0, 2.0, 0.8, "beat", 0, 0),
        ]
        text_anims = [
            TextAnimation(
                text="Test",
                start_time=0.0,
                duration=2.0,
                entrance_animation="fade_in",
                exit_animation="fade_out",
            ),
        ]
        return Edit(
            timeline=cuts,
            transitions=[],
            text_animations=text_anims,
            particle_effects=[],
            color_grade=ColorGrade(preset="cinematic"),
            total_duration=2.0,
            generation_params={},
            quality_score=0.8,
        )

    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            path="/test/video.mp4",
            width=1080,
            height=1920,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=100000000,
        )

    @pytest.fixture
    def sample_beat_info(self):
        """Create sample beat info."""
        return BeatInfo(
            beat_times=[0.0, 0.5, 1.0],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.8],
            energy_peaks=[0.5],
        )

    def test_export_to_capcut(
        self, sample_edit, sample_video_metadata, sample_beat_info
    ):
        """Test exporting to CapCut format."""
        exporter = CapCutExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "project.capcut")

            result = exporter.export_to_capcut(
                sample_edit,
                sample_video_metadata,
                "/test/audio.mp3",
                output_path,
                sample_beat_info,
            )

            assert os.path.exists(result)

            with open(result, 'r') as f:
                data = json.load(f)

            assert "tracks" in data
            assert "materials" in data
            assert "beat_markers" in data


class TestFCPExporter:
    """Tests for FCPExporter class."""

    @pytest.fixture
    def sample_edit(self):
        """Create sample edit."""
        cuts = [
            Cut(0.0, 2.0, 0.0, 2.0, 0.8, "beat", 0, 0),
        ]
        return Edit(
            timeline=cuts,
            transitions=[],
            text_animations=[],
            particle_effects=[],
            color_grade=ColorGrade(),
            total_duration=2.0,
            generation_params={},
            quality_score=0.8,
        )

    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            path="/test/video.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=100000000,
        )

    def test_export_to_fcp_xml(self, sample_edit, sample_video_metadata):
        """Test exporting to FCP XML format."""
        exporter = FCPExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "project.fcpxml")

            result = exporter.export_to_fcp_xml(
                sample_edit,
                sample_video_metadata,
                "/test/audio.mp3",
                output_path,
            )

            assert os.path.exists(result)

            with open(result, 'r') as f:
                content = f.read()

            assert "fcpxml" in content
            assert "resources" in content
            assert "spine" in content
