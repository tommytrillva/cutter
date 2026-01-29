"""Integration tests for the full pipeline."""
import pytest
import os
import tempfile
import numpy as np

from src.config import ConfigLoader
from src.data_structures import BeatInfo, Shot, Edit


class TestPipelineIntegration:
    """Integration tests for the pipeline components."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "project_name": "test_project",
            "input_video": "/test/video.mp4",
            "audio_track": "/test/audio.mp3",
            "output_directory": "./test_output",
            "audio_analysis": {
                "audio_sensitivity": "medium",
                "beat_detection_threshold": 0.5,
            },
            "shot_quality": {
                "motion_weight": 0.25,
                "face_detection_weight": 0.25,
                "composition_weight": 0.15,
                "contrast_weight": 0.25,
                "lighting_weight": 0.10,
                "min_quality_score": 0.4,
            },
            "cut_timing": {
                "target_video_length": 60,
                "min_cut_duration": 0.5,
                "max_cut_duration": 5.0,
                "cut_pattern": "balanced",
            },
            "variations": {
                "count": 3,
                "sensitivities": ["low", "medium", "high"],
            },
            "export": {
                "formats": ["mp4"],
                "capcut_projects": True,
                "video": {"quality": 18, "fps": 30},
                "platform": "tiktok",
            },
            "motion_graphics": {
                "enabled": True,
                "text": {"enabled": False},
                "transitions": {"enabled": True, "style": "dynamic"},
                "particles": {"enabled": True, "type": "confetti"},
            },
            "color_grading": {
                "enabled": True,
                "preset_profile": "cinematic",
            },
        }

    def test_config_loading(self, sample_config):
        """Test configuration loading and validation."""
        from src.config import ConfigValidator

        validator = ConfigValidator(sample_config)
        assert validator.validate() is True

    def test_shot_selection_integration(self, sample_config):
        """Test shot selection with beat info."""
        from src.edit_generator import ShotSelector

        # Create sample shots
        shots = [
            Shot(0, 90, 0.0, 3.0, 0.9, 0.95, 45),
            Shot(100, 190, 3.33, 6.33, 0.7, 0.8, 145),
            Shot(200, 290, 6.67, 9.67, 0.5, 0.6, 245),
            Shot(300, 390, 10.0, 13.0, 0.85, 0.9, 345),
        ]

        # Create sample beat info
        beat_info = BeatInfo(
            beat_times=list(np.arange(0, 20, 0.5)),
            bpm=120.0,
            beat_strength=[0.7] * 40,
            energy_peaks=[2.0, 5.0, 8.0, 12.0],
        )

        selector = ShotSelector(
            min_quality=sample_config["shot_quality"]["min_quality_score"]
        )

        selected = selector.select_shots(
            shots,
            sample_config["cut_timing"]["target_video_length"],
            beat_info,
        )

        assert len(selected) > 0

    def test_variation_generation_integration(self, sample_config):
        """Test generating multiple edit variations."""
        from src.edit_generator import EditVariationGenerator

        # Create sample data
        shots = [
            Shot(i * 100, i * 100 + 90, i * 3.33, (i + 1) * 3.33, 0.7, 0.8, i * 100 + 45)
            for i in range(10)
        ]

        beat_info = BeatInfo(
            beat_times=list(np.arange(0, 60, 0.5)),
            bpm=120.0,
            beat_strength=[0.7] * 120,
            energy_peaks=[5.0, 15.0, 30.0, 45.0],
        )

        generator = EditVariationGenerator(
            sample_config,
            num_variations=sample_config["variations"]["count"],
            sensitivities=sample_config["variations"]["sensitivities"],
        )

        variations = generator.generate_variations(
            shots,
            beat_info,
            target_duration=30.0,
        )

        assert len(variations) == 3
        assert all(isinstance(v, Edit) for v in variations)

        # Check each variation has different parameters
        sensitivities = [v.generation_params.get("sensitivity") for v in variations]
        assert len(set(sensitivities)) == 3  # All unique

    def test_exporter_integration(self, sample_config):
        """Test exporting variations to multiple formats."""
        from src.edit_generator import EditVariationGenerator
        from src.project_exporter import TimelineSerializer
        from src.data_structures import VideoMetadata

        # Create sample data
        shots = [
            Shot(i * 100, i * 100 + 90, i * 3.33, (i + 1) * 3.33, 0.7, 0.8, i * 100 + 45)
            for i in range(5)
        ]

        beat_info = BeatInfo(
            beat_times=list(np.arange(0, 30, 0.5)),
            bpm=120.0,
            beat_strength=[0.7] * 60,
            energy_peaks=[5.0, 15.0],
        )

        video_metadata = VideoMetadata(
            path="/test/video.mp4",
            width=1080,
            height=1920,
            fps=30.0,
            duration=60.0,
            codec="h264",
            total_frames=1800,
            file_size=100000000,
        )

        # Generate variations
        generator = EditVariationGenerator(sample_config, num_variations=2)
        variations = generator.generate_variations(shots, beat_info, 15.0)

        # Export to JSON
        serializer = TimelineSerializer(sample_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = serializer.export_all_variations(
                variations,
                beat_info,
                video_metadata,
                "/test/audio.mp3",
                tmpdir,
            )

            assert len(output_paths) == 2
            for path in output_paths:
                assert os.path.exists(path)
                assert path.endswith(".json")


class TestDataFlowIntegration:
    """Test data flow between components."""

    def test_beat_info_to_cuts(self):
        """Test that beat info properly influences cut placement."""
        from src.edit_generator.timeline_builder import TimelineBuilder

        # Create beat info with specific beat times
        beat_info = BeatInfo(
            beat_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bpm=60.0,
            beat_strength=[0.9, 0.5, 0.9, 0.5, 0.9, 0.5],
            energy_peaks=[2.0, 4.0],
        )

        # Create shots
        shots = [
            Shot(0, 30, 0.0, 1.0, 0.8, 0.9, 15),
            Shot(60, 90, 2.0, 3.0, 0.7, 0.8, 75),
        ]

        selected = [
            (shots[0], 0.0, 1.0),
            (shots[1], 1.0, 2.0),
        ]

        builder = TimelineBuilder()
        cuts = builder.build_timeline(selected, beat_info, 2.0)

        # Check beat alignment info is set
        assert all(cut.beat_index >= 0 or cut.beat_alignment == "none" for cut in cuts)

    def test_quality_scores_preserved(self):
        """Test that quality scores are preserved through pipeline."""
        from src.edit_generator import ShotSelector
        from src.edit_generator.timeline_builder import EditBuilder

        config = {
            "cut_timing": {"min_cut_duration": 0.5, "max_cut_duration": 5.0},
            "motion_graphics": {
                "transitions": {"enabled": True, "style": "dynamic", "duration": 0.3},
                "particles": {"enabled": False},
                "text": {"enabled": False},
            },
            "color_grading": {"enabled": True, "preset_profile": "cinematic"},
        }

        # Create shots with known quality scores
        shots = [
            Shot(0, 90, 0.0, 3.0, 0.9, 0.95, 45),
            Shot(100, 190, 3.33, 6.33, 0.5, 0.6, 145),
        ]

        beat_info = BeatInfo(
            beat_times=[0.0, 0.5, 1.0, 1.5, 2.0],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.8, 0.6, 0.8],
            energy_peaks=[1.0],
        )

        selector = ShotSelector(min_quality=0.4)
        selected = selector.select_shots(shots, 4.0, beat_info)

        builder = EditBuilder(config)
        edit = builder.build_edit(selected, beat_info, 4.0)

        # Check quality scores are preserved in cuts
        for cut in edit.timeline:
            assert 0 <= cut.quality_score <= 1
