"""Tests for configuration management."""
import pytest
import tempfile
import os
import yaml

from src.config import ConfigLoader, ConfigValidator


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_defaults(self):
        """Test loading default configuration."""
        loader = ConfigLoader()
        config = loader.load_config()

        assert "audio_analysis" in config
        assert "shot_quality" in config
        assert "cut_timing" in config
        assert "motion_graphics" in config

    def test_load_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "project_name": "test_project",
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "audio_analysis": {
                "audio_sensitivity": "high",
            },
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()

            assert config["project_name"] == "test_project"
            assert config["audio_analysis"]["audio_sensitivity"] == "high"
            # Check defaults are merged
            assert "beat_detection_threshold" in config["audio_analysis"]
        finally:
            os.unlink(config_path)

    def test_get_nested_key(self):
        """Test getting nested configuration values."""
        loader = ConfigLoader()
        config = loader.load_config()

        threshold = loader.get("audio_analysis.beat_detection_threshold")
        assert threshold == 0.5

        missing = loader.get("nonexistent.key", default="default")
        assert missing == "default"

    def test_set_nested_key(self):
        """Test setting nested configuration values."""
        loader = ConfigLoader()
        loader.load_config()

        loader.set("cut_timing.target_video_length", 90)
        assert loader.get("cut_timing.target_video_length") == 90

    def test_file_not_found(self):
        """Test error when config file not found."""
        loader = ConfigLoader("/nonexistent/path/config.yaml")

        with pytest.raises(FileNotFoundError):
            loader.load_config()


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
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
            "export": {
                "platform": "tiktok",
                "video": {"quality": 18, "fps": 30},
            },
            "variations": {
                "count": 3,
                "sensitivities": ["low", "medium", "high"],
            },
        }

        validator = ConfigValidator(config)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        config = {}

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert "input_video is required" in validator.errors
        assert "audio_track is required" in validator.errors

    def test_invalid_sensitivity(self):
        """Test validation fails for invalid sensitivity."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "audio_analysis": {
                "audio_sensitivity": "invalid",
            },
        }

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("audio_sensitivity" in e for e in validator.errors)

    def test_invalid_threshold_range(self):
        """Test validation fails for out-of-range threshold."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "audio_analysis": {
                "beat_detection_threshold": 1.5,  # > 1.0
            },
        }

        validator = ConfigValidator(config)
        assert validator.validate() is False

    def test_invalid_cut_durations(self):
        """Test validation fails when min > max duration."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "cut_timing": {
                "min_cut_duration": 5.0,
                "max_cut_duration": 2.0,  # < min
            },
        }

        validator = ConfigValidator(config)
        assert validator.validate() is False

    def test_weight_sum_warning(self):
        """Test warning when quality weights don't sum to 1."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "shot_quality": {
                "motion_weight": 0.5,
                "face_detection_weight": 0.5,
                "composition_weight": 0.5,  # Sum > 1
                "contrast_weight": 0.5,
                "lighting_weight": 0.5,
            },
        }

        validator = ConfigValidator(config)
        validator.validate()
        assert len(validator.warnings) > 0

    def test_invalid_variations_count(self):
        """Test validation fails for invalid variations count."""
        config = {
            "input_video": "test.mp4",
            "audio_track": "test.mp3",
            "variations": {
                "count": 10,  # > 5
            },
        }

        validator = ConfigValidator(config)
        assert validator.validate() is False
