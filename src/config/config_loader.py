"""Configuration loader and validator for Cutter."""
import os
import logging
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_CONFIG = {
    "project_name": "untitled",
    "input_video": None,
    "input_video_folder": None,
    "audio_track": None,
    "output_directory": "./output",
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
    "motion_graphics": {
        "enabled": True,
        "intensity": "high",
        "text": {
            "enabled": True,
            "entrance_animation": "slide_in",
            "exit_animation": "fade_out",
            "font_family": "Arial",
            "font_size": 96,
            "font_color": "#FFFFFF",
            "glow_enabled": True,
            "glow_intensity": 0.5,
            "drop_shadow": True,
            "items": [],
        },
        "transitions": {
            "enabled": True,
            "style": "dynamic",
            "duration": 0.3,
            "types": [],
        },
        "particles": {
            "enabled": True,
            "type": "confetti",
            "density": "medium",
            "colors": [],
            "audio_reactive": True,
        },
    },
    "color_grading": {
        "enabled": True,
        "lut_file": None,
        "lut_strength": 75,
        "preset_profile": "cinematic",
        "saturation": 1.0,
        "contrast": 1.0,
        "brightness": 1.0,
        "hue_shift": 0,
        "effects": {
            "blur": 0.0,
            "glow": 0.0,
            "grain": 0.0,
            "chromatic_aberration": 0.0,
            "vignette": 0.0,
        },
    },
    "export": {
        "formats": ["mp4"],
        "capcut_projects": True,
        "fcp_xml": False,
        "video": {
            "codec": "h264",
            "quality": 18,
            "fps": 30,
        },
        "audio": {
            "codec": "aac",
            "bitrate": "192k",
        },
        "platform": "tiktok",
    },
    "variations": {
        "count": 3,
        "sensitivities": ["low", "medium", "high"],
    },
    "logging": {
        "level": "INFO",
        "file": None,
    },
}


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML file. Uses self.config_path if not provided.

        Returns:
            Configuration dictionary with defaults merged.
        """
        path = config_path or self.config_path

        if path is None:
            logger.warning("No config path provided, using defaults")
            self._config = self._deep_copy(DEFAULT_CONFIG)
            return self._config

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

        # Merge with defaults
        self._config = self._merge_configs(
            self._deep_copy(DEFAULT_CONFIG), user_config
        )

        logger.info(f"Loaded configuration from {path}")
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "audio_analysis.beat_detection_threshold")
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "audio_analysis.beat_detection_threshold")
            value: Value to set.
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge override config into base config.

        Args:
            base: Base configuration dictionary.
            override: Override values.

        Returns:
            Merged configuration.
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a configuration object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj


class ConfigValidator:
    """Validate configuration values."""

    VALID_SENSITIVITIES = {"low", "medium", "high"}
    VALID_CUT_PATTERNS = {"low", "balanced", "high"}
    VALID_TRANSITION_STYLES = {"geometric", "organic", "camera", "distortion", "dynamic"}
    VALID_PARTICLE_TYPES = {"confetti", "sparkles", "rain", "snow", "explosion", "bubbles"}
    VALID_COLOR_PRESETS = {
        "cinematic", "cyberpunk", "vintage", "vhs", "neon",
        "sepia", "noir", "cool", "warm", "saturated"
    }
    VALID_PLATFORMS = {"tiktok", "instagram_reels", "youtube_shorts", "youtube", "web"}
    VALID_ENTRANCE_ANIMATIONS = {
        "fade_in", "slide_in", "scale_up", "bounce_in",
        "morph", "typewriter", "blur_in", "glitch_in"
    }
    VALID_EXIT_ANIMATIONS = {
        "fade_out", "slide_out", "scale_down", "bounce_out", "blur_out"
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration.

        Args:
            config: Configuration dictionary to validate.
        """
        self.config = config
        self.errors: list = []
        self.warnings: list = []

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if configuration is valid, False otherwise.
        """
        self.errors = []
        self.warnings = []

        self._validate_required_fields()
        self._validate_audio_analysis()
        self._validate_shot_quality()
        self._validate_cut_timing()
        self._validate_motion_graphics()
        self._validate_color_grading()
        self._validate_export()
        self._validate_variations()

        if self.errors:
            logger.error(f"Configuration validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        return len(self.errors) == 0

    def _validate_required_fields(self) -> None:
        """Validate required fields exist."""
        if not self.config.get("input_video") and not self.config.get("input_video_folder"):
            self.errors.append("input_video or input_video_folder is required")

        if not self.config.get("audio_track"):
            self.errors.append("audio_track is required")

    def _validate_audio_analysis(self) -> None:
        """Validate audio analysis settings."""
        aa = self.config.get("audio_analysis", {})

        sensitivity = aa.get("audio_sensitivity", "medium")
        if sensitivity not in self.VALID_SENSITIVITIES:
            self.errors.append(
                f"Invalid audio_sensitivity '{sensitivity}'. "
                f"Must be one of: {self.VALID_SENSITIVITIES}"
            )

        threshold = aa.get("beat_detection_threshold", 0.5)
        if not (0.0 <= threshold <= 1.0):
            self.errors.append(
                f"beat_detection_threshold must be between 0.0 and 1.0, got {threshold}"
            )

    def _validate_shot_quality(self) -> None:
        """Validate shot quality settings."""
        sq = self.config.get("shot_quality", {})

        # Check weights sum to approximately 1.0
        weights = [
            sq.get("motion_weight", 0.25),
            sq.get("face_detection_weight", 0.25),
            sq.get("composition_weight", 0.15),
            sq.get("contrast_weight", 0.25),
            sq.get("lighting_weight", 0.10),
        ]

        weight_sum = sum(weights)
        if not (0.99 <= weight_sum <= 1.01):
            self.warnings.append(
                f"Shot quality weights sum to {weight_sum:.2f}, expected ~1.0"
            )

        # Check individual weights
        for weight in weights:
            if not (0.0 <= weight <= 1.0):
                self.errors.append(
                    f"Shot quality weights must be between 0.0 and 1.0"
                )
                break

        min_score = sq.get("min_quality_score", 0.4)
        if not (0.0 <= min_score <= 1.0):
            self.errors.append(
                f"min_quality_score must be between 0.0 and 1.0, got {min_score}"
            )

    def _validate_cut_timing(self) -> None:
        """Validate cut timing settings."""
        ct = self.config.get("cut_timing", {})

        target_length = ct.get("target_video_length", 60)
        if target_length <= 0:
            self.errors.append(
                f"target_video_length must be positive, got {target_length}"
            )

        min_dur = ct.get("min_cut_duration", 0.5)
        max_dur = ct.get("max_cut_duration", 5.0)
        if min_dur <= 0:
            self.errors.append(
                f"min_cut_duration must be positive, got {min_dur}"
            )
        if max_dur <= 0:
            self.errors.append(
                f"max_cut_duration must be positive, got {max_dur}"
            )
        if min_dur > max_dur:
            self.errors.append(
                f"min_cut_duration ({min_dur}) cannot be greater than "
                f"max_cut_duration ({max_dur})"
            )

        pattern = ct.get("cut_pattern", "balanced")
        if pattern not in self.VALID_CUT_PATTERNS:
            self.errors.append(
                f"Invalid cut_pattern '{pattern}'. "
                f"Must be one of: {self.VALID_CUT_PATTERNS}"
            )

    def _validate_motion_graphics(self) -> None:
        """Validate motion graphics settings."""
        mg = self.config.get("motion_graphics", {})

        if not mg.get("enabled", True):
            return  # Skip validation if disabled

        intensity = mg.get("intensity", "high")
        if intensity not in self.VALID_SENSITIVITIES:
            self.errors.append(
                f"Invalid motion_graphics intensity '{intensity}'. "
                f"Must be one of: {self.VALID_SENSITIVITIES}"
            )

        # Validate text settings
        text = mg.get("text", {})
        if text.get("enabled", True):
            entrance = text.get("entrance_animation", "slide_in")
            if entrance not in self.VALID_ENTRANCE_ANIMATIONS:
                self.errors.append(
                    f"Invalid entrance_animation '{entrance}'. "
                    f"Must be one of: {self.VALID_ENTRANCE_ANIMATIONS}"
                )

            exit_anim = text.get("exit_animation", "fade_out")
            if exit_anim not in self.VALID_EXIT_ANIMATIONS:
                self.errors.append(
                    f"Invalid exit_animation '{exit_anim}'. "
                    f"Must be one of: {self.VALID_EXIT_ANIMATIONS}"
                )

        # Validate transitions
        trans = mg.get("transitions", {})
        if trans.get("enabled", True):
            style = trans.get("style", "dynamic")
            if style not in self.VALID_TRANSITION_STYLES:
                self.errors.append(
                    f"Invalid transition style '{style}'. "
                    f"Must be one of: {self.VALID_TRANSITION_STYLES}"
                )

        # Validate particles
        particles = mg.get("particles", {})
        if particles.get("enabled", True):
            ptype = particles.get("type", "confetti")
            if ptype not in self.VALID_PARTICLE_TYPES:
                self.errors.append(
                    f"Invalid particle type '{ptype}'. "
                    f"Must be one of: {self.VALID_PARTICLE_TYPES}"
                )

            density = particles.get("density", "medium")
            if density not in self.VALID_SENSITIVITIES:
                self.errors.append(
                    f"Invalid particle density '{density}'. "
                    f"Must be one of: {self.VALID_SENSITIVITIES}"
                )

    def _validate_color_grading(self) -> None:
        """Validate color grading settings."""
        cg = self.config.get("color_grading", {})

        if not cg.get("enabled", True):
            return

        preset = cg.get("preset_profile")
        if preset and preset not in self.VALID_COLOR_PRESETS:
            self.errors.append(
                f"Invalid color preset '{preset}'. "
                f"Must be one of: {self.VALID_COLOR_PRESETS}"
            )

        lut_strength = cg.get("lut_strength", 75)
        if not (0 <= lut_strength <= 100):
            self.errors.append(
                f"lut_strength must be between 0 and 100, got {lut_strength}"
            )

        # Check adjustment ranges
        for adj in ["saturation", "contrast", "brightness"]:
            value = cg.get(adj, 1.0)
            if not (0.0 <= value <= 2.0):
                self.errors.append(
                    f"{adj} must be between 0.0 and 2.0, got {value}"
                )

        hue_shift = cg.get("hue_shift", 0)
        if not (-180 <= hue_shift <= 180):
            self.errors.append(
                f"hue_shift must be between -180 and 180, got {hue_shift}"
            )

    def _validate_export(self) -> None:
        """Validate export settings."""
        exp = self.config.get("export", {})

        platform = exp.get("platform", "tiktok")
        if platform not in self.VALID_PLATFORMS:
            self.errors.append(
                f"Invalid platform '{platform}'. "
                f"Must be one of: {self.VALID_PLATFORMS}"
            )

        video = exp.get("video", {})
        quality = video.get("quality", 18)
        if not (0 <= quality <= 51):
            self.errors.append(
                f"video quality (CRF) must be between 0 and 51, got {quality}"
            )

        fps = video.get("fps", 30)
        if fps <= 0:
            self.errors.append(f"fps must be positive, got {fps}")

    def _validate_variations(self) -> None:
        """Validate variation settings."""
        var = self.config.get("variations", {})

        count = var.get("count", 3)
        if not (1 <= count <= 5):
            self.errors.append(
                f"variations count must be between 1 and 5, got {count}"
            )

        sensitivities = var.get("sensitivities", ["low", "medium", "high"])
        for sens in sensitivities:
            if sens not in self.VALID_SENSITIVITIES:
                self.errors.append(
                    f"Invalid sensitivity '{sens}' in variations. "
                    f"Must be one of: {self.VALID_SENSITIVITIES}"
                )


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging based on configuration.

    Args:
        config: Configuration dictionary with logging settings.
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    log_file = log_config.get("file")
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
