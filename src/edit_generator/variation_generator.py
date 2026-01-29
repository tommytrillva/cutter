"""Edit variation generation for Cutter."""
import logging
from typing import List, Dict, Any, Optional
import random

from .shot_selector import ShotSelector
from .timeline_builder import EditBuilder
from ..data_structures import Shot, BeatInfo, Edit, Sensitivity

logger = logging.getLogger(__name__)


# Variation presets
VARIATION_PRESETS = {
    Sensitivity.LOW: {
        "name": "Smooth",
        "description": "Longer cuts, smoother transitions",
        "min_cut_duration": 1.5,
        "max_cut_duration": 6.0,
        "min_quality": 0.5,
        "transition_style": "organic",
        "particle_density": "low",
    },
    Sensitivity.MEDIUM: {
        "name": "Balanced",
        "description": "Balanced cuts and transitions",
        "min_cut_duration": 0.8,
        "max_cut_duration": 4.0,
        "min_quality": 0.4,
        "transition_style": "dynamic",
        "particle_density": "medium",
    },
    Sensitivity.HIGH: {
        "name": "Aggressive",
        "description": "Fast cuts, dynamic transitions",
        "min_cut_duration": 0.3,
        "max_cut_duration": 2.0,
        "min_quality": 0.35,
        "transition_style": "distortion",
        "particle_density": "high",
    },
}


class EditVariationGenerator:
    """Generate multiple edit variations with different styles."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        num_variations: int = 3,
        sensitivities: Optional[List[str]] = None,
    ):
        """Initialize variation generator.

        Args:
            base_config: Base configuration dictionary.
            num_variations: Number of variations to generate.
            sensitivities: List of sensitivity levels for variations.
        """
        self.base_config = base_config
        self.num_variations = num_variations

        if sensitivities:
            self.sensitivities = [Sensitivity(s.lower()) for s in sensitivities]
        else:
            # Default: low, medium, high
            self.sensitivities = [
                Sensitivity.LOW,
                Sensitivity.MEDIUM,
                Sensitivity.HIGH,
            ]

        # Ensure we have enough sensitivities
        while len(self.sensitivities) < num_variations:
            self.sensitivities.append(random.choice(list(Sensitivity)))

    def generate_variations(
        self,
        shots: List[Shot],
        beat_info: BeatInfo,
        target_duration: float,
    ) -> List[Edit]:
        """Generate multiple edit variations.

        Args:
            shots: Available shots to use.
            beat_info: Beat information.
            target_duration: Target video duration.

        Returns:
            List of Edit objects (one per variation).
        """
        variations = []

        for i in range(self.num_variations):
            sensitivity = self.sensitivities[i % len(self.sensitivities)]

            logger.info(
                f"Generating variation {i + 1}/{self.num_variations} "
                f"({sensitivity.value} sensitivity)"
            )

            edit = self._generate_variation(
                shots, beat_info, target_duration, sensitivity, i
            )
            variations.append(edit)

        logger.info(f"Generated {len(variations)} edit variations")
        return variations

    def _generate_variation(
        self,
        shots: List[Shot],
        beat_info: BeatInfo,
        target_duration: float,
        sensitivity: Sensitivity,
        variation_id: int,
    ) -> Edit:
        """Generate a single variation.

        Args:
            shots: Available shots.
            beat_info: Beat information.
            target_duration: Target duration.
            sensitivity: Sensitivity level.
            variation_id: Variation identifier.

        Returns:
            Edit object.
        """
        preset = VARIATION_PRESETS[sensitivity]

        # Create modified config for this variation
        variation_config = self._create_variation_config(preset)

        # Initialize components with variation config
        shot_selector = ShotSelector(
            min_quality=preset["min_quality"],
            prefer_variety=True,
            beat_alignment_weight=0.3 if sensitivity == Sensitivity.HIGH else 0.2,
        )

        edit_builder = EditBuilder(variation_config)

        # Select shots for this variation
        selected_shots = shot_selector.select_shots(
            shots, target_duration, beat_info
        )

        # Build the edit
        generation_params = {
            "sensitivity": sensitivity.value,
            "preset_name": preset["name"],
            "min_cut_duration": preset["min_cut_duration"],
            "max_cut_duration": preset["max_cut_duration"],
            "transition_style": preset["transition_style"],
        }

        edit = edit_builder.build_edit(
            selected_shots,
            beat_info,
            target_duration,
            variation_id=variation_id,
            generation_params=generation_params,
        )

        return edit

    def _create_variation_config(
        self, preset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create configuration for a variation.

        Args:
            preset: Variation preset parameters.

        Returns:
            Modified configuration dictionary.
        """
        config = self._deep_copy(self.base_config)

        # Update cut timing
        config["cut_timing"] = config.get("cut_timing", {})
        config["cut_timing"]["min_cut_duration"] = preset["min_cut_duration"]
        config["cut_timing"]["max_cut_duration"] = preset["max_cut_duration"]

        # Update shot quality
        config["shot_quality"] = config.get("shot_quality", {})
        config["shot_quality"]["min_quality_score"] = preset["min_quality"]

        # Update motion graphics
        config["motion_graphics"] = config.get("motion_graphics", {})
        config["motion_graphics"]["transitions"] = config["motion_graphics"].get("transitions", {})
        config["motion_graphics"]["transitions"]["style"] = preset["transition_style"]

        config["motion_graphics"]["particles"] = config["motion_graphics"].get("particles", {})
        config["motion_graphics"]["particles"]["density"] = preset["particle_density"]

        return config

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a configuration object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def generate_custom_variation(
        self,
        shots: List[Shot],
        beat_info: BeatInfo,
        target_duration: float,
        custom_params: Dict[str, Any],
        variation_id: int = 0,
    ) -> Edit:
        """Generate a variation with custom parameters.

        Args:
            shots: Available shots.
            beat_info: Beat information.
            target_duration: Target duration.
            custom_params: Custom generation parameters.
            variation_id: Variation identifier.

        Returns:
            Edit object.
        """
        # Merge custom params with base config
        variation_config = self._deep_copy(self.base_config)

        if "cut_timing" in custom_params:
            variation_config["cut_timing"].update(custom_params["cut_timing"])

        if "shot_quality" in custom_params:
            variation_config["shot_quality"].update(custom_params["shot_quality"])

        if "motion_graphics" in custom_params:
            for key, value in custom_params["motion_graphics"].items():
                if key in variation_config["motion_graphics"]:
                    if isinstance(value, dict):
                        variation_config["motion_graphics"][key].update(value)
                    else:
                        variation_config["motion_graphics"][key] = value

        # Generate with custom config
        min_quality = custom_params.get("min_quality", 0.4)
        shot_selector = ShotSelector(min_quality=min_quality)
        edit_builder = EditBuilder(variation_config)

        selected_shots = shot_selector.select_shots(
            shots, target_duration, beat_info
        )

        return edit_builder.build_edit(
            selected_shots,
            beat_info,
            target_duration,
            variation_id=variation_id,
            generation_params=custom_params,
        )


# Type alias for backwards compatibility
Any = object
