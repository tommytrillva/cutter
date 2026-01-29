"""Timeline building for edit generation."""
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..data_structures import (
    Shot, BeatInfo, Cut, Edit, Transition, TextAnimation,
    ParticleEffect, ColorGrade, Sensitivity
)

logger = logging.getLogger(__name__)


# Transition styles by category
TRANSITION_STYLES = {
    "geometric": ["wipe_left", "wipe_right", "wipe_up", "wipe_down", "zoom_in",
                  "zoom_out", "rotate_cw", "rotate_ccw", "iris_in", "iris_out",
                  "kaleidoscope", "diagonal_tl", "diagonal_br"],
    "organic": ["dissolve", "ink_splash", "liquid", "smoke", "water", "fire",
                "cloud", "paper_tear", "paint_drip"],
    "camera": ["pan_left", "pan_right", "push_in", "pull_out", "parallax",
               "reveal_left", "reveal_right", "blinds_h", "blinds_v", "shutter"],
    "distortion": ["whip", "flurry", "prism", "glitch", "mosaic", "pixelate",
                   "wave", "ripple", "lens_blur"],
}

ALL_TRANSITIONS = [t for styles in TRANSITION_STYLES.values() for t in styles]


class TimelineBuilder:
    """Build edit timelines from shots and beats."""

    def __init__(
        self,
        min_cut_duration: float = 0.5,
        max_cut_duration: float = 5.0,
        transition_duration: float = 0.3,
        transition_style: str = "dynamic",
    ):
        """Initialize timeline builder.

        Args:
            min_cut_duration: Minimum cut duration in seconds.
            max_cut_duration: Maximum cut duration in seconds.
            transition_duration: Default transition duration.
            transition_style: Transition style category or 'dynamic'.
        """
        self.min_cut_duration = min_cut_duration
        self.max_cut_duration = max_cut_duration
        self.transition_duration = transition_duration
        self.transition_style = transition_style

    def build_timeline(
        self,
        selected_shots: List[Tuple[Shot, float, float]],
        beat_info: BeatInfo,
        target_duration: float,
        add_transitions: bool = True,
    ) -> List[Cut]:
        """Build timeline from selected shots.

        Args:
            selected_shots: List of (shot, start, end) tuples.
            beat_info: Beat information for alignment.
            target_duration: Target duration.
            add_transitions: Whether to calculate transition points.

        Returns:
            List of Cut objects forming the timeline.
        """
        cuts = []

        for i, (shot, timeline_start, timeline_end) in enumerate(selected_shots):
            # Calculate source segment (use from middle of shot for quality)
            shot_duration = shot.end_time - shot.start_time
            cut_duration = timeline_end - timeline_start

            # Prefer using segment around peak quality frame
            peak_time = shot.start_time + (shot.peak_frame - shot.start_frame) / 30.0  # Assume 30fps

            # Center cut around peak if possible
            if cut_duration < shot_duration:
                source_start = max(
                    shot.start_time,
                    min(
                        peak_time - cut_duration / 2,
                        shot.end_time - cut_duration
                    )
                )
            else:
                source_start = shot.start_time

            source_end = source_start + cut_duration

            # Determine beat alignment type
            beat_alignment, beat_index = self._get_beat_alignment(
                timeline_start, beat_info
            )

            cut = Cut(
                source_start=source_start,
                source_end=source_end,
                timeline_start=timeline_start,
                timeline_end=timeline_end,
                quality_score=shot.avg_quality,
                beat_alignment=beat_alignment,
                beat_index=beat_index,
                shot_index=i,
            )
            cuts.append(cut)

        logger.info(f"Built timeline with {len(cuts)} cuts")
        return cuts

    def align_to_beats(
        self,
        cuts: List[Cut],
        beat_info: BeatInfo,
        snap_threshold: float = 0.15,
    ) -> List[Cut]:
        """Adjust cut timing to align with beats.

        Args:
            cuts: List of cuts to align.
            beat_info: Beat information.
            snap_threshold: Maximum distance to snap to beat.

        Returns:
            List of aligned cuts.
        """
        if not beat_info.beat_times:
            return cuts

        aligned_cuts = []

        for cut in cuts:
            # Try to snap start time to beat
            _, nearest_beat, distance = beat_info.get_nearest_beat(cut.timeline_start)

            if distance <= snap_threshold:
                # Snap to beat
                time_diff = nearest_beat - cut.timeline_start
                new_start = nearest_beat
                new_end = cut.timeline_end + time_diff
            else:
                new_start = cut.timeline_start
                new_end = cut.timeline_end

            aligned_cut = Cut(
                source_start=cut.source_start,
                source_end=cut.source_end,
                timeline_start=new_start,
                timeline_end=new_end,
                quality_score=cut.quality_score,
                beat_alignment=cut.beat_alignment,
                beat_index=cut.beat_index,
                shot_index=cut.shot_index,
            )
            aligned_cuts.append(aligned_cut)

        return aligned_cuts

    def _get_beat_alignment(
        self,
        time: float,
        beat_info: BeatInfo,
    ) -> Tuple[str, int]:
        """Determine beat alignment type for a time.

        Args:
            time: Time in seconds.
            beat_info: Beat information.

        Returns:
            Tuple of (alignment_type, beat_index).
        """
        if not beat_info.beat_times:
            return ("none", -1)

        idx, beat_time, distance = beat_info.get_nearest_beat(time)

        # Check if on a strong beat
        if distance < 0.1 and idx < len(beat_info.beat_strength):
            if beat_info.beat_strength[idx] > 0.6:
                return ("strong_beat", idx)
            else:
                return ("beat", idx)

        # Check if near energy peak
        for peak_time in beat_info.energy_peaks:
            if abs(time - peak_time) < 0.2:
                return ("energy_peak", idx)

        return ("transition", idx)

    def create_transitions(
        self,
        cuts: List[Cut],
        beat_info: BeatInfo,
    ) -> List[Transition]:
        """Create transitions between cuts.

        Args:
            cuts: List of cuts.
            beat_info: Beat information.

        Returns:
            List of Transition objects.
        """
        transitions = []

        for i in range(len(cuts) - 1):
            current_cut = cuts[i]
            next_cut = cuts[i + 1]

            # Choose transition style
            style = self._choose_transition_style(
                current_cut, next_cut, beat_info
            )

            # Determine duration (extend on beats)
            duration = self.transition_duration
            if current_cut.beat_alignment in ("strong_beat", "energy_peak"):
                duration *= 1.5

            # Calculate transition start (overlap with cuts)
            trans_start = next_cut.timeline_start - duration / 2

            transition = Transition(
                style=style,
                duration=duration,
                timeline_start=trans_start,
                easing="ease-in-out",
            )
            transitions.append(transition)

        logger.info(f"Created {len(transitions)} transitions")
        return transitions

    def _choose_transition_style(
        self,
        current_cut: Cut,
        next_cut: Cut,
        beat_info: BeatInfo,
    ) -> str:
        """Choose transition style based on context.

        Args:
            current_cut: Current cut.
            next_cut: Next cut.
            beat_info: Beat information.

        Returns:
            Transition style name.
        """
        if self.transition_style == "dynamic":
            # Use more dramatic transitions on strong beats
            if current_cut.beat_alignment == "strong_beat":
                styles = TRANSITION_STYLES["distortion"] + TRANSITION_STYLES["camera"]
            elif current_cut.beat_alignment == "energy_peak":
                styles = TRANSITION_STYLES["organic"]
            else:
                styles = TRANSITION_STYLES["geometric"]
            return random.choice(styles)

        elif self.transition_style in TRANSITION_STYLES:
            return random.choice(TRANSITION_STYLES[self.transition_style])

        else:
            return random.choice(ALL_TRANSITIONS)


class EditBuilder:
    """Build complete Edit objects with all effects."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize edit builder.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.timeline_builder = TimelineBuilder(
            min_cut_duration=config.get("cut_timing", {}).get("min_cut_duration", 0.5),
            max_cut_duration=config.get("cut_timing", {}).get("max_cut_duration", 5.0),
            transition_duration=config.get("motion_graphics", {}).get("transitions", {}).get("duration", 0.3),
            transition_style=config.get("motion_graphics", {}).get("transitions", {}).get("style", "dynamic"),
        )

    def build_edit(
        self,
        selected_shots: List[Tuple[Shot, float, float]],
        beat_info: BeatInfo,
        target_duration: float,
        variation_id: int = 0,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Edit:
        """Build complete edit with timeline and effects.

        Args:
            selected_shots: Selected shots with timing.
            beat_info: Beat information.
            target_duration: Target duration.
            variation_id: Variation identifier.
            generation_params: Parameters used for generation.

        Returns:
            Complete Edit object.
        """
        # Build timeline
        cuts = self.timeline_builder.build_timeline(
            selected_shots, beat_info, target_duration
        )

        # Align to beats
        cuts = self.timeline_builder.align_to_beats(cuts, beat_info)

        # Create transitions
        mg_config = self.config.get("motion_graphics", {})
        transitions = []
        if mg_config.get("transitions", {}).get("enabled", True):
            transitions = self.timeline_builder.create_transitions(cuts, beat_info)

        # Create text animations
        text_animations = self._create_text_animations(cuts, beat_info)

        # Create particle effects
        particle_effects = self._create_particle_effects(beat_info, target_duration)

        # Create color grade
        color_grade = self._create_color_grade()

        # Calculate total duration
        if cuts:
            total_duration = cuts[-1].timeline_end
        else:
            total_duration = 0.0

        # Calculate average quality
        if cuts:
            avg_quality = np.mean([c.quality_score for c in cuts])
        else:
            avg_quality = 0.0

        return Edit(
            timeline=cuts,
            transitions=transitions,
            text_animations=text_animations,
            particle_effects=particle_effects,
            color_grade=color_grade,
            total_duration=total_duration,
            generation_params=generation_params or {},
            quality_score=float(avg_quality),
            variation_id=variation_id,
        )

    def _create_text_animations(
        self,
        cuts: List[Cut],
        beat_info: BeatInfo,
    ) -> List[TextAnimation]:
        """Create text animations based on config.

        Args:
            cuts: Timeline cuts.
            beat_info: Beat information.

        Returns:
            List of TextAnimation objects.
        """
        text_config = self.config.get("motion_graphics", {}).get("text", {})
        if not text_config.get("enabled", True):
            return []

        text_items = text_config.get("items", [])
        animations = []

        for item in text_items:
            if isinstance(item, str):
                # Simple text string - show at start
                text = item
                start_time = 0.0
                duration = 3.0
            elif isinstance(item, dict):
                text = item.get("text", "")
                start_time = item.get("start_time", 0.0)
                duration = item.get("duration", 3.0)
            else:
                continue

            if not text:
                continue

            animation = TextAnimation(
                text=text,
                start_time=start_time,
                duration=duration,
                entrance_animation=text_config.get("entrance_animation", "slide_in"),
                exit_animation=text_config.get("exit_animation", "fade_out"),
                font_family=text_config.get("font_family", "Arial"),
                font_size=text_config.get("font_size", 96),
                font_color=text_config.get("font_color", "#FFFFFF"),
                glow_enabled=text_config.get("glow_enabled", True),
                glow_intensity=text_config.get("glow_intensity", 0.5),
                drop_shadow=text_config.get("drop_shadow", True),
            )
            animations.append(animation)

        return animations

    def _create_particle_effects(
        self,
        beat_info: BeatInfo,
        duration: float,
    ) -> List[ParticleEffect]:
        """Create particle effects.

        Args:
            beat_info: Beat information.
            duration: Total duration.

        Returns:
            List of ParticleEffect objects.
        """
        particle_config = self.config.get("motion_graphics", {}).get("particles", {})
        if not particle_config.get("enabled", True):
            return []

        effects = []
        particle_type = particle_config.get("type", "confetti")
        density = particle_config.get("density", "medium")
        colors = particle_config.get("colors", [])
        audio_reactive = particle_config.get("audio_reactive", True)

        if audio_reactive and beat_info.energy_peaks:
            # Create particle bursts at energy peaks
            for peak_time in beat_info.energy_peaks:
                if peak_time < duration:
                    effect = ParticleEffect(
                        particle_type=particle_type,
                        start_time=peak_time,
                        duration=2.0,
                        density=density,
                        colors=colors,
                        audio_reactive=True,
                    )
                    effects.append(effect)
        else:
            # Single continuous effect
            effect = ParticleEffect(
                particle_type=particle_type,
                start_time=0.0,
                duration=duration,
                density=density,
                colors=colors,
                audio_reactive=False,
            )
            effects.append(effect)

        return effects

    def _create_color_grade(self) -> ColorGrade:
        """Create color grading settings.

        Returns:
            ColorGrade object.
        """
        cg_config = self.config.get("color_grading", {})

        if not cg_config.get("enabled", True):
            return ColorGrade()

        effects = cg_config.get("effects", {})

        return ColorGrade(
            preset=cg_config.get("preset_profile"),
            lut_file=cg_config.get("lut_file"),
            lut_strength=cg_config.get("lut_strength", 75) / 100.0,
            saturation=cg_config.get("saturation", 1.0),
            contrast=cg_config.get("contrast", 1.0),
            brightness=cg_config.get("brightness", 1.0),
            hue_shift=cg_config.get("hue_shift", 0),
            blur=effects.get("blur", 0.0),
            glow=effects.get("glow", 0.0),
            grain=effects.get("grain", 0.0),
            chromatic_aberration=effects.get("chromatic_aberration", 0.0),
            vignette=effects.get("vignette", 0.0),
        )
