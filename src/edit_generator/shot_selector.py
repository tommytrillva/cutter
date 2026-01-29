"""Shot selection for edit generation."""
import logging
from typing import List, Optional, Tuple
import numpy as np

from ..data_structures import Shot, BeatInfo, FrameQuality

logger = logging.getLogger(__name__)


class ShotSelector:
    """Select best shots for the edit timeline."""

    def __init__(
        self,
        min_quality: float = 0.4,
        prefer_variety: bool = True,
        beat_alignment_weight: float = 0.3,
    ):
        """Initialize shot selector.

        Args:
            min_quality: Minimum quality score for shot inclusion.
            prefer_variety: Avoid consecutive use of same shots.
            beat_alignment_weight: Weight for beat alignment in selection.
        """
        self.min_quality = min_quality
        self.prefer_variety = prefer_variety
        self.beat_alignment_weight = beat_alignment_weight

    def select_shots(
        self,
        shots: List[Shot],
        target_duration: float,
        beat_info: Optional[BeatInfo] = None,
    ) -> List[Tuple[Shot, float, float]]:
        """Select shots to fill target duration.

        Args:
            shots: Available shots to choose from.
            target_duration: Target video duration in seconds.
            beat_info: Beat information for alignment.

        Returns:
            List of (shot, start_time, end_time) tuples.
        """
        # Filter shots by quality
        qualified_shots = [s for s in shots if s.avg_quality >= self.min_quality]

        if not qualified_shots:
            logger.warning(
                f"No shots meet quality threshold {self.min_quality}. "
                f"Using all {len(shots)} shots."
            )
            qualified_shots = shots

        # Sort by quality (best first)
        qualified_shots.sort(key=lambda s: s.avg_quality, reverse=True)

        logger.info(
            f"Selecting from {len(qualified_shots)} qualified shots "
            f"(min quality: {self.min_quality})"
        )

        # Select shots to fill duration
        selected = self._fill_duration(
            qualified_shots, target_duration, beat_info
        )

        return selected

    def _fill_duration(
        self,
        shots: List[Shot],
        target_duration: float,
        beat_info: Optional[BeatInfo],
    ) -> List[Tuple[Shot, float, float]]:
        """Fill target duration with shots.

        Args:
            shots: Sorted list of shots (best first).
            target_duration: Target duration in seconds.
            beat_info: Beat information.

        Returns:
            List of (shot, timeline_start, timeline_end) tuples.
        """
        selected = []
        current_time = 0.0
        used_shots = set()

        while current_time < target_duration and shots:
            remaining_time = target_duration - current_time

            # Find best shot for this position
            best_shot, best_score = self._find_best_shot(
                shots, used_shots, current_time, remaining_time, beat_info
            )

            if best_shot is None:
                # No more suitable shots, reuse shots
                used_shots.clear()
                best_shot, best_score = self._find_best_shot(
                    shots, used_shots, current_time, remaining_time, beat_info
                )

            if best_shot is None:
                break

            # Calculate cut duration
            cut_duration = min(best_shot.duration, remaining_time)

            # Align to beat if possible
            if beat_info and cut_duration > 0.5:
                cut_duration = self._align_duration_to_beat(
                    current_time, cut_duration, beat_info
                )

            end_time = current_time + cut_duration
            selected.append((best_shot, current_time, end_time))

            if self.prefer_variety:
                used_shots.add(id(best_shot))

            current_time = end_time

        logger.info(f"Selected {len(selected)} cuts totaling {current_time:.2f}s")
        return selected

    def _find_best_shot(
        self,
        shots: List[Shot],
        used_shots: set,
        current_time: float,
        remaining_time: float,
        beat_info: Optional[BeatInfo],
    ) -> Tuple[Optional[Shot], float]:
        """Find best shot for current position.

        Args:
            shots: Available shots.
            used_shots: Set of already-used shot IDs.
            current_time: Current timeline position.
            remaining_time: Remaining time to fill.
            beat_info: Beat information.

        Returns:
            Tuple of (best_shot, score) or (None, 0).
        """
        best_shot = None
        best_score = -1

        for shot in shots:
            if id(shot) in used_shots:
                continue

            # Base score is quality
            score = shot.avg_quality

            # Bonus for shots that fit well in remaining time
            duration_fit = min(1.0, shot.duration / max(remaining_time, 0.5))
            score += duration_fit * 0.2

            # Bonus for beat alignment
            if beat_info and self.beat_alignment_weight > 0:
                alignment_score = self._calculate_beat_alignment(
                    shot, current_time, beat_info
                )
                score += alignment_score * self.beat_alignment_weight

            if score > best_score:
                best_score = score
                best_shot = shot

        return best_shot, best_score

    def _calculate_beat_alignment(
        self,
        shot: Shot,
        current_time: float,
        beat_info: BeatInfo,
    ) -> float:
        """Calculate how well a shot aligns with beats.

        Args:
            shot: Shot to evaluate.
            current_time: Current timeline position.
            beat_info: Beat information.

        Returns:
            Alignment score 0-1.
        """
        if not beat_info.beat_times:
            return 0.5

        # Find nearest beat to current time
        _, nearest_beat_time, distance = beat_info.get_nearest_beat(current_time)

        # Score based on distance to beat (closer = better)
        # Use beat period as reference
        beat_period = 60.0 / beat_info.bpm if beat_info.bpm > 0 else 0.5
        alignment = 1.0 - min(1.0, distance / (beat_period / 2))

        return alignment

    def _align_duration_to_beat(
        self,
        start_time: float,
        duration: float,
        beat_info: BeatInfo,
    ) -> float:
        """Align cut duration to end on a beat.

        Args:
            start_time: Cut start time.
            duration: Proposed duration.
            beat_info: Beat information.

        Returns:
            Adjusted duration.
        """
        end_time = start_time + duration

        # Find nearest beat to proposed end
        _, nearest_beat, distance = beat_info.get_nearest_beat(end_time)

        # Only adjust if beat is reasonably close
        beat_period = 60.0 / beat_info.bpm if beat_info.bpm > 0 else 0.5
        if distance < beat_period / 2:
            adjusted_duration = nearest_beat - start_time
            # Ensure minimum duration
            if adjusted_duration >= 0.3:
                return adjusted_duration

        return duration

    def select_shots_by_energy(
        self,
        shots: List[Shot],
        beat_info: BeatInfo,
        num_cuts: int,
    ) -> List[Shot]:
        """Select shots based on energy peaks.

        Places high-quality shots at energy peaks.

        Args:
            shots: Available shots.
            beat_info: Beat information with energy peaks.
            num_cuts: Number of cuts to select.

        Returns:
            List of selected shots.
        """
        # Sort shots by quality
        sorted_shots = sorted(shots, key=lambda s: s.avg_quality, reverse=True)

        # Get energy peak count
        num_peaks = len(beat_info.energy_peaks)

        # Allocate best shots to peaks
        peak_shots = sorted_shots[:min(num_peaks, len(sorted_shots))]

        # Fill remaining with other shots
        remaining_needed = num_cuts - len(peak_shots)
        other_shots = sorted_shots[len(peak_shots):len(peak_shots) + remaining_needed]

        selected = peak_shots + other_shots

        logger.info(
            f"Selected {len(peak_shots)} shots for energy peaks, "
            f"{len(other_shots)} additional shots"
        )

        return selected

    def diversify_selection(
        self,
        shots: List[Shot],
        selection: List[Shot],
        diversity_factor: float = 0.3,
    ) -> List[Shot]:
        """Add variety to shot selection.

        Args:
            shots: All available shots.
            selection: Current selection.
            diversity_factor: Portion of shots to replace for variety.

        Returns:
            Diversified selection.
        """
        if len(shots) <= len(selection):
            return selection

        num_to_replace = int(len(selection) * diversity_factor)
        if num_to_replace == 0:
            return selection

        # Find shots not in selection
        selection_ids = {id(s) for s in selection}
        unused = [s for s in shots if id(s) not in selection_ids]

        if not unused:
            return selection

        # Replace some lower-quality shots with unused ones
        result = selection.copy()
        result.sort(key=lambda s: s.avg_quality)

        for i in range(min(num_to_replace, len(unused))):
            # Replace lowest quality shot in selection
            if unused[i].avg_quality >= result[i].avg_quality * 0.8:
                result[i] = unused[i]

        return result
