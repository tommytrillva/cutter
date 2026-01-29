"""Quality scoring for shots and frames."""
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
import numpy as np

from .motion_detector import MotionDetector
from .face_detector import FaceDetector
from .composition_analyzer import CompositionAnalyzer
from ..data_structures import FrameQuality, Shot, VideoMetadata

logger = logging.getLogger(__name__)


class QualityScorer:
    """Score frames and shots for visual quality."""

    def __init__(
        self,
        motion_weight: float = 0.25,
        face_weight: float = 0.25,
        composition_weight: float = 0.15,
        contrast_weight: float = 0.25,
        lighting_weight: float = 0.10,
    ):
        """Initialize quality scorer.

        Args:
            motion_weight: Weight for motion score.
            face_weight: Weight for face detection score.
            composition_weight: Weight for composition score.
            contrast_weight: Weight for contrast score.
            lighting_weight: Weight for lighting score.
        """
        # Normalize weights to sum to 1
        total = motion_weight + face_weight + composition_weight + contrast_weight + lighting_weight
        self.motion_weight = motion_weight / total
        self.face_weight = face_weight / total
        self.composition_weight = composition_weight / total
        self.contrast_weight = contrast_weight / total
        self.lighting_weight = lighting_weight / total

        # Initialize detectors
        self.motion_detector = MotionDetector()
        self.face_detector = FaceDetector()
        self.composition_analyzer = CompositionAnalyzer()

        self._prev_frame: Optional[np.ndarray] = None

    def score_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> FrameQuality:
        """Score a single frame.

        Args:
            frame: Frame as RGB numpy array.
            prev_frame: Previous frame for motion detection.
            frame_index: Index of the frame.
            timestamp: Timestamp in seconds.

        Returns:
            FrameQuality object with all scores.
        """
        # Use stored prev_frame if not provided
        if prev_frame is None:
            prev_frame = self._prev_frame

        # Calculate individual scores
        if prev_frame is not None:
            motion_score = self.motion_detector.calculate_motion(frame, prev_frame)
        else:
            motion_score = 0.5  # Neutral score for first frame

        face_score = self.face_detector.calculate_face_score(frame)
        composition_score = self.composition_analyzer.analyze_composition(frame)
        contrast_score = self.composition_analyzer.analyze_contrast(frame)
        lighting_score = self.composition_analyzer.analyze_lighting(frame)

        # Calculate weighted composite score
        composite_score = (
            motion_score * self.motion_weight +
            face_score * self.face_weight +
            composition_score * self.composition_weight +
            contrast_score * self.contrast_weight +
            lighting_score * self.lighting_weight
        )

        # Store current frame for next iteration
        self._prev_frame = frame

        return FrameQuality(
            frame_index=frame_index,
            timestamp=timestamp,
            motion_score=float(motion_score),
            face_score=float(face_score),
            composition_score=float(composition_score),
            contrast_score=float(contrast_score),
            lighting_score=float(lighting_score),
            composite_score=float(composite_score),
        )

    def score_all_frames(
        self,
        frame_generator: Generator,
        metadata: VideoMetadata,
        sample_rate: int = 1,
    ) -> List[FrameQuality]:
        """Score all frames from a generator.

        Args:
            frame_generator: Generator yielding (frame, index, timestamp).
            metadata: Video metadata.
            sample_rate: Process every Nth frame.

        Returns:
            List of FrameQuality objects.
        """
        scores = []
        prev_frame = None
        processed = 0

        for frame, idx, timestamp in frame_generator:
            if idx % sample_rate != 0:
                continue

            quality = self.score_frame(
                frame,
                prev_frame,
                frame_index=idx,
                timestamp=timestamp,
            )
            scores.append(quality)
            prev_frame = frame
            processed += 1

            if processed % 100 == 0:
                logger.info(f"Scored {processed} frames...")

        logger.info(f"Scored {len(scores)} frames total")
        return scores

    def score_frames_chunked(
        self,
        frames: np.ndarray,
        start_index: int,
        fps: float,
    ) -> List[FrameQuality]:
        """Score a chunk of frames.

        Args:
            frames: Array of frames (N, H, W, C).
            start_index: Starting frame index.
            fps: Frames per second.

        Returns:
            List of FrameQuality objects.
        """
        scores = []

        for i, frame in enumerate(frames):
            idx = start_index + i
            timestamp = idx / fps

            prev_frame = frames[i - 1] if i > 0 else self._prev_frame

            quality = self.score_frame(
                frame,
                prev_frame,
                frame_index=idx,
                timestamp=timestamp,
            )
            scores.append(quality)

        # Store last frame for next chunk
        if len(frames) > 0:
            self._prev_frame = frames[-1]

        return scores

    def detect_shots(
        self,
        frame_qualities: List[FrameQuality],
        scene_threshold: float = 0.5,
        min_shot_length: int = 10,
    ) -> List[Shot]:
        """Detect shot boundaries and create shot objects.

        Args:
            frame_qualities: List of frame quality scores.
            scene_threshold: Threshold for scene change detection.
            min_shot_length: Minimum frames for a valid shot.

        Returns:
            List of Shot objects.
        """
        if not frame_qualities:
            return []

        shots = []
        current_shot_start = 0

        for i in range(1, len(frame_qualities)):
            current = frame_qualities[i]
            previous = frame_qualities[i - 1]

            # Detect scene change based on quality score difference
            score_diff = abs(current.composite_score - previous.composite_score)

            # Also check motion - sudden drops might indicate cuts
            motion_drop = previous.motion_score > 0.3 and current.motion_score < 0.1

            is_scene_change = score_diff > scene_threshold or motion_drop

            if is_scene_change:
                # End current shot if it's long enough
                shot_length = i - current_shot_start
                if shot_length >= min_shot_length:
                    shot = self._create_shot(
                        frame_qualities[current_shot_start:i]
                    )
                    shots.append(shot)

                current_shot_start = i

        # Add final shot
        if len(frame_qualities) - current_shot_start >= min_shot_length:
            shot = self._create_shot(
                frame_qualities[current_shot_start:]
            )
            shots.append(shot)

        logger.info(f"Detected {len(shots)} shots")
        return shots

    def _create_shot(
        self, frame_qualities: List[FrameQuality]
    ) -> Shot:
        """Create a Shot object from frame qualities.

        Args:
            frame_qualities: Frame qualities for this shot.

        Returns:
            Shot object.
        """
        scores = [fq.composite_score for fq in frame_qualities]
        avg_quality = float(np.mean(scores))
        peak_quality = float(np.max(scores))
        peak_idx = int(np.argmax(scores))

        return Shot(
            start_frame=frame_qualities[0].frame_index,
            end_frame=frame_qualities[-1].frame_index,
            start_time=frame_qualities[0].timestamp,
            end_time=frame_qualities[-1].timestamp,
            avg_quality=avg_quality,
            peak_quality=peak_quality,
            peak_frame=frame_qualities[peak_idx].frame_index,
        )

    def get_best_shots(
        self,
        shots: List[Shot],
        min_quality: float = 0.4,
        top_n: Optional[int] = None,
    ) -> List[Shot]:
        """Get best shots above quality threshold.

        Args:
            shots: List of Shot objects.
            min_quality: Minimum average quality score.
            top_n: Return only top N shots (or all if None).

        Returns:
            List of best shots sorted by quality.
        """
        # Filter by minimum quality
        good_shots = [s for s in shots if s.avg_quality >= min_quality]

        # Sort by average quality (descending)
        good_shots.sort(key=lambda s: s.avg_quality, reverse=True)

        if top_n is not None:
            good_shots = good_shots[:top_n]

        logger.info(
            f"Selected {len(good_shots)} shots with quality >= {min_quality}"
        )
        return good_shots

    def get_quality_summary(
        self, frame_qualities: List[FrameQuality]
    ) -> Dict[str, Any]:
        """Get summary statistics of quality scores.

        Args:
            frame_qualities: List of frame quality scores.

        Returns:
            Dictionary with summary statistics.
        """
        if not frame_qualities:
            return {"error": "No frames analyzed"}

        composite_scores = [fq.composite_score for fq in frame_qualities]
        motion_scores = [fq.motion_score for fq in frame_qualities]
        face_scores = [fq.face_score for fq in frame_qualities]

        return {
            "total_frames": len(frame_qualities),
            "composite": {
                "mean": float(np.mean(composite_scores)),
                "std": float(np.std(composite_scores)),
                "min": float(np.min(composite_scores)),
                "max": float(np.max(composite_scores)),
                "median": float(np.median(composite_scores)),
            },
            "motion": {
                "mean": float(np.mean(motion_scores)),
                "high_motion_frames": sum(1 for s in motion_scores if s > 0.5),
            },
            "faces": {
                "mean": float(np.mean(face_scores)),
                "frames_with_faces": sum(1 for s in face_scores if s > 0.1),
            },
        }

    def reset(self):
        """Reset internal state."""
        self._prev_frame = None

    def close(self):
        """Release resources."""
        self.face_detector.close()
