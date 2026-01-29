"""Format validation for Cutter."""
import logging
from typing import Dict, Any, Tuple, List

from ..data_structures import VideoMetadata, AudioMetadata

logger = logging.getLogger(__name__)


class FormatValidator:
    """Validate video and audio files meet requirements."""

    # Minimum requirements
    MIN_VIDEO_WIDTH = 1280  # 720p width
    MIN_VIDEO_HEIGHT = 720
    MIN_FPS = 24
    MIN_DURATION = 5.0  # seconds

    # Maximum limits (to prevent memory issues)
    MAX_DURATION = 3600.0  # 1 hour
    MAX_RESOLUTION = 7680  # 8K width

    def __init__(
        self,
        min_width: int = MIN_VIDEO_WIDTH,
        min_height: int = MIN_VIDEO_HEIGHT,
        min_fps: float = MIN_FPS,
        min_duration: float = MIN_DURATION,
    ):
        """Initialize validator with requirements.

        Args:
            min_width: Minimum video width.
            min_height: Minimum video height.
            min_fps: Minimum frames per second.
            min_duration: Minimum duration in seconds.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_fps = min_fps
        self.min_duration = min_duration

    def validate_video(
        self, metadata: VideoMetadata
    ) -> Tuple[bool, List[str]]:
        """Validate video metadata meets requirements.

        Args:
            metadata: VideoMetadata object.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check resolution
        if metadata.width < self.min_width:
            errors.append(
                f"Video width {metadata.width} is below minimum {self.min_width}"
            )

        if metadata.height < self.min_height:
            errors.append(
                f"Video height {metadata.height} is below minimum {self.min_height}"
            )

        if metadata.width > self.MAX_RESOLUTION:
            errors.append(
                f"Video width {metadata.width} exceeds maximum {self.MAX_RESOLUTION}"
            )

        # Check FPS
        if metadata.fps < self.min_fps:
            errors.append(
                f"Video FPS {metadata.fps:.2f} is below minimum {self.min_fps}"
            )

        # Check duration
        if metadata.duration < self.min_duration:
            errors.append(
                f"Video duration {metadata.duration:.2f}s is below "
                f"minimum {self.min_duration}s"
            )

        if metadata.duration > self.MAX_DURATION:
            errors.append(
                f"Video duration {metadata.duration:.2f}s exceeds "
                f"maximum {self.MAX_DURATION}s"
            )

        # Check frames
        if metadata.total_frames <= 0:
            errors.append("Video has no readable frames")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Video validation passed: {metadata.path}")
        else:
            logger.warning(
                f"Video validation failed for {metadata.path}: {errors}"
            )

        return is_valid, errors

    def validate_audio(
        self, metadata: AudioMetadata, video_metadata: VideoMetadata = None
    ) -> Tuple[bool, List[str]]:
        """Validate audio metadata meets requirements.

        Args:
            metadata: AudioMetadata object.
            video_metadata: Optional video metadata for duration comparison.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check sample rate
        if metadata.sample_rate < 8000:
            errors.append(
                f"Audio sample rate {metadata.sample_rate}Hz is too low. "
                f"Minimum 8000Hz recommended."
            )

        # Check duration
        if metadata.duration < self.min_duration:
            errors.append(
                f"Audio duration {metadata.duration:.2f}s is below "
                f"minimum {self.min_duration}s"
            )

        # Compare with video if provided
        if video_metadata:
            duration_diff = abs(metadata.duration - video_metadata.duration)
            if duration_diff > 1.0:
                logger.warning(
                    f"Audio duration ({metadata.duration:.2f}s) differs from "
                    f"video duration ({video_metadata.duration:.2f}s) "
                    f"by {duration_diff:.2f}s"
                )

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Audio validation passed: {metadata.path}")
        else:
            logger.warning(
                f"Audio validation failed for {metadata.path}: {errors}"
            )

        return is_valid, errors

    def get_quality_assessment(
        self, video_metadata: VideoMetadata
    ) -> Dict[str, Any]:
        """Get quality assessment of video.

        Args:
            video_metadata: VideoMetadata object.

        Returns:
            Dictionary with quality metrics.
        """
        # Calculate quality score components
        resolution_score = min(
            1.0,
            (video_metadata.width * video_metadata.height) / (1920 * 1080)
        )

        fps_score = min(1.0, video_metadata.fps / 60.0)

        # Estimate bitrate (rough approximation)
        estimated_bitrate = (
            video_metadata.file_size * 8 / video_metadata.duration
            if video_metadata.duration > 0 else 0
        )
        bitrate_mbps = estimated_bitrate / 1_000_000

        # Quality tier assessment
        if video_metadata.width >= 3840:
            resolution_tier = "4K"
        elif video_metadata.width >= 1920:
            resolution_tier = "1080p"
        elif video_metadata.width >= 1280:
            resolution_tier = "720p"
        else:
            resolution_tier = "SD"

        return {
            "resolution_tier": resolution_tier,
            "resolution_score": resolution_score,
            "fps_score": fps_score,
            "estimated_bitrate_mbps": round(bitrate_mbps, 2),
            "is_portrait": video_metadata.is_portrait,
            "aspect_ratio": round(video_metadata.aspect_ratio, 3),
            "overall_quality": round(
                (resolution_score * 0.6 + fps_score * 0.4), 3
            ),
        }
