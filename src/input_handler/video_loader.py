"""Video loading and frame extraction for Cutter."""
import os
import logging
from typing import Dict, Any, Generator, Optional, Tuple
import cv2
import numpy as np

from ..data_structures import VideoMetadata

logger = logging.getLogger(__name__)


class VideoLoader:
    """Load video files and extract frames."""

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}

    def __init__(self):
        """Initialize video loader."""
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_path: Optional[str] = None

    def load_video(self, path: str) -> Tuple[np.ndarray, VideoMetadata]:
        """Load entire video into memory.

        Warning: This loads ALL frames into memory. Use load_video_chunked
        for large videos.

        Args:
            path: Path to video file.

        Returns:
            Tuple of (frames array, metadata).

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video format not supported or can't be read.
        """
        metadata = self.get_metadata(path)
        frames = []

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from video: {path}")

        logger.info(f"Loaded {len(frames)} frames from {path}")
        return np.array(frames), metadata

    def load_video_chunked(
        self, path: str, chunk_size: int = 100
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Load video frames in chunks.

        Args:
            path: Path to video file.
            chunk_size: Number of frames per chunk.

        Yields:
            Tuple of (frames array for chunk, starting frame index).

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video can't be read.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            frame_idx = 0
            while True:
                frames = []
                start_idx = frame_idx

                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_idx += 1

                if not frames:
                    break

                yield np.array(frames), start_idx
        finally:
            cap.release()

    def get_metadata(self, path: str) -> VideoMetadata:
        """Get video metadata without loading frames.

        Args:
            path: Path to video file.

        Returns:
            VideoMetadata object.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video can't be read or format not supported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {ext}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode fourcc to codec string
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            duration = total_frames / fps if fps > 0 else 0
            file_size = os.path.getsize(path)

            metadata = VideoMetadata(
                path=path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                codec=codec.strip(),
                total_frames=total_frames,
                file_size=file_size,
            )

            logger.info(
                f"Video metadata: {width}x{height}, {fps:.2f}fps, "
                f"{duration:.2f}s, {total_frames} frames, codec={codec}"
            )

            return metadata

        finally:
            cap.release()

    def get_frame(self, path: str, frame_index: int) -> np.ndarray:
        """Get a specific frame from the video.

        Args:
            path: Path to video file.
            frame_index: Index of frame to retrieve.

        Returns:
            Frame as RGB numpy array.

        Raises:
            ValueError: If frame can't be read.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Could not read frame {frame_index} from {path}")

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def get_frame_at_time(self, path: str, time_seconds: float) -> np.ndarray:
        """Get frame at a specific time.

        Args:
            path: Path to video file.
            time_seconds: Time in seconds.

        Returns:
            Frame as RGB numpy array.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(
                    f"Could not read frame at {time_seconds}s from {path}"
                )

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def extract_frames_at_times(
        self, path: str, times: list
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Extract frames at specific times.

        Args:
            path: Path to video file.
            times: List of times in seconds.

        Yields:
            Tuple of (frame, time).
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            for t in sorted(times):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()

                if ret:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t
                else:
                    logger.warning(f"Could not read frame at {t}s")
        finally:
            cap.release()

    def sample_frames(
        self, path: str, num_samples: int = 100
    ) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """Sample frames evenly distributed across video.

        Args:
            path: Path to video file.
            num_samples: Number of frames to sample.

        Yields:
            Tuple of (frame, frame_index, timestamp).
        """
        metadata = self.get_metadata(path)
        total_frames = metadata.total_frames
        fps = metadata.fps

        if num_samples >= total_frames:
            # Return all frames
            indices = range(total_frames)
        else:
            # Evenly distribute samples
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()

                if ret:
                    timestamp = idx / fps if fps > 0 else 0
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(idx), timestamp
        finally:
            cap.release()
