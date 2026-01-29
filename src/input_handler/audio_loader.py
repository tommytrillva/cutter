"""Audio loading for Cutter."""
import os
import logging
from typing import Tuple, Optional
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

from ..data_structures import AudioMetadata

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load audio files for analysis."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}

    def __init__(self, target_sr: int = 22050):
        """Initialize audio loader.

        Args:
            target_sr: Target sample rate for loaded audio. Default 22050 is
                      standard for librosa analysis.
        """
        self.target_sr = target_sr

        if librosa is None:
            raise ImportError(
                "librosa is required for audio loading. "
                "Install with: pip install librosa"
            )

    def load_audio(
        self, path: str, sr: Optional[int] = None, mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Load audio file.

        Args:
            path: Path to audio file.
            sr: Target sample rate. If None, uses self.target_sr.
            mono: Convert to mono. Default True for analysis.

        Returns:
            Tuple of (audio samples as numpy array, sample rate).

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If format not supported or can't be loaded.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {ext}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        sample_rate = sr or self.target_sr

        try:
            audio, sr_loaded = librosa.load(path, sr=sample_rate, mono=mono)
        except Exception as e:
            raise ValueError(f"Could not load audio file: {path}. Error: {e}")

        logger.info(
            f"Loaded audio: {path}, {len(audio)} samples, "
            f"{sr_loaded}Hz, duration={len(audio)/sr_loaded:.2f}s"
        )

        return audio, sr_loaded

    def get_metadata(self, path: str) -> AudioMetadata:
        """Get audio metadata without loading full file.

        Args:
            path: Path to audio file.

        Returns:
            AudioMetadata object.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Load just enough to get metadata
        try:
            duration = librosa.get_duration(path=path)
            y, sr = librosa.load(path, sr=None, duration=0.1)
            channels = 1 if len(y.shape) == 1 else y.shape[0]
        except Exception as e:
            raise ValueError(f"Could not read audio metadata: {path}. Error: {e}")

        return AudioMetadata(
            path=path,
            sample_rate=sr,
            duration=duration,
            channels=channels,
        )

    def load_segment(
        self, path: str, start_time: float, end_time: float
    ) -> Tuple[np.ndarray, int]:
        """Load a segment of audio.

        Args:
            path: Path to audio file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Tuple of (audio samples, sample rate).
        """
        duration = end_time - start_time

        try:
            audio, sr = librosa.load(
                path,
                sr=self.target_sr,
                mono=True,
                offset=start_time,
                duration=duration,
            )
        except Exception as e:
            raise ValueError(
                f"Could not load audio segment [{start_time}:{end_time}] "
                f"from {path}. Error: {e}"
            )

        return audio, sr

    def extract_audio_from_video(
        self, video_path: str, output_path: str
    ) -> str:
        """Extract audio track from video file.

        Uses ffmpeg to extract audio. Requires ffmpeg installed.

        Args:
            video_path: Path to video file.
            output_path: Path for extracted audio file.

        Returns:
            Path to extracted audio file.

        Raises:
            RuntimeError: If ffmpeg extraction fails.
        """
        import subprocess

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM audio
            "-ar", "44100",  # Sample rate
            "-ac", "2",  # Stereo
            "-y",  # Overwrite
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Extracted audio from {video_path} to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to extract audio: {e.stderr}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to extract audio from video."
            )
