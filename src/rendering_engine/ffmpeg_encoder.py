"""FFmpeg encoding for video output."""
import logging
import os
import subprocess
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FFmpegEncoder:
    """Encode video using FFmpeg."""

    # Platform presets
    PLATFORM_PRESETS = {
        "tiktok": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "instagram_reels": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
        "youtube_shorts": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "10M",
            "audio_bitrate": "256k",
        },
        "youtube": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "bitrate": "12M",
            "audio_bitrate": "256k",
        },
        "web": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "bitrate": "5M",
            "audio_bitrate": "192k",
        },
    }

    def __init__(
        self,
        codec: str = "h264",
        quality: int = 18,
        fps: int = 30,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ):
        """Initialize FFmpeg encoder.

        Args:
            codec: Video codec (h264, h265).
            quality: CRF quality (0-51, lower is better).
            fps: Frames per second.
            audio_codec: Audio codec.
            audio_bitrate: Audio bitrate.
        """
        self.codec = codec
        self.quality = quality
        self.fps = fps
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

        # Check FFmpeg availability
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg returned error")
            logger.info("FFmpeg is available")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )

    def encode_from_frames(
        self,
        frames_dir: str,
        output_path: str,
        audio_path: Optional[str] = None,
        start_number: int = 0,
    ) -> str:
        """Encode video from frame sequence.

        Args:
            frames_dir: Directory containing PNG frames.
            output_path: Output video path.
            audio_path: Optional audio file to mux.
            start_number: Starting frame number.

        Returns:
            Path to output video.
        """
        # Build FFmpeg command
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(self.fps),
            "-start_number", str(start_number),
            "-i", frame_pattern,
        ]

        # Add audio if provided
        if audio_path:
            cmd.extend(["-i", audio_path])

        # Video codec settings
        if self.codec == "h264":
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(self.quality),
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
            ])
        elif self.codec == "h265":
            cmd.extend([
                "-c:v", "libx265",
                "-crf", str(self.quality),
                "-preset", "medium",
            ])

        # Audio codec settings
        if audio_path:
            cmd.extend([
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-shortest",  # End when shortest stream ends
            ])

        cmd.append(output_path)

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

        logger.info(f"Encoded video to {output_path}")
        return output_path

    def add_audio_track(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Add audio track to video.

        Args:
            video_path: Input video path.
            audio_path: Audio file path.
            output_path: Output path (defaults to replacing input).

        Returns:
            Path to output video.
        """
        if output_path is None:
            # Create temp output, then replace
            import tempfile
            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            replace_input = True
        else:
            replace_input = False

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio mux failed: {result.stderr}")

        if replace_input:
            os.replace(output_path, video_path)
            return video_path

        logger.info(f"Added audio track to {output_path}")
        return output_path

    def encode_with_preset(
        self,
        frames_dir: str,
        output_path: str,
        platform: str,
        audio_path: Optional[str] = None,
    ) -> str:
        """Encode video using platform preset.

        Args:
            frames_dir: Directory containing frames.
            output_path: Output video path.
            platform: Platform preset name.
            audio_path: Optional audio file.

        Returns:
            Path to output video.
        """
        if platform not in self.PLATFORM_PRESETS:
            raise ValueError(
                f"Unknown platform: {platform}. "
                f"Available: {list(self.PLATFORM_PRESETS.keys())}"
            )

        preset = self.PLATFORM_PRESETS[platform]
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(preset["fps"]),
            "-i", frame_pattern,
        ]

        if audio_path:
            cmd.extend(["-i", audio_path])

        # Scale if needed
        cmd.extend([
            "-vf", f"scale={preset['width']}:{preset['height']}:force_original_aspect_ratio=decrease,pad={preset['width']}:{preset['height']}:(ow-iw)/2:(oh-ih)/2",
        ])

        # Video settings
        cmd.extend([
            "-c:v", "libx264",
            "-b:v", preset["bitrate"],
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ])

        # Audio settings
        if audio_path:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", preset["audio_bitrate"],
                "-shortest",
            ])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

        logger.info(f"Encoded video with {platform} preset to {output_path}")
        return output_path

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with video information.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        import json
        return json.loads(result.stdout)

    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = "mp3",
    ) -> str:
        """Extract audio from video.

        Args:
            video_path: Input video path.
            output_path: Output audio path.
            format: Output format (mp3, wav, aac).

        Returns:
            Path to extracted audio.
        """
        codec_map = {
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "aac": "aac",
        }

        codec = codec_map.get(format, "libmp3lame")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-c:a", codec,
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")

        logger.info(f"Extracted audio to {output_path}")
        return output_path
