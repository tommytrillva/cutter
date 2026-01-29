"""FFmpeg encoding with NVIDIA GPU acceleration for video output."""
import logging
import os
import subprocess
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detect and manage GPU capabilities."""

    _instance = None
    _gpu_info = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def detect_nvidia_gpu(self) -> Dict[str, Any]:
        """Detect NVIDIA GPU and its capabilities.

        Returns:
            Dictionary with GPU info or empty if not available.
        """
        if self._gpu_info is not None:
            return self._gpu_info

        self._gpu_info = {
            "available": False,
            "name": None,
            "nvenc_available": False,
            "nvdec_available": False,
            "cuda_available": False,
            "vram_mb": 0,
        }

        # Check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    self._gpu_info["available"] = True
                    self._gpu_info["name"] = parts[0]
                    self._gpu_info["vram_mb"] = int(parts[1])
                    logger.info(f"Detected NVIDIA GPU: {parts[0]} with {parts[1]}MB VRAM")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("nvidia-smi not available")

        # Check FFmpeg for NVENC/NVDEC support
        if self._gpu_info["available"]:
            self._gpu_info["nvenc_available"] = self._check_nvenc()
            self._gpu_info["nvdec_available"] = self._check_nvdec()

        # Check CUDA via Python
        try:
            import cv2
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            self._gpu_info["cuda_available"] = cuda_count > 0
            if cuda_count > 0:
                logger.info(f"OpenCV CUDA enabled with {cuda_count} device(s)")
        except (ImportError, AttributeError, cv2.error):
            logger.debug("OpenCV CUDA not available")

        return self._gpu_info

    def _check_nvenc(self) -> bool:
        """Check if NVENC encoder is available in FFmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "h264_nvenc" in result.stdout:
                logger.info("NVENC H.264 encoder available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False

    def _check_nvdec(self) -> bool:
        """Check if NVDEC decoder is available in FFmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-decoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "h264_cuvid" in result.stdout:
                logger.info("NVDEC H.264 decoder available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False

    @property
    def has_nvidia(self) -> bool:
        """Check if NVIDIA GPU is available."""
        return self.detect_nvidia_gpu()["available"]

    @property
    def has_nvenc(self) -> bool:
        """Check if NVENC encoding is available."""
        return self.detect_nvidia_gpu()["nvenc_available"]

    @property
    def has_cuda(self) -> bool:
        """Check if CUDA is available for OpenCV."""
        return self.detect_nvidia_gpu()["cuda_available"]


# Global GPU detector instance
gpu_detector = GPUDetector()


class FFmpegEncoder:
    """Encode video using FFmpeg with NVIDIA GPU acceleration."""

    # Platform presets with higher quality for GPU encoding
    PLATFORM_PRESETS = {
        "tiktok": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "12M",  # Higher bitrate with GPU
            "audio_bitrate": "192k",
        },
        "instagram_reels": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "12M",
            "audio_bitrate": "192k",
        },
        "youtube_shorts": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "bitrate": "15M",
            "audio_bitrate": "256k",
        },
        "youtube": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "bitrate": "20M",
            "audio_bitrate": "256k",
        },
        "youtube_4k": {
            "width": 3840,
            "height": 2160,
            "fps": 30,
            "bitrate": "50M",
            "audio_bitrate": "320k",
        },
        "web": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "bitrate": "8M",
            "audio_bitrate": "192k",
        },
    }

    # NVENC quality presets (p1=fastest, p7=slowest/best quality)
    NVENC_PRESETS = {
        "ultrafast": "p1",
        "fast": "p3",
        "medium": "p4",
        "slow": "p5",
        "quality": "p6",
        "max_quality": "p7",
    }

    def __init__(
        self,
        codec: str = "h264",
        quality: int = 18,
        fps: int = 30,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        use_gpu: bool = True,
        gpu_preset: str = "quality",
    ):
        """Initialize FFmpeg encoder with GPU support.

        Args:
            codec: Video codec (h264, h265/hevc).
            quality: CRF quality (0-51, lower is better). For NVENC, maps to CQ.
            fps: Frames per second.
            audio_codec: Audio codec.
            audio_bitrate: Audio bitrate.
            use_gpu: Enable GPU acceleration if available.
            gpu_preset: NVENC preset (ultrafast, fast, medium, slow, quality, max_quality).
        """
        self.codec = codec
        self.quality = quality
        self.fps = fps
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.use_gpu = use_gpu and gpu_detector.has_nvenc
        self.gpu_preset = gpu_preset

        # Check FFmpeg availability
        self._check_ffmpeg()

        if self.use_gpu:
            logger.info(f"GPU encoding enabled with NVENC (preset: {gpu_preset})")
        else:
            logger.info("Using CPU encoding (GPU not available or disabled)")

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

    def _get_video_codec_args(self, bitrate: Optional[str] = None) -> list:
        """Get video codec arguments based on GPU availability.

        Args:
            bitrate: Optional bitrate override.

        Returns:
            List of FFmpeg arguments for video encoding.
        """
        args = []

        if self.use_gpu:
            # NVENC encoding
            if self.codec in ("h264", "avc"):
                args.extend(["-c:v", "h264_nvenc"])
            elif self.codec in ("h265", "hevc"):
                args.extend(["-c:v", "hevc_nvenc"])
            else:
                args.extend(["-c:v", "h264_nvenc"])

            # NVENC preset
            nvenc_preset = self.NVENC_PRESETS.get(self.gpu_preset, "p5")
            args.extend(["-preset", nvenc_preset])

            # Quality settings - use CQ (constant quality) mode
            args.extend(["-rc", "vbr"])
            args.extend(["-cq", str(self.quality)])

            # B-frames for better compression
            args.extend(["-bf", "3"])

            # Use GPU memory efficiently
            args.extend(["-gpu", "0"])

            # Lookahead for better quality
            args.extend(["-rc-lookahead", "32"])

            # Spatial AQ for better quality in complex scenes
            args.extend(["-spatial-aq", "1"])

            # Temporal AQ
            args.extend(["-temporal-aq", "1"])

            if bitrate:
                args.extend(["-b:v", bitrate])
                args.extend(["-maxrate", bitrate])
                # Buffer size = 2x bitrate
                bitrate_value = int(bitrate.rstrip("MmKk")) * (1000000 if "M" in bitrate else 1000)
                args.extend(["-bufsize", str(bitrate_value * 2)])

        else:
            # CPU encoding fallback
            if self.codec in ("h264", "avc"):
                args.extend(["-c:v", "libx264"])
            elif self.codec in ("h265", "hevc"):
                args.extend(["-c:v", "libx265"])
            else:
                args.extend(["-c:v", "libx264"])

            args.extend(["-crf", str(self.quality)])
            args.extend(["-preset", "medium"])

            if bitrate:
                args.extend(["-b:v", bitrate])

        args.extend(["-pix_fmt", "yuv420p"])

        return args

    def _get_hwaccel_input_args(self, input_path: str) -> list:
        """Get hardware acceleration arguments for input decoding.

        Args:
            input_path: Input file path.

        Returns:
            List of FFmpeg arguments for hardware-accelerated decoding.
        """
        args = []

        if self.use_gpu and gpu_detector.detect_nvidia_gpu()["nvdec_available"]:
            # Use CUDA for hardware decoding
            args.extend(["-hwaccel", "cuda"])
            args.extend(["-hwaccel_output_format", "cuda"])

        return args

    def encode_from_frames(
        self,
        frames_dir: str,
        output_path: str,
        audio_path: Optional[str] = None,
        start_number: int = 0,
        bitrate: Optional[str] = None,
    ) -> str:
        """Encode video from frame sequence with GPU acceleration.

        Args:
            frames_dir: Directory containing PNG frames.
            output_path: Output video path.
            audio_path: Optional audio file to mux.
            start_number: Starting frame number.
            bitrate: Optional bitrate override.

        Returns:
            Path to output video.
        """
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

        cmd = ["ffmpeg", "-y"]

        # Input settings
        cmd.extend(["-framerate", str(self.fps)])
        cmd.extend(["-start_number", str(start_number)])
        cmd.extend(["-i", frame_pattern])

        # Add audio if provided
        if audio_path:
            cmd.extend(["-i", audio_path])

        # Video codec settings with GPU
        cmd.extend(self._get_video_codec_args(bitrate))

        # Audio codec settings
        if audio_path:
            cmd.extend([
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-shortest",
            ])

        cmd.append(output_path)

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            # Try CPU fallback if GPU fails
            if self.use_gpu:
                logger.warning("GPU encoding failed, falling back to CPU")
                self.use_gpu = False
                return self.encode_from_frames(frames_dir, output_path, audio_path, start_number, bitrate)
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

        logger.info(f"Encoded video to {output_path}")
        return output_path

    def encode_video_to_video(
        self,
        input_path: str,
        output_path: str,
        audio_path: Optional[str] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        scale: Optional[Tuple[int, int]] = None,
        bitrate: Optional[str] = None,
    ) -> str:
        """Re-encode video with GPU acceleration.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            audio_path: Optional replacement audio.
            start_time: Start time in seconds.
            duration: Duration in seconds.
            scale: Output resolution (width, height).
            bitrate: Bitrate override.

        Returns:
            Path to output video.
        """
        cmd = ["ffmpeg", "-y"]

        # Hardware acceleration for input
        cmd.extend(self._get_hwaccel_input_args(input_path))

        # Input time range
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])

        cmd.extend(["-i", input_path])

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        # Add replacement audio if provided
        if audio_path:
            cmd.extend(["-i", audio_path])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

        # Video filters
        vf_filters = []
        if scale:
            if self.use_gpu:
                # Use GPU scaling
                vf_filters.append(f"scale_cuda={scale[0]}:{scale[1]}")
            else:
                vf_filters.append(f"scale={scale[0]}:{scale[1]}")

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        # Video codec
        cmd.extend(self._get_video_codec_args(bitrate))

        # Audio codec
        cmd.extend(["-c:a", self.audio_codec, "-b:a", self.audio_bitrate])

        if audio_path:
            cmd.append("-shortest")

        cmd.append(output_path)

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if self.use_gpu:
                logger.warning("GPU encoding failed, falling back to CPU")
                self.use_gpu = False
                return self.encode_video_to_video(input_path, output_path, audio_path, start_time, duration, scale, bitrate)
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

        logger.info(f"Encoded video to {output_path}")
        return output_path

    def add_audio_track(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Add audio track to video (fast, no re-encoding).

        Args:
            video_path: Input video path.
            audio_path: Audio file path.
            output_path: Output path (defaults to replacing input).

        Returns:
            Path to output video.
        """
        if output_path is None:
            import tempfile
            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            replace_input = True
        else:
            replace_input = False

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # No video re-encoding
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
        """Encode video using platform preset with GPU acceleration.

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

        cmd = ["ffmpeg", "-y"]

        # Input
        cmd.extend(["-framerate", str(preset["fps"])])
        cmd.extend(["-i", frame_pattern])

        if audio_path:
            cmd.extend(["-i", audio_path])

        # Scale filter
        scale_filter = f"scale={preset['width']}:{preset['height']}:force_original_aspect_ratio=decrease,pad={preset['width']}:{preset['height']}:(ow-iw)/2:(oh-ih)/2"
        cmd.extend(["-vf", scale_filter])

        # Video codec with GPU and preset bitrate
        cmd.extend(self._get_video_codec_args(preset["bitrate"]))

        # Audio
        if audio_path:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", preset["audio_bitrate"],
                "-shortest",
            ])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if self.use_gpu:
                logger.warning("GPU encoding failed, falling back to CPU")
                self.use_gpu = False
                return self.encode_with_preset(frames_dir, output_path, platform, audio_path)
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

        logger.info(f"Encoded video with {platform} preset to {output_path}")
        return output_path

    def concatenate_videos(
        self,
        video_paths: list,
        output_path: str,
        audio_path: Optional[str] = None,
    ) -> str:
        """Concatenate multiple videos with GPU acceleration.

        Args:
            video_paths: List of input video paths.
            output_path: Output video path.
            audio_path: Optional audio to replace.

        Returns:
            Path to output video.
        """
        import tempfile

        # Create concat list file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            concat_file = f.name

        try:
            cmd = ["ffmpeg", "-y"]

            # Hardware acceleration
            cmd.extend(self._get_hwaccel_input_args(video_paths[0]))

            cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])

            if audio_path:
                cmd.extend(["-i", audio_path])
                cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

            # Re-encode with GPU
            cmd.extend(self._get_video_codec_args())

            cmd.extend(["-c:a", self.audio_codec, "-b:a", self.audio_bitrate])

            if audio_path:
                cmd.append("-shortest")

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        finally:
            os.unlink(concat_file)

        logger.info(f"Concatenated {len(video_paths)} videos to {output_path}")
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
            "ffmpeg", "-y",
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

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information and capabilities.

        Returns:
            Dictionary with GPU info.
        """
        return gpu_detector.detect_nvidia_gpu()
