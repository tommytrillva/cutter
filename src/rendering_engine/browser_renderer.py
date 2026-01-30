"""Browser-based video rendering using Playwright."""
import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Install with: pip install playwright")


class BrowserVideoRenderer:
    """Render video frames using Chromium browser."""

    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
        headless: bool = True,
    ):
        """Initialize browser renderer.

        Args:
            width: Output video width.
            height: Output video height.
            fps: Frames per second.
            headless: Run browser in headless mode.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for browser rendering. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.headless = headless

        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    async def _init_browser(self):
        """Initialize browser instance."""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )
        logger.info("Browser initialized")

    async def _init_page(self, html_path: str):
        """Initialize page with HTML template.

        Args:
            html_path: Path to HTML template file.
        """
        if self._browser is None:
            await self._init_browser()

        self._page = await self._browser.new_page(
            viewport={"width": self.width, "height": self.height}
        )

        # Load HTML template
        html_url = f"file://{os.path.abspath(html_path)}"
        await self._page.goto(html_url)
        await self._page.wait_for_load_state("networkidle")

        logger.info(f"Loaded template: {html_path}")

    async def render_timeline(
        self,
        timeline_json: str,
        html_template: str,
        output_dir: str,
        source_video: str,
    ) -> str:
        """Render timeline to PNG frame sequence.

        Args:
            timeline_json: Path to timeline JSON file.
            html_template: Path to HTML template.
            output_dir: Directory for output frames.
            source_video: Path to source video file.

        Returns:
            Path to output directory with frames.
        """
        # Load timeline
        with open(timeline_json, 'r') as f:
            timeline = json.load(f)

        # Initialize browser and page
        await self._init_page(html_template)

        # Inject timeline data (without source video - we'll inject frames)
        await self._page.evaluate(f"""
            window.timelineData = {json.dumps(timeline)};
            window.sourceVideo = null;
            window.useFrameInjection = true;
        """)

        # Wait for initialization
        await self._page.wait_for_function("window.rendererReady === true", timeout=30000)

        # Calculate total frames
        total_duration = timeline.get("metadata", {}).get("total_duration", 60)
        total_frames = int(total_duration * self.fps)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open source video for frame extraction
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")

        logger.info(f"Rendering {total_frames} frames with server-side frame extraction...")

        timeline_clips = timeline.get("timeline", [])

        # Render each frame
        for frame_idx in range(total_frames):
            frame_time = frame_idx / self.fps

            # Extract frame from source video at correct time
            frame_data_url = self._extract_frame_as_data_url(
                cap, timeline_clips, frame_time, self.width, self.height
            )

            # Inject frame into browser
            await self._page.evaluate(f"""
                window.currentFrameDataUrl = "{frame_data_url}";
            """)

            # Call render function in browser
            await self._page.evaluate(f"window.renderFrame({frame_idx}, {frame_time})")

            # Capture screenshot
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            await self._page.screenshot(path=frame_path)

            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames) * 100
                logger.info(f"Rendered frame {frame_idx}/{total_frames} ({pct:.1f}%)")

        cap.release()
        logger.info(f"Rendered {total_frames} frames to {output_dir}")
        return output_dir

    def _extract_frame_as_data_url(
        self,
        cap,
        timeline_clips: list,
        frame_time: float,
        width: int,
        height: int,
    ) -> str:
        """Extract a frame from source video and convert to data URL.

        Args:
            cap: OpenCV VideoCapture object.
            timeline_clips: List of timeline clips.
            frame_time: Current time in seconds.
            width: Output width.
            height: Output height.

        Returns:
            Base64 data URL of the frame.
        """
        # Find active clip at this time
        active_clip = None
        for clip in timeline_clips:
            if clip.get("type") == "video_clip":
                start = clip.get("timeline_start", 0)
                end = clip.get("timeline_end", 0)
                if start <= frame_time < end:
                    active_clip = clip
                    break

        if active_clip:
            # Calculate source time
            source_time = active_clip["source_start"] + (frame_time - active_clip["timeline_start"])
            cap.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000)
            ret, frame = cap.read()

            if ret:
                # Resize to output dimensions
                frame = cv2.resize(frame, (width, height))
            else:
                # Black frame if read fails
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Black frame between clips
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert BGR to RGB for browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode to JPEG (faster than PNG for large frames)
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Convert to base64 data URL
        base64_data = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

    async def render_single_frame(
        self,
        timeline_data: Dict[str, Any],
        frame_index: int,
        html_template: str,
        output_path: str,
        source_video: str,
    ) -> str:
        """Render a single frame.

        Args:
            timeline_data: Timeline data dictionary.
            frame_index: Frame index to render.
            html_template: Path to HTML template.
            output_path: Output PNG path.
            source_video: Path to source video.

        Returns:
            Path to rendered frame.
        """
        await self._init_page(html_template)

        # Inject data
        await self._page.evaluate(f"""
            window.timelineData = {json.dumps(timeline_data)};
            window.sourceVideo = "{source_video}";
        """)

        await self._page.wait_for_function("window.rendererReady === true", timeout=30000)

        frame_time = frame_index / self.fps
        await self._page.evaluate(f"window.renderFrame({frame_index}, {frame_time})")

        await self._page.screenshot(path=output_path)

        return output_path

    async def close(self):
        """Close browser and release resources."""
        if self._page:
            await self._page.close()
            self._page = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if hasattr(self, '_playwright'):
            await self._playwright.stop()

        logger.info("Browser closed")

    def render_timeline_sync(
        self,
        timeline_json: str,
        html_template: str,
        output_dir: str,
        source_video: str,
    ) -> str:
        """Synchronous wrapper for render_timeline.

        Args:
            timeline_json: Path to timeline JSON file.
            html_template: Path to HTML template.
            output_dir: Directory for output frames.
            source_video: Path to source video file.

        Returns:
            Path to output directory with frames.
        """
        return asyncio.run(self.render_timeline(
            timeline_json, html_template, output_dir, source_video
        ))


class SimpleBrowserRenderer:
    """Simplified renderer for testing without full Playwright setup."""

    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
    ):
        """Initialize simple renderer.

        Args:
            width: Output width.
            height: Output height.
            fps: Frames per second.
        """
        self.width = width
        self.height = height
        self.fps = fps

    def render_placeholder_frames(
        self,
        timeline_data: Dict[str, Any],
        output_dir: str,
        source_video: str,
    ) -> str:
        """Render placeholder frames (for testing).

        Args:
            timeline_data: Timeline data.
            output_dir: Output directory.
            source_video: Source video path.

        Returns:
            Path to output directory.
        """
        import cv2
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        total_duration = timeline_data.get("metadata", {}).get("total_duration", 60)
        total_frames = int(total_duration * self.fps)

        # Open source video
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")

        timeline = timeline_data.get("timeline", [])
        color_grading = timeline_data.get("color_grading", {})

        for frame_idx in range(total_frames):
            frame_time = frame_idx / self.fps

            # Find active clip at this time
            active_clip = None
            for clip in timeline:
                if clip.get("type") == "video_clip":
                    start = clip.get("timeline_start", 0)
                    end = clip.get("timeline_end", 0)
                    if start <= frame_time < end:
                        active_clip = clip
                        break

            if active_clip:
                # Read frame from source
                source_time = active_clip["source_start"] + (frame_time - active_clip["timeline_start"])
                cap.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000)
                ret, frame = cap.read()

                if ret:
                    # Resize to output dimensions
                    frame = cv2.resize(frame, (self.width, self.height))

                    # Apply simple color grading
                    if color_grading.get("preset") == "cinematic":
                        # Simple cinematic look: reduce saturation, add contrast
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255).astype(np.uint8)
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=-10)
                else:
                    # Black frame if read fails
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                # Black frame between clips
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)

            if frame_idx % 100 == 0:
                logger.info(f"Rendered frame {frame_idx}/{total_frames}")

        cap.release()
        logger.info(f"Rendered {total_frames} placeholder frames to {output_dir}")
        return output_dir


class StreamingRenderer:
    """High-performance renderer that streams directly to FFmpeg.

    No intermediate files - frames are piped directly to FFmpeg,
    eliminating disk space issues with large videos.
    """

    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
    ):
        """Initialize streaming renderer.

        Args:
            width: Output width.
            height: Output height.
            fps: Frames per second.
        """
        self.width = width
        self.height = height
        self.fps = fps

    def render_video(
        self,
        timeline_data: Dict[str, Any],
        output_path: str,
        source_video: str,
        audio_path: Optional[str] = None,
        platform: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Render video by streaming frames directly to FFmpeg.

        Args:
            timeline_data: Timeline data dictionary.
            output_path: Output video path.
            source_video: Source video path.
            audio_path: Optional audio file to mux.
            platform: Platform preset for encoding settings.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            Path to output video.
        """
        import cv2
        import numpy as np
        import subprocess

        # Get dimensions from timeline or use defaults
        metadata = timeline_data.get("metadata", {})
        output_width = self.width
        output_height = self.height
        fps = metadata.get("fps", self.fps)
        total_duration = metadata.get("total_duration", 60)
        total_frames = int(total_duration * fps)

        # Platform presets for encoding
        platform_settings = {
            "tiktok": {"width": 1080, "height": 1920, "bitrate": "12M"},
            "instagram_reels": {"width": 1080, "height": 1920, "bitrate": "12M"},
            "youtube_shorts": {"width": 1080, "height": 1920, "bitrate": "15M"},
            "youtube": {"width": 1920, "height": 1080, "bitrate": "20M"},
            "youtube_4k": {"width": 3840, "height": 2160, "bitrate": "50M"},
            "web": {"width": 1920, "height": 1080, "bitrate": "8M"},
        }

        if platform and platform in platform_settings:
            settings = platform_settings[platform]
            output_width = settings["width"]
            output_height = settings["height"]
            bitrate = settings["bitrate"]
        else:
            bitrate = "15M"

        # Open source video
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")

        # Check for hardware encoders
        has_nvenc = self._check_nvenc()
        has_videotoolbox = self._check_videotoolbox()

        # Build FFmpeg command for piped input
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            # Input: raw video from pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{output_width}x{output_height}",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
        ]

        # Add audio input if provided
        if audio_path:
            ffmpeg_cmd.extend(["-i", audio_path])

        # Video encoding settings - prefer NVENC (fastest), then VideoToolbox, then CPU
        if has_nvenc:
            # Use NVIDIA hardware encoder (RTX 4090, etc.)
            ffmpeg_cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p4",  # p1=fastest, p7=best quality
                "-rc", "vbr",
                "-cq", "18",
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", str(int(bitrate.rstrip("M")) * 2) + "M",
                "-profile:v", "high",
                "-bf", "3",  # B-frames for better compression
                "-spatial-aq", "1",
                "-temporal-aq", "1",
            ])
            logger.info("Using NVENC hardware encoding (NVIDIA GPU)")
        elif has_videotoolbox:
            # Use macOS hardware encoder
            ffmpeg_cmd.extend([
                "-c:v", "h264_videotoolbox",
                "-b:v", bitrate,
                "-profile:v", "high",
            ])
            logger.info("Using VideoToolbox hardware encoding")
        else:
            # CPU fallback
            ffmpeg_cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
            ])
            logger.info("Using CPU encoding (libx264)")
            logger.info("Using CPU encoding (libx264)")

        ffmpeg_cmd.extend(["-pix_fmt", "yuv420p"])

        # Audio encoding
        if audio_path:
            ffmpeg_cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
            ])

        ffmpeg_cmd.append(output_path)

        logger.info(f"Starting FFmpeg: {' '.join(ffmpeg_cmd[:15])}...")
        logger.info(f"Rendering {total_frames} frames at {output_width}x{output_height}")

        # Start FFmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        timeline = timeline_data.get("timeline", [])
        color_grading = timeline_data.get("color_grading", {})
        transitions = timeline_data.get("transitions", [])

        try:
            for frame_idx in range(total_frames):
                frame_time = frame_idx / fps

                # Find active clip at this time
                frame = self._get_frame_at_time(
                    cap, timeline, frame_time, output_width, output_height
                )

                # Apply color grading
                frame = self._apply_color_grading(frame, color_grading)

                # Apply transitions
                frame = self._apply_transitions(
                    frame, transitions, frame_time, cap, timeline, output_width, output_height
                )

                # Write frame to FFmpeg
                process.stdin.write(frame.tobytes())

                # Progress logging
                if frame_idx % 100 == 0:
                    pct = (frame_idx / total_frames) * 100
                    logger.info(f"Rendering: {frame_idx}/{total_frames} ({pct:.1f}%)")

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        except BrokenPipeError:
            # FFmpeg crashed or closed
            stderr = process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg pipe broken: {stderr}")

        finally:
            cap.release()
            if process.stdin:
                process.stdin.close()

        # Wait for FFmpeg to finish
        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg encoding failed: {stderr}")

        logger.info(f"Rendered video to {output_path}")
        return output_path

    def _check_videotoolbox(self) -> bool:
        """Check if VideoToolbox encoder is available (macOS)."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "h264_videotoolbox" in result.stdout
        except Exception:
            return False

    def _check_nvenc(self) -> bool:
        """Check if NVENC encoder is available (NVIDIA GPU)."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    def _get_frame_at_time(
        self,
        cap,
        timeline: list,
        frame_time: float,
        width: int,
        height: int,
    ):
        """Get the appropriate frame for a given time.

        Args:
            cap: OpenCV VideoCapture object.
            timeline: List of timeline clips.
            frame_time: Current time in seconds.
            width: Output width.
            height: Output height.

        Returns:
            Frame as numpy array (BGR).
        """
        import cv2
        import numpy as np

        # Find active clip at this time
        active_clip = None
        for clip in timeline:
            if clip.get("type") == "video_clip":
                start = clip.get("timeline_start", 0)
                end = clip.get("timeline_end", 0)
                if start <= frame_time < end:
                    active_clip = clip
                    break

        if active_clip:
            # Calculate source time
            source_time = active_clip["source_start"] + (frame_time - active_clip["timeline_start"])
            cap.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000)
            ret, frame = cap.read()

            if ret:
                # Resize to output dimensions
                frame = cv2.resize(frame, (width, height))
                return frame

        # Return black frame if no active clip
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _apply_color_grading(self, frame, color_grading: dict):
        """Apply color grading to frame.

        Args:
            frame: Input frame (BGR).
            color_grading: Color grading settings.

        Returns:
            Color graded frame.
        """
        import cv2
        import numpy as np

        if not color_grading:
            return frame

        preset = color_grading.get("preset_profile") or color_grading.get("preset")

        # Preset color grades
        if preset == "cinematic":
            # Desaturate slightly, boost contrast, add slight teal to shadows
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=-10)

        elif preset == "vintage":
            # Warm tones, reduced contrast, slight fade
            frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=20)
            b, g, r = cv2.split(frame)
            r = np.clip(r.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
            frame = cv2.merge([b, g, r])

        elif preset == "cyberpunk":
            # High contrast, boosted blues and magentas
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=-20)
            b, g, r = cv2.split(frame)
            b = np.clip(b.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            frame = cv2.merge([b, g, r])

        elif preset == "noir":
            # High contrast black and white
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=-30)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif preset == "warm":
            # Warm orange tones
            b, g, r = cv2.split(frame)
            r = np.clip(r.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
            frame = cv2.merge([b, g, r])

        elif preset == "cool":
            # Cool blue tones
            b, g, r = cv2.split(frame)
            b = np.clip(b.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
            frame = cv2.merge([b, g, r])

        # Apply manual adjustments
        saturation = color_grading.get("saturation", 1.0)
        contrast = color_grading.get("contrast", 1.0)
        brightness = color_grading.get("brightness", 1.0)

        if saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if contrast != 1.0 or brightness != 1.0:
            beta = (brightness - 1.0) * 50  # Convert to beta offset
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=beta)

        return frame

    def _apply_transitions(
        self,
        frame,
        transitions: list,
        frame_time: float,
        cap,
        timeline: list,
        width: int,
        height: int,
    ):
        """Apply transition effects.

        Args:
            frame: Current frame.
            transitions: List of transitions.
            frame_time: Current time.
            cap: Video capture.
            timeline: Timeline clips.
            width: Output width.
            height: Output height.

        Returns:
            Frame with transitions applied.
        """
        import cv2
        import numpy as np

        for trans in transitions:
            trans_start = trans.get("timeline_start", 0)
            trans_duration = trans.get("duration", 0.3)
            trans_end = trans_start + trans_duration

            if trans_start <= frame_time < trans_end:
                progress = (frame_time - trans_start) / trans_duration
                style = trans.get("style", "dissolve")

                # For dissolve, we need the next frame
                if style in ("dissolve", "fade", "crossfade"):
                    # Get next clip's frame
                    next_frame = self._get_next_clip_frame(
                        cap, timeline, trans_end + 0.01, width, height
                    )
                    if next_frame is not None:
                        # Blend frames
                        frame = cv2.addWeighted(
                            frame, 1 - progress,
                            next_frame, progress,
                            0
                        )

                elif style == "wipe_left":
                    next_frame = self._get_next_clip_frame(
                        cap, timeline, trans_end + 0.01, width, height
                    )
                    if next_frame is not None:
                        wipe_x = int(width * progress)
                        frame[:, :wipe_x] = next_frame[:, :wipe_x]

                elif style == "wipe_right":
                    next_frame = self._get_next_clip_frame(
                        cap, timeline, trans_end + 0.01, width, height
                    )
                    if next_frame is not None:
                        wipe_x = int(width * (1 - progress))
                        frame[:, wipe_x:] = next_frame[:, wipe_x:]

                elif style == "fade_black":
                    if progress < 0.5:
                        # Fade to black
                        alpha = 1 - (progress * 2)
                        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
                    else:
                        # Fade from black
                        next_frame = self._get_next_clip_frame(
                            cap, timeline, trans_end + 0.01, width, height
                        )
                        if next_frame is not None:
                            alpha = (progress - 0.5) * 2
                            frame = cv2.convertScaleAbs(next_frame, alpha=alpha, beta=0)

        return frame

    def _get_next_clip_frame(self, cap, timeline: list, time: float, width: int, height: int):
        """Get frame from the next clip after a given time."""
        import cv2
        import numpy as np

        for clip in timeline:
            if clip.get("type") == "video_clip":
                start = clip.get("timeline_start", 0)
                end = clip.get("timeline_end", 0)
                if start <= time < end:
                    source_time = clip["source_start"] + (time - start)
                    cap.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000)
                    ret, frame = cap.read()
                    if ret:
                        return cv2.resize(frame, (width, height))

        return None
