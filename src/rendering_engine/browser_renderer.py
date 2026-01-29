"""Browser-based video rendering using Playwright."""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

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

        # Inject timeline data
        await self._page.evaluate(f"""
            window.timelineData = {json.dumps(timeline)};
            window.sourceVideo = "{source_video}";
        """)

        # Wait for initialization
        await self._page.wait_for_function("window.rendererReady === true", timeout=30000)

        # Calculate total frames
        total_duration = timeline.get("metadata", {}).get("total_duration", 60)
        total_frames = int(total_duration * self.fps)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Rendering {total_frames} frames...")

        # Render each frame
        for frame_idx in range(total_frames):
            frame_time = frame_idx / self.fps

            # Call render function in browser
            await self._page.evaluate(f"window.renderFrame({frame_idx}, {frame_time})")

            # Capture screenshot
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            await self._page.screenshot(path=frame_path)

            if frame_idx % 100 == 0:
                logger.info(f"Rendered frame {frame_idx}/{total_frames}")

        logger.info(f"Rendered {total_frames} frames to {output_dir}")
        return output_dir

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
