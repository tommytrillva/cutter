"""Motion detection for shot quality scoring."""
import logging
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detect motion between frames using optical flow."""

    def __init__(
        self,
        flow_method: str = "farneback",
        scale_factor: float = 0.5,
    ):
        """Initialize motion detector.

        Args:
            flow_method: Method for optical flow ('farneback' or 'lucas_kanade').
            scale_factor: Scale factor for processing (lower = faster, less accurate).
        """
        self.flow_method = flow_method
        self.scale_factor = scale_factor

        # Farneback parameters
        self.farneback_params = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }

    def calculate_motion(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
    ) -> float:
        """Calculate motion score between two frames.

        Args:
            frame: Current frame (RGB or BGR).
            prev_frame: Previous frame (RGB or BGR).

        Returns:
            Motion score between 0 and 1.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_frame

        # Resize for faster processing
        if self.scale_factor < 1.0:
            height, width = gray.shape
            new_size = (
                int(width * self.scale_factor),
                int(height * self.scale_factor),
            )
            gray = cv2.resize(gray, new_size)
            prev_gray = cv2.resize(prev_gray, new_size)

        # Calculate optical flow
        if self.flow_method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                **self.farneback_params,
            )
        else:
            # Lucas-Kanade method (sparse)
            return self._lucas_kanade_motion(prev_gray, gray)

        # Calculate magnitude of flow
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate average motion magnitude
        avg_magnitude = np.mean(magnitude)

        # Normalize to 0-1 (using empirical max of ~20 pixels)
        motion_score = min(1.0, avg_magnitude / 20.0)

        return float(motion_score)

    def _lucas_kanade_motion(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
    ) -> float:
        """Calculate motion using Lucas-Kanade sparse optical flow.

        Args:
            prev_gray: Previous frame grayscale.
            gray: Current frame grayscale.

        Returns:
            Motion score 0-1.
        """
        # Detect features in previous frame
        feature_params = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 7,
            "blockSize": 7,
        }

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        if p0 is None or len(p0) < 10:
            return 0.0

        # Calculate optical flow
        lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, **lk_params
        )

        if p1 is None:
            return 0.0

        # Select good points
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        if len(good_old) < 5:
            return 0.0

        # Calculate average displacement
        displacements = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
        avg_displacement = np.mean(displacements)

        # Normalize to 0-1
        motion_score = min(1.0, avg_displacement / 50.0)

        return float(motion_score)

    def detect_scene_change(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """Detect if there's a scene change between frames.

        Args:
            frame: Current frame.
            prev_frame: Previous frame.
            threshold: Histogram difference threshold for scene change.

        Returns:
            Tuple of (is_scene_change, difference_score).
        """
        # Convert to HSV for histogram comparison
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
        else:
            # Grayscale
            hsv = frame
            prev_hsv = prev_frame

        # Calculate histograms
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges
        channels = [0, 1]

        if len(hsv.shape) == 3:
            hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
            prev_hist = cv2.calcHist([prev_hsv], channels, None, hist_size, ranges)
        else:
            hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
            prev_hist = cv2.calcHist([prev_hsv], [0], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(hist, hist)
        cv2.normalize(prev_hist, prev_hist)

        # Compare histograms
        correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)

        # Convert correlation (1 = identical, 0 = different) to difference
        difference = 1.0 - correlation

        is_scene_change = difference > threshold

        return is_scene_change, float(difference)

    def calculate_motion_direction(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate predominant motion direction.

        Args:
            frame: Current frame.
            prev_frame: Previous frame.

        Returns:
            Tuple of (horizontal_motion, vertical_motion) normalized -1 to 1.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            prev_gray = prev_frame

        # Resize for speed
        if self.scale_factor < 1.0:
            height, width = gray.shape
            new_size = (
                int(width * self.scale_factor),
                int(height * self.scale_factor),
            )
            gray = cv2.resize(gray, new_size)
            prev_gray = cv2.resize(prev_gray, new_size)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            **self.farneback_params,
        )

        # Get average flow direction
        avg_flow_x = np.mean(flow[..., 0])
        avg_flow_y = np.mean(flow[..., 1])

        # Normalize to -1 to 1 (using empirical max of 10 pixels)
        horizontal = np.clip(avg_flow_x / 10.0, -1.0, 1.0)
        vertical = np.clip(avg_flow_y / 10.0, -1.0, 1.0)

        return float(horizontal), float(vertical)

    def get_motion_regions(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        threshold: float = 0.3,
    ) -> np.ndarray:
        """Get binary mask of high-motion regions.

        Args:
            frame: Current frame.
            prev_frame: Previous frame.
            threshold: Motion threshold for masking.

        Returns:
            Binary mask of motion regions.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            prev_gray = prev_frame

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            **self.farneback_params,
        )

        # Calculate magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize
        magnitude_norm = magnitude / (magnitude.max() + 1e-6)

        # Create binary mask
        motion_mask = (magnitude_norm > threshold).astype(np.uint8) * 255

        return motion_mask
