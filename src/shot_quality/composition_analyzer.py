"""Composition analysis for shot quality scoring."""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class CompositionAnalyzer:
    """Analyze frame composition for visual interest."""

    def __init__(self):
        """Initialize composition analyzer."""
        pass

    def analyze_composition(self, frame: np.ndarray) -> float:
        """Analyze overall composition quality.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Composition score between 0 and 1.
        """
        # Multiple composition factors
        thirds_score = self._rule_of_thirds_score(frame)
        balance_score = self._visual_balance_score(frame)
        edge_score = self._edge_interest_score(frame)

        # Weighted combination
        score = (
            thirds_score * 0.4 +
            balance_score * 0.35 +
            edge_score * 0.25
        )

        return float(score)

    def _rule_of_thirds_score(self, frame: np.ndarray) -> float:
        """Score based on rule of thirds alignment.

        High interest regions should align with thirds lines.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Score between 0 and 1.
        """
        h, w = frame.shape[:2]

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Detect edges (areas of interest)
        edges = cv2.Canny(gray, 50, 150)

        # Define thirds lines (and intersection points)
        third_x = [w // 3, 2 * w // 3]
        third_y = [h // 3, 2 * h // 3]

        # Create weight map emphasizing thirds lines
        weight_map = np.zeros_like(gray, dtype=np.float32)

        # Vertical thirds lines
        line_width = max(5, w // 20)
        for x in third_x:
            weight_map[:, max(0, x - line_width):min(w, x + line_width)] += 0.5

        # Horizontal thirds lines
        for y in third_y:
            weight_map[max(0, y - line_width):min(h, y + line_width), :] += 0.5

        # Intersection points (power points) get extra weight
        for x in third_x:
            for y in third_y:
                cv2.circle(weight_map, (x, y), line_width * 2, 1.0, -1)

        # Normalize weight map
        weight_map = weight_map / weight_map.max()

        # Calculate weighted edge score
        edge_norm = edges.astype(np.float32) / 255.0
        weighted_score = np.sum(edge_norm * weight_map) / (np.sum(weight_map) + 1e-6)

        return float(np.clip(weighted_score * 3, 0, 1))

    def _visual_balance_score(self, frame: np.ndarray) -> float:
        """Score based on visual weight balance.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Score between 0 and 1 (1 = well balanced).
        """
        h, w = frame.shape[:2]

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Calculate visual weight (using brightness as proxy)
        # Invert so darker objects have more "weight"
        weight = 255 - gray

        # Calculate center of mass
        total_weight = np.sum(weight)
        if total_weight == 0:
            return 0.5  # Neutral score for empty frames

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_x = np.sum(x_coords * weight) / total_weight
        center_y = np.sum(y_coords * weight) / total_weight

        # Ideal center is frame center
        ideal_x, ideal_y = w / 2, h / 2

        # Calculate normalized distance from ideal
        distance_x = abs(center_x - ideal_x) / (w / 2)
        distance_y = abs(center_y - ideal_y) / (h / 2)

        distance = np.sqrt(distance_x ** 2 + distance_y ** 2)

        # Convert distance to score (closer to center = higher score)
        # But also allow for intentional off-center composition
        # Score highest at center OR at thirds positions
        thirds_distance_x = min(
            abs(center_x - w / 3) / (w / 3),
            abs(center_x - 2 * w / 3) / (w / 3)
        )
        thirds_distance_y = min(
            abs(center_y - h / 3) / (h / 3),
            abs(center_y - 2 * h / 3) / (h / 3)
        )

        thirds_distance = np.sqrt(thirds_distance_x ** 2 + thirds_distance_y ** 2)

        # Best score if centered OR on thirds
        center_score = 1.0 - np.clip(distance, 0, 1)
        thirds_score = 1.0 - np.clip(thirds_distance, 0, 1)

        score = max(center_score, thirds_score * 0.9)

        return float(score)

    def _edge_interest_score(self, frame: np.ndarray) -> float:
        """Score based on edge detection (visual complexity).

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Score between 0 and 1.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size

        # Optimal edge density is moderate (not too sparse, not too busy)
        # Peak score at ~10% edge density
        optimal_density = 0.10
        score = 1.0 - abs(edge_density - optimal_density) / optimal_density

        return float(np.clip(score, 0, 1))

    def analyze_contrast(self, frame: np.ndarray) -> float:
        """Analyze contrast quality.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Contrast score between 0 and 1.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Calculate contrast using standard deviation
        std = np.std(gray)

        # Normalize (typical std range is 20-80)
        score = np.clip((std - 20) / 60, 0, 1)

        return float(score)

    def analyze_lighting(self, frame: np.ndarray) -> float:
        """Analyze lighting quality.

        Checks for over/under exposure and lighting balance.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Lighting score between 0 and 1.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Check for clipping (over/under exposure)
        underexposed = np.sum(hist[:20])  # Dark pixels
        overexposed = np.sum(hist[235:])  # Bright pixels
        clipping_penalty = (underexposed + overexposed) * 2

        # Check for mean brightness (ideal is ~120-140)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 130) / 130

        # Check histogram spread (should use full range)
        percentile_5 = np.percentile(gray, 5)
        percentile_95 = np.percentile(gray, 95)
        range_score = (percentile_95 - percentile_5) / 255

        # Combine scores
        score = (
            brightness_score * 0.3 +
            range_score * 0.4 +
            (1.0 - clipping_penalty) * 0.3
        )

        return float(np.clip(score, 0, 1))

    def get_color_analysis(
        self, frame: np.ndarray
    ) -> Dict[str, Any]:
        """Get detailed color analysis.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Dictionary with color metrics.
        """
        # Convert to HSV
        if len(frame.shape) != 3:
            return {
                "saturation": 0.0,
                "dominant_hue": 0,
                "color_variety": 0.0,
            }

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Mean saturation
        saturation = np.mean(hsv[:, :, 1]) / 255

        # Dominant hue
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hue_hist))

        # Color variety (using saturation histogram spread)
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        sat_hist = sat_hist.flatten() / sat_hist.sum()
        color_variety = float(np.std(sat_hist) * 100)

        return {
            "saturation": float(saturation),
            "dominant_hue": dominant_hue,
            "color_variety": color_variety,
        }
