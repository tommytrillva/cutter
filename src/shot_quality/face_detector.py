"""Face and subject detection for shot quality scoring."""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Using OpenCV cascade for face detection.")


class FaceDetector:
    """Detect faces and subjects in frames."""

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_detection_confidence: float = 0.5,
    ):
        """Initialize face detector.

        Args:
            use_mediapipe: Use MediaPipe if available (more accurate).
            min_detection_confidence: Minimum confidence for detection.
        """
        self.min_detection_confidence = min_detection_confidence
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE

        if self.use_mediapipe:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close range, 1 for full range
                min_detection_confidence=min_detection_confidence,
            )
            logger.info("Using MediaPipe face detection")
        else:
            # Fallback to OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Using OpenCV Haar cascade face detection")

    def detect_faces(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect faces in a frame.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            List of detected faces with bounding boxes and confidence.
        """
        if self.use_mediapipe:
            return self._detect_faces_mediapipe(frame)
        else:
            return self._detect_faces_opencv(frame)

    def _detect_faces_mediapipe(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            List of face detection results.
        """
        results = self.face_detector.process(frame)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]

                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure bounds are within frame
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                faces.append({
                    "bbox": (x, y, width, height),
                    "confidence": float(confidence),
                    "center": (x + width // 2, y + height // 2),
                    "area": width * height,
                    "relative_area": (width * height) / (w * h),
                })

        return faces

    def _detect_faces_opencv(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar cascade.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            List of face detection results.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = frame.shape[:2]

        # Detect faces
        detections = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        faces = []
        for (x, y, width, height) in detections:
            # Estimate confidence (Haar cascade doesn't provide this)
            confidence = 0.7  # Default confidence

            faces.append({
                "bbox": (x, y, width, height),
                "confidence": confidence,
                "center": (x + width // 2, y + height // 2),
                "area": width * height,
                "relative_area": (width * height) / (w * h),
            })

        return faces

    def calculate_face_score(
        self, frame: np.ndarray
    ) -> float:
        """Calculate face presence score for a frame.

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Score between 0 and 1 based on face presence and prominence.
        """
        faces = self.detect_faces(frame)

        if not faces:
            return 0.0

        # Score based on:
        # 1. Number of faces (diminishing returns)
        # 2. Total face area
        # 3. Face confidence

        num_faces_score = min(1.0, len(faces) / 3.0)  # Max score at 3 faces

        total_area = sum(f["relative_area"] for f in faces)
        area_score = min(1.0, total_area * 5)  # Max at 20% frame coverage

        avg_confidence = np.mean([f["confidence"] for f in faces])

        # Weighted combination
        score = (
            num_faces_score * 0.3 +
            area_score * 0.4 +
            avg_confidence * 0.3
        )

        return float(score)

    def get_main_subject_position(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Get position of main subject (largest face).

        Args:
            frame: Frame as RGB numpy array.

        Returns:
            Normalized (x, y) position of main subject or None.
        """
        faces = self.detect_faces(frame)

        if not faces:
            return None

        # Find largest face
        largest = max(faces, key=lambda f: f["area"])

        h, w = frame.shape[:2]
        x_norm = largest["center"][0] / w
        y_norm = largest["center"][1] / h

        return (x_norm, y_norm)

    def is_face_centered(
        self,
        frame: np.ndarray,
        tolerance: float = 0.2,
    ) -> Tuple[bool, Optional[float]]:
        """Check if main face is approximately centered.

        Args:
            frame: Frame as RGB numpy array.
            tolerance: How far from center is acceptable (0-0.5).

        Returns:
            Tuple of (is_centered, distance_from_center).
        """
        position = self.get_main_subject_position(frame)

        if position is None:
            return False, None

        # Calculate distance from center (0.5, 0.5)
        distance = np.sqrt((position[0] - 0.5) ** 2 + (position[1] - 0.5) ** 2)

        is_centered = distance <= tolerance

        return is_centered, float(distance)

    def close(self):
        """Release resources."""
        if self.use_mediapipe and hasattr(self, 'face_detector'):
            self.face_detector.close()
