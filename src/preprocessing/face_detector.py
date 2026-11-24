"""
Face Detection and extraction module for DeepFake Detection.
Supports modular detection algorithms, primary face extraction,
bounding box margin padding, and graceful handling of missing or corrupted faces.
"""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np


class FaceDetector:
    """
    Modular Face Detector supporting OpenCV Haar Cascades and expandable to DNNs.
    """

    def __init__(
        self,
        method: str = "haar",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (40, 40),
        margin_ratio: float = 0.15,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initializes the face detector.

        Args:
            method: Face detection method ('haar').
            scale_factor: Scale factor for multiscale detection.
            min_neighbors: Minimum neighbor rectangles for candidate retention.
            min_face_size: Minimum face bounding box size (width, height).
            margin_ratio: Percentage margin to expand around detected face.
            target_size: Output dimensions (width, height) for cropped face (default 224x224).
        """
        self.method = method.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.margin_ratio = margin_ratio
        self.target_size = target_size

        if self.method == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unsupported face detection method: {self.method}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects all candidate faces in a given BGR or RGB frame.

        Args:
            frame: Numpy array representing the image frame.

        Returns:
            List of bounding boxes [(x, y, w, h), ...].
        """
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale for Haar detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return [tuple(box) for box in faces]

    def select_primary_face(
        self, faces: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Selects the primary/main face from multiple detections.
        Prioritizes the largest bounding box area, with tie-break closest to image center.

        Args:
            faces: List of bounding boxes [(x, y, w, h)].
            frame_shape: (height, width, channels) of the frame.

        Returns:
            The primary bounding box (x, y, w, h) or None if no faces found.
        """
        if not faces:
            return None

        h_frame, w_frame = frame_shape[:2]
        center_x, center_y = w_frame / 2.0, h_frame / 2.0

        def score_face(box: Tuple[int, int, int, int]) -> float:
            x, y, w, h = box
            area = w * h
            box_cx = x + w / 2.0
            box_cy = y + h / 2.0
            # Distance from center normalized by frame diagonal
            dist = np.hypot(box_cx - center_x, box_cy - center_y)
            # Area score minus center distance penalty
            return area - (dist * 10.0)

        best_face = max(faces, key=score_face)
        return best_face

    def crop_and_resize(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        is_bgr: bool = True,
    ) -> np.ndarray:
        """
        Crops face with margin expansion, converts color from BGR to RGB,
        and resizes to target_size (224x224).

        Args:
            frame: Source image frame.
            box: (x, y, w, h) bounding box.
            is_bgr: True if input frame is BGR (default for cv2.VideoCapture).

        Returns:
            RGB numpy array of shape (target_size[1], target_size[0], 3).
        """
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = box

        # Add margin
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_frame, x + w + margin_x)
        y2 = min(h_frame, y + h + margin_y)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            # Fallback if crop is invalid
            cropped = frame

        # Convert to RGB if input is BGR
        if is_bgr and len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Resize to target 224x224
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    def extract_face(
        self, frame: np.ndarray, is_bgr: bool = True
    ) -> Optional[np.ndarray]:
        """
        End-to-end extraction: detects faces, selects primary face, crops, and resizes.

        Args:
            frame: Source image frame.
            is_bgr: True if input frame is in BGR format.

        Returns:
            RGB face numpy array (224, 224, 3) or None if no face detected.
        """
        if frame is None or frame.size == 0:
            return None

        faces = self.detect_faces(frame)
        if not faces:
            return None

        primary_box = self.select_primary_face(faces, frame.shape)
        if primary_box is None:
            return None

        return self.crop_and_resize(frame, primary_box, is_bgr=is_bgr)
