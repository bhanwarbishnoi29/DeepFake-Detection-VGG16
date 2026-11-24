"""
Frame extraction module for DeepFake Detection.
Handles robust video loading, sampling strategies, and frame retrieval via OpenCV.
"""

import os
from typing import Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np


class FrameExtractor:
    """
    Extracts evenly spaced frames from video files with error handling.
    """

    def __init__(self, default_frames_per_video: int = 10):
        """
        Args:
            default_frames_per_video: Number of frames to sample from each video.
        """
        self.default_frames_per_video = default_frames_per_video

    def get_video_metadata(self, video_path: str) -> Optional[Dict[str, float]]:
        """
        Extracts metadata from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict containing total_frames, fps, duration_sec, width, height.
        """
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / fps) if (fps and fps > 0) else 0.0

        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps if fps else 0.0,
            "duration_sec": duration,
            "width": width,
            "height": height,
        }

    def compute_sample_indices(
        self, total_frames: int, num_frames: int, margin_frames: int = 2
    ) -> List[int]:
        """
        Computes evenly spaced frame indices, avoiding boundary artifacts.

        Args:
            total_frames: Total number of frames in the video.
            num_frames: Desired number of sampled frames.
            margin_frames: Frames to skip at beginning and end.

        Returns:
            List of integer frame indices.
        """
        if total_frames <= 0 or num_frames <= 0:
            return []

        if total_frames <= num_frames:
            return list(range(total_frames))

        start = min(margin_frames, max(0, total_frames - 1))
        end = max(start, total_frames - margin_frames - 1)

        if start >= end:
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        else:
            indices = np.linspace(start, end, num=num_frames, dtype=int)

        return sorted(list(set(indices.tolist())))

    def extract_frames(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Extracts sampled frames from a video file.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample (defaults to self.default_frames_per_video).

        Returns:
            Tuple of (list of BGR frames, metadata dict).
        """
        if num_frames is None:
            num_frames = self.default_frames_per_video

        metadata = self.get_video_metadata(video_path)
        if metadata is None or metadata["total_frames"] == 0:
            return [], {}

        total_frames = int(metadata["total_frames"])
        target_indices = set(self.compute_sample_indices(total_frames, num_frames))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], metadata

        frames = []
        current_idx = 0

        while cap.isOpened() and len(frames) < num_frames and current_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if current_idx in target_indices:
                frames.append(frame)

            current_idx += 1

        cap.release()
        return frames, metadata
