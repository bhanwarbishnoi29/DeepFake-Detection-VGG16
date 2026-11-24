"""
Video processor module orchestrating frame extraction, face detection,
and image normalization for Celeb-DF videos.
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.utils.logging_utils import setup_logger
from src.utils.config import load_config

logger = setup_logger("VideoProcessor")


class VideoProcessor:
    """
    High-level processor that extracts valid face crops from videos.
    """

    def __init__(
        self,
        frames_per_video: int = 10,
        target_size: Tuple[int, int] = (224, 224),
        face_method: str = "haar",
        margin_ratio: float = 0.15,
    ):
        """
        Args:
            frames_per_video: Number of frames to sample per video.
            target_size: Target dimension for cropped faces (width, height).
            face_method: Detector method ('haar').
            margin_ratio: Padding ratio around face bounding box.
        """
        self.frames_per_video = frames_per_video
        self.target_size = target_size
        self.frame_extractor = FrameExtractor(default_frames_per_video=frames_per_video)
        self.face_detector = FaceDetector(
            method=face_method,
            margin_ratio=margin_ratio,
            target_size=target_size,
        )

    def process_video(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Processes a single video: samples frames, detects faces, crops to 224x224 RGB.

        Args:
            video_path: Path to video file.
            num_frames: Optional override for sampled frames.

        Returns:
            Tuple of:
                - numpy array of shape (N_valid, 224, 224, 3) with uint8 RGB values.
                - metadata dict containing frames_analyzed, valid_faces, total_frames, fps.
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file does not exist: {video_path}")
            return np.empty((0, *self.target_size, 3), dtype=np.uint8), {
                "frames_analyzed": 0,
                "valid_faces": 0,
                "error": "File not found",
            }

        frames, meta = self.frame_extractor.extract_frames(
            video_path, num_frames=num_frames or self.frames_per_video
        )

        valid_faces = []
        for i, frame in enumerate(frames):
            try:
                face = self.face_detector.extract_face(frame, is_bgr=True)
                if face is not None:
                    valid_faces.append(face)
            except Exception as e:
                logger.warning(f"Error extracting face from frame {i} of {video_path}: {e}")

        meta_info = {
            "frames_analyzed": len(frames),
            "valid_faces": len(valid_faces),
            "total_frames": meta.get("total_frames", 0),
            "fps": meta.get("fps", 0.0),
            "duration_sec": meta.get("duration_sec", 0.0),
        }

        if valid_faces:
            faces_array = np.stack(valid_faces, axis=0).astype(np.uint8)
        else:
            faces_array = np.empty((0, *self.target_size, 3), dtype=np.uint8)

        return faces_array, meta_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sample videos for testing")
    parser.add_argument("--video", type=str, help="Path to single test video")
    parser.add_argument("--frames", type=int, default=10, help="Frames to sample")
    args = parser.parse_args()

    cfg = load_config()
    processor = VideoProcessor(
        frames_per_video=args.frames,
        target_size=tuple(cfg["preprocessing"]["image_size"]),
    )

    if args.video:
        faces, info = processor.process_video(args.video)
        logger.info(f"Processed {args.video}: {info['valid_faces']}/{info['frames_analyzed']} valid faces detected.")
    else:
        logger.info("VideoProcessor initialized successfully.")
