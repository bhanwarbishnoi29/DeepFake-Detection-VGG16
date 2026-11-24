"""
Standalone Video DeepFake Inference Engine.
Processes an uploaded or target video, detects faces, samples frames,
runs VGG16 model inference, aggregates frame probabilities, and outputs
structured prediction and confidence scores.
"""

import os
import sys

# Configure sandboxed cache directories before importing tensorflow or matplotlib
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
from typing import Any, Dict, Optional
import numpy as np
import tensorflow as tf

from src.utils.config import load_config, get_model_save_path
from src.utils.logging_utils import setup_logger
from src.preprocessing.video_processor import VideoProcessor
from src.models.model_utils import load_trained_model

logger = setup_logger("PredictInference")


def predict_video(
    video_path: str,
    model: Optional[tf.keras.Model] = None,
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    frames_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs full inference pipeline on a single video file.

    Args:
        video_path: Path to target MP4 video.
        model: Optional preloaded Keras model instance.
        model_path: Optional path to model weights file.
        config: Configuration dictionary.
        frames_override: Optional override for sampled frames.

    Returns:
        dict: Standardized inference result:
            {
                "prediction": "FAKE" | "REAL" | "UNCERTAIN",
                "confidence": 94.37,
                "frames_analyzed": 10,
                "valid_frames": 9,
                "average_probability": 0.9437,
                "frame_probabilities": [0.91, 0.95, ...]
            }
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {
                "preprocessing": {
                    "frames_per_video": 10,
                    "image_size": [224, 224],
                    "face_detection": {"method": "haar", "margin_ratio": 0.15},
                },
                "training": {
                    "model_dir": "models/trained_model",
                    "model_filename": "vgg16_deepfake_detector.keras",
                },
            }

    if not os.path.exists(video_path):
        return {
            "prediction": "ERROR",
            "confidence": 0.0,
            "frames_analyzed": 0,
            "valid_frames": 0,
            "average_probability": 0.0,
            "error": f"Video file not found at: {video_path}",
        }

    # Load model if not provided
    if model is None:
        target_model_path = model_path or get_model_save_path(config)
        if not os.path.exists(target_model_path):
            return {
                "prediction": "ERROR",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "valid_frames": 0,
                "average_probability": 0.0,
                "error": f"Trained model not found at {target_model_path}. Train the model first.",
            }
        model = load_trained_model(target_model_path)

    # Initialize video processor
    num_frames = frames_override or config["preprocessing"]["frames_per_video"]
    processor = VideoProcessor(
        frames_per_video=num_frames,
        target_size=tuple(config["preprocessing"]["image_size"]),
        face_method=config["preprocessing"]["face_detection"].get("method", "haar"),
        margin_ratio=config["preprocessing"]["face_detection"].get("margin_ratio", 0.15),
    )

    faces, meta = processor.process_video(video_path, num_frames=num_frames)

    frames_analyzed = meta.get("frames_analyzed", 0)
    valid_frames = meta.get("valid_faces", 0)

    if valid_frames == 0 or len(faces) == 0:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 50.0,
            "frames_analyzed": frames_analyzed,
            "valid_frames": 0,
            "average_probability": 0.5,
            "error": "No valid human faces detected across sampled video frames.",
        }

    # Preprocess faces for VGG16
    faces_preprocessed = tf.keras.applications.vgg16.preprocess_input(faces.astype(np.float32))

    # Predict probabilities for each face frame
    preds = model.predict(faces_preprocessed, verbose=0).flatten()
    frame_probs = [round(float(p), 4) for p in preds]

    # Arithmetic mean pooling
    avg_prob = float(np.mean(preds))

    if avg_prob >= 0.5:
        prediction = "FAKE"
        confidence = round(avg_prob * 100, 2)
    else:
        prediction = "REAL"
        confidence = round((1.0 - avg_prob) * 100, 2)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "frames_analyzed": frames_analyzed,
        "valid_frames": valid_frames,
        "average_probability": round(avg_prob, 4),
        "frame_probabilities": frame_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict DeepFake probability for an input video.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default=None, help="Optional model path")
    parser.add_argument("--frames", type=int, default=None, help="Sampled frames count")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    result = predict_video(
        video_path=args.video,
        model_path=args.model,
        frames_override=args.frames,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 45)
        print("  DEEPFAKE DETECTION RESULT")
        print("=" * 45)
        if "error" in result and result.get("prediction") == "ERROR":
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction:          {result['prediction']}")
            print(f"Confidence:          {result['confidence']:.2f}%")
            print(f"Frames Analyzed:     {result['frames_analyzed']}")
            print(f"Valid Face Frames:   {result['valid_frames']}")
            print(f"Average Probability: {result['average_probability']:.4f}")
            if "error" in result:
                print(f"Notice:              {result['error']}")
        print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
