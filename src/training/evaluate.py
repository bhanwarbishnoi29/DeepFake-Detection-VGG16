"""
Model evaluation module for Celeb-DF DeepFake Detection.
Performs rigorous Video-Level and Frame-Level evaluation using official test split.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and generates Confusion Matrix plots.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import load_config, get_model_save_path
from src.utils.logging_utils import setup_logger
from src.dataset.dataset_loader import load_splits, LABEL_REAL, LABEL_FAKE
from src.preprocessing.video_processor import VideoProcessor
from src.models.model_utils import load_trained_model

logger = setup_logger("Evaluation")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/plots/confusion_matrix.png",
    title: str = "Video-Level Confusion Matrix",
):
    """Generates and saves a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel("Ground Truth Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {output_path}")


def evaluate_model(
    config_path: str = "config/config.yaml",
    model_path: Optional[str] = None,
    max_test_videos: Optional[int] = None,
) -> Dict[str, any]:
    """
    Evaluates model on the test video dataset.

    Args:
        config_path: Path to configuration file.
        model_path: Path to trained model (defaults to config path).
        max_test_videos: Optional cap for quick evaluation testing.

    Returns:
        Dict containing comprehensive video-level and frame-level metrics.
    """
    config = load_config(config_path)
    model_path = model_path or get_model_save_path(config)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train model first.")

    model = load_trained_model(model_path)
    _, _, test_df = load_splits(config["dataset"]["splits_dir"])

    if max_test_videos:
        # Sample proportionally across classes to ensure both REAL and FAKE are evaluated
        half = max(1, max_test_videos // 2)
        real_subset = test_df[test_df["label"] == LABEL_REAL].head(half)
        fake_subset = test_df[test_df["label"] == LABEL_FAKE].head(max_test_videos - len(real_subset))
        test_df = pd.concat([real_subset, fake_subset]).reset_index(drop=True)
        logger.info(f"Capping test evaluation to {len(test_df)} videos ({len(real_subset)} REAL, {len(fake_subset)} FAKE).")

    processor = VideoProcessor(
        frames_per_video=config["preprocessing"]["frames_per_video"],
        target_size=tuple(config["preprocessing"]["image_size"]),
        face_method=config["preprocessing"]["face_detection"]["method"],
        margin_ratio=config["preprocessing"]["face_detection"]["margin_ratio"],
    )

    threshold = config["evaluation"].get("threshold", 0.5)

    video_true_labels = []
    video_pred_probs = []
    video_pred_labels = []
    video_results = []

    frame_true_labels = []
    frame_pred_probs = []

    logger.info(f"Beginning evaluation on {len(test_df)} test videos...")

    for idx, row in test_df.iterrows():
        video_path = row["full_path"]
        true_label = int(row["label"])

        faces, meta = processor.process_video(video_path)

        if len(faces) == 0:
            logger.warning(f"No face detected in test video: {row['relative_path']}. Defaulting to unconfident score.")
            avg_prob = 0.5
            pred_label = 1 if avg_prob >= threshold else 0
        else:
            # Preprocess for VGG16
            faces_preprocessed = tf.keras.applications.vgg16.preprocess_input(faces.astype(np.float32))
            preds = model.predict(faces_preprocessed, verbose=0).flatten()

            avg_prob = float(np.mean(preds))
            pred_label = 1 if avg_prob >= threshold else 0

            # Store frame-level predictions
            for p in preds:
                frame_true_labels.append(true_label)
                frame_pred_probs.append(float(p))

        video_true_labels.append(true_label)
        video_pred_probs.append(avg_prob)
        video_pred_labels.append(pred_label)

        video_results.append({
            "video": row["relative_path"],
            "true_label": "FAKE" if true_label == 1 else "REAL",
            "pred_label": "FAKE" if pred_label == 1 else "REAL",
            "avg_probability": round(avg_prob, 4),
            "valid_faces": meta.get("valid_faces", 0),
            "frames_analyzed": meta.get("frames_analyzed", 0),
        })

    # Video-level metrics calculation
    y_true_v = np.array(video_true_labels)
    y_pred_v = np.array(video_pred_labels)
    y_prob_v = np.array(video_pred_probs)

    acc_v = accuracy_score(y_true_v, y_pred_v)
    prec_v = precision_score(y_true_v, y_pred_v, zero_division=0)
    rec_v = recall_score(y_true_v, y_pred_v, zero_division=0)
    f1_v = f1_score(y_true_v, y_pred_v, zero_division=0)

    try:
        auc_v = roc_auc_score(y_true_v, y_prob_v)
    except Exception:
        auc_v = float("nan")

    cm_v = confusion_matrix(y_true_v, y_pred_v, labels=[LABEL_REAL, LABEL_FAKE])

    # Frame-level metrics
    if frame_true_labels:
        y_true_f = np.array(frame_true_labels)
        y_prob_f = np.array(frame_pred_probs)
        y_pred_f = (y_prob_f >= threshold).astype(int)

        acc_f = accuracy_score(y_true_f, y_pred_f)
        prec_f = precision_score(y_true_f, y_pred_f, zero_division=0)
        rec_f = recall_score(y_true_f, y_pred_f, zero_division=0)
        f1_f = f1_score(y_true_f, y_pred_f, zero_division=0)
        try:
            auc_f = roc_auc_score(y_true_f, y_prob_f)
        except Exception:
            auc_f = float("nan")
    else:
        acc_f, prec_f, rec_f, f1_f, auc_f = 0.0, 0.0, 0.0, 0.0, 0.0

    metrics_summary = {
        "video_level": {
            "total_videos_evaluated": len(video_true_labels),
            "accuracy": float(acc_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "f1_score": float(f1_v),
            "roc_auc": float(auc_v) if not np.isnan(auc_v) else None,
            "confusion_matrix": {
                "true_real_pred_real (TN)": int(cm_v[0, 0]),
                "true_real_pred_fake (FP)": int(cm_v[0, 1]),
                "true_fake_pred_real (FN)": int(cm_v[1, 0]),
                "true_fake_pred_fake (TP)": int(cm_v[1, 1]),
            },
        },
        "frame_level": {
            "total_frames_evaluated": len(frame_true_labels),
            "accuracy": float(acc_f),
            "precision": float(prec_f),
            "recall": float(rec_f),
            "f1_score": float(f1_f),
            "roc_auc": float(auc_f) if not np.isnan(auc_f) else None,
        },
        "video_predictions_sample": video_results[:15],
    }

    # Save reports
    reports_dir = config["evaluation"]["reports_dir"]
    plots_dir = config["evaluation"]["plots_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot confusion matrix
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm_v, ["REAL", "FAKE"], output_path=cm_plot_path)

    auc_display = f"{auc_v:.4f}" if (auc_v is not None and not np.isnan(auc_v)) else "N/A"
    logger.info(
        f"\n========================================\n"
        f"VIDEO-LEVEL EVALUATION RESULTS:\n"
        f"  Total Videos: {len(video_true_labels)}\n"
        f"  Accuracy:     {acc_v * 100:.2f}%\n"
        f"  Precision:    {prec_v * 100:.2f}%\n"
        f"  Recall:       {rec_v * 100:.2f}%\n"
        f"  F1 Score:     {f1_v * 100:.2f}%\n"
        f"  ROC-AUC:      {auc_display}\n"
        f"========================================"
    )

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepFake Detection Model on Test Set")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    evaluate_model(config_path=args.config, model_path=args.model, max_test_videos=args.max_videos)
