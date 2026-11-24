"""
Training pipeline for VGG16 Transfer Learning DeepFake Detection.
Supports incremental batch loading, caching, class weighting, callbacks,
fine-tuning, and metric/history visualization.
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

import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_model_save_path
from src.utils.logging_utils import setup_logger
from src.dataset.dataset_loader import load_splits, calculate_class_weights, LABEL_REAL, LABEL_FAKE
from src.preprocessing.video_processor import VideoProcessor
from src.models.vgg16_model import build_vgg16_classifier, setup_fine_tuning
from src.models.model_utils import save_trained_model, load_trained_model

logger = setup_logger("TrainingPipeline", log_file="outputs/reports/training_run.log")


class EpochProgressCallback(Callback):
    """
    Saves intermediate epoch metrics and status to disk after every epoch
    to guarantee safe resumption and real-time monitoring.
    """

    def __init__(self, progress_file: str = "outputs/reports/training_progress.json", phase: str = "transfer_learning"):
        super().__init__()
        self.progress_file = progress_file
        self.phase = phase
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - (self.start_time or time.time())
        os.makedirs(os.path.dirname(os.path.abspath(self.progress_file)), exist_ok=True)

        data = {}
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if self.phase not in data:
            data[self.phase] = []

        entry = {
            "epoch": epoch + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            **{k: round(float(v), 5) for k, v in logs.items()},
        }
        data[self.phase].append(entry)
        data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["last_phase"] = self.phase
        data["total_elapsed_seconds"] = round(elapsed, 2)

        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Progress recorded to {self.progress_file} for epoch {epoch + 1}.")


class VideoFaceDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator that streams face crops from video files.
    Optionally caches extracted faces to disk to prevent redundant computation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_processor: VideoProcessor,
        cache_dir: Optional[str] = "data/processed/faces",
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.video_processor = video_processor
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.df))

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.df) / self.batch_size)))

    def _get_cache_path(self, rel_path: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe_name = rel_path.replace(os.sep, "__").replace(".mp4", ".npz")
        return os.path.join(self.cache_dir, safe_name)

    def _load_video_faces(self, full_path: str, rel_path: str) -> np.ndarray:
        cache_path = self._get_cache_path(rel_path)
        if cache_path and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data["faces"]
            except Exception as e:
                logger.warning(f"Error reading cache {cache_path}: {e}")

        # Extract using video processor
        faces, _ = self.video_processor.process_video(full_path)

        # Cache if valid
        if cache_path and faces is not None and len(faces) > 0:
            try:
                np.savez_compressed(cache_path, faces=faces)
            except Exception as e:
                logger.warning(f"Error saving cache {cache_path}: {e}")

        return faces

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        batch_faces = []
        batch_labels = []

        for _, row in batch_df.iterrows():
            faces = self._load_video_faces(row["full_path"], row["relative_path"])
            label = float(row["label"])

            if faces is not None and len(faces) > 0:
                for face in faces:
                    batch_faces.append(face)
                    batch_labels.append(label)

        # If no faces were extracted in entire batch (rare edge case), return a dummy zero-tensor
        if len(batch_faces) == 0:
            dummy_face = np.zeros((self.video_processor.target_size[0], self.video_processor.target_size[1], 3), dtype=np.float32)
            return np.expand_dims(dummy_face, 0), np.array([0.0], dtype=np.float32)

        x_batch = np.array(batch_faces, dtype=np.float32)
        y_batch = np.array(batch_labels, dtype=np.float32)

        # Apply VGG16 standard preprocessing (zero-centering ImageNet mean)
        x_batch = tf.keras.applications.vgg16.preprocess_input(x_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def plot_and_save_history(
    history_dict: Dict[str, List[float]],
    output_path: str = "outputs/plots/training_history.png",
):
    """Plots and saves loss and accuracy curves."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history_dict.get("loss", []), label="Train Loss", color="royalblue", lw=2)
    if "val_loss" in history_dict:
        axes[0].plot(history_dict["val_loss"], label="Val Loss", color="darkorange", lw=2, linestyle="--")
    axes[0].set_title("VGG16 Transfer Learning — Binary Crossentropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    axes[1].plot(history_dict.get("accuracy", []), label="Train Accuracy", color="royalblue", lw=2)
    if "val_accuracy" in history_dict:
        axes[1].plot(history_dict["val_accuracy"], label="Val Accuracy", color="darkorange", lw=2, linestyle="--")
    axes[1].set_title("VGG16 Transfer Learning — Frame Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Training history curves saved to: {output_path}")


def train_pipeline(
    config_path: str = "config/config.yaml",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    epochs_override: Optional[int] = None,
    resume: bool = False,
) -> Tuple[tf.keras.Model, Dict[str, any]]:
    """
    Executes the training and optional fine-tuning pipeline.

    Args:
        config_path: Path to configuration file.
        max_train_samples: Optional cap on training videos (for testing/smoke tests).
        max_val_samples: Optional cap on validation videos.
        epochs_override: Optional override for epochs.
        resume: Whether to resume training from an existing checkpoint if available.

    Returns:
        Tuple of (trained_model, history_metrics).
    """
    config = load_config(config_path)
    logger.info("Initializing training pipeline...")

    # Load splits
    train_df, val_df, _ = load_splits(config["dataset"]["splits_dir"])

    if max_train_samples:
        train_df = train_df.head(max_train_samples)
        logger.info(f"Capping training set to {len(train_df)} videos for testing.")

    if max_val_samples:
        val_df = val_df.head(max_val_samples)
        logger.info(f"Capping validation set to {len(val_df)} videos for testing.")

    class_weights = calculate_class_weights(train_df)

    processor = VideoProcessor(
        frames_per_video=config["preprocessing"]["frames_per_video"],
        target_size=tuple(config["preprocessing"]["image_size"]),
        face_method=config["preprocessing"]["face_detection"]["method"],
        margin_ratio=config["preprocessing"]["face_detection"]["margin_ratio"],
    )

    batch_size = config["training"]["batch_size"]
    train_gen = VideoFaceDataGenerator(
        train_df,
        processor,
        cache_dir=os.path.join(config["dataset"]["processed_dir"], "faces"),
        batch_size=batch_size,
        shuffle=True,
    )
    val_gen = VideoFaceDataGenerator(
        val_df,
        processor,
        cache_dir=os.path.join(config["dataset"]["processed_dir"], "faces"),
        batch_size=batch_size,
        shuffle=False,
    )

    checkpoint_path = os.path.join(
        config["training"]["model_dir"], "vgg16_full_training_checkpoint.keras"
    )
    model_save_path = get_model_save_path(config)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Build or resume model
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from existing checkpoint: {checkpoint_path}")
        model = load_trained_model(checkpoint_path)
    else:
        model = build_vgg16_classifier(
            input_shape=tuple(config["model"]["input_shape"]),
            weights=config["model"]["weights"],
            dense_units=config["model"]["dense_units"],
            dropout_rate=config["model"]["dropout_rate"],
            learning_rate=config["model"]["learning_rate"],
            freeze_base=True,
        )

    epochs = epochs_override or config["training"]["epochs"]
    progress_file = os.path.join(config["evaluation"]["reports_dir"], "training_progress.json")

    tl_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["training"]["reduce_lr_factor"],
            patience=config["training"]["reduce_lr_patience"],
            min_lr=config["training"]["min_lr"],
            verbose=1,
        ),
        EpochProgressCallback(progress_file=progress_file, phase="transfer_learning"),
    ]

    logger.info(f"Starting initial Transfer Learning phase for {epochs} epochs on {len(train_df)} training and {len(val_df)} validation videos...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=tl_callbacks,
        verbose=1,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Optional fine-tuning phase
    if config["model"]["fine_tune"]["enabled"]:
        fine_tune_epochs = config["model"]["fine_tune"]["epochs"] if not epochs_override else max(1, epochs_override // 2)
        logger.info(f"Starting Fine-Tuning phase for {fine_tune_epochs} epochs...")
        model = setup_fine_tuning(
            model,
            unfreeze_layers=config["model"]["fine_tune"]["unfreeze_layers"],
            fine_tune_lr=config["model"]["fine_tune"]["learning_rate"],
        )

        ft_callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=config["training"]["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=config["training"]["reduce_lr_factor"],
                patience=config["training"]["reduce_lr_patience"],
                min_lr=config["training"]["min_lr"],
                verbose=1,
            ),
            EpochProgressCallback(progress_file=progress_file, phase="fine_tuning"),
        ]

        ft_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=ft_callbacks,
            verbose=1,
        )

        for k, vals in ft_history.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    # Promote validated checkpoint to production model_save_path
    if os.path.exists(checkpoint_path):
        import shutil
        shutil.copyfile(checkpoint_path, model_save_path)
        logger.info(f"Promoted validated checkpoint {checkpoint_path} -> {model_save_path}")
    else:
        save_trained_model(model, model_save_path)

    # Save plot and json history
    plot_path = os.path.join(config["evaluation"]["plots_dir"], "training_history.png")
    plot_and_save_history(combined_history, plot_path)

    reports_dir = config["evaluation"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    history_json_path = os.path.join(reports_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    logger.info(f"Training history JSON saved to: {history_json_path}")

    return model, combined_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG16 DeepFake Detector")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test with small sample")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    args = parser.parse_args()

    if args.smoke_test:
        train_pipeline(
            config_path=args.config,
            max_train_samples=20,
            max_val_samples=10,
            epochs_override=args.epochs or 2,
            resume=args.resume,
        )
    else:
        train_pipeline(
            config_path=args.config,
            epochs_override=args.epochs,
            resume=args.resume,
        )
