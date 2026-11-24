"""
Dataset loading and splitting module for Celeb-DF.
Strictly guarantees video-level separation to eliminate data leakage.
Integrates official Celeb-DF test protocol and builds tf.data pipelines.
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logging_utils import setup_logger
from src.utils.config import load_config
from src.preprocessing.video_processor import VideoProcessor

logger = setup_logger("DatasetLoader")

# Canonical label mapping
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "REAL", LABEL_FAKE: "FAKE"}


def discover_all_videos(raw_dir: str = "Celeb-DF") -> pd.DataFrame:
    """
    Recursively scans the raw dataset directory for all MP4 files and categorizes them.

    Args:
        raw_dir: Root dataset folder.

    Returns:
        pd.DataFrame with columns: ['relative_path', 'full_path', 'category', 'label']
    """
    records = []
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw dataset directory '{raw_dir}' not found.")

    for root, _, files in os.walk(raw_dir):
        for f in sorted(files):
            if f.lower().endswith(".mp4"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, raw_dir)
                category = rel_path.split(os.sep)[0]

                if category in ["Celeb-real", "YouTube-real"]:
                    label = LABEL_REAL
                elif category == "Celeb-synthesis":
                    label = LABEL_FAKE
                else:
                    continue  # Ignore unknown directories

                records.append({
                    "relative_path": rel_path,
                    "full_path": full_path,
                    "category": category,
                    "label": label,
                    "video_name": f,
                })

    df = pd.DataFrame(records)
    logger.info(
        f"Discovered {len(df)} total videos in {raw_dir}: "
        f"{(df['label'] == LABEL_REAL).sum()} REAL, {(df['label'] == LABEL_FAKE).sum()} FAKE."
    )
    return df


def load_official_test_split(
    test_list_path: str = "Celeb-DF/List_of_testing_videos.txt",
    raw_dir: str = "Celeb-DF",
) -> pd.DataFrame:
    """
    Loads official test split list.
    Note: Celeb-DF official List_of_testing_videos.txt has format:
    '<label> <relative_path>' where '0' was synthesis (FAKE) and '1' was real (REAL).
    We map synthesis to LABEL_FAKE (1) and real to LABEL_REAL (0).

    Args:
        test_list_path: Path to List_of_testing_videos.txt.
        raw_dir: Root dataset directory.

    Returns:
        pd.DataFrame of official test videos.
    """
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Official test list not found at: {test_list_path}")

    records = []
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                official_code, rel_path = parts[0], parts[1]
                full_path = os.path.join(raw_dir, rel_path)
                category = rel_path.split("/")[0]

                # Map to internal labels (0 = REAL, 1 = FAKE)
                if "synthesis" in category:
                    label = LABEL_FAKE
                else:
                    label = LABEL_REAL

                records.append({
                    "relative_path": rel_path,
                    "full_path": full_path,
                    "category": category,
                    "label": label,
                    "video_name": os.path.basename(rel_path),
                })

    df_test = pd.DataFrame(records)
    logger.info(
        f"Loaded official test split: {len(df_test)} videos "
        f"({(df_test['label'] == LABEL_REAL).sum()} REAL, {(df_test['label'] == LABEL_FAKE).sum()} FAKE)."
    )
    return df_test


def create_splits(
    config: Optional[Dict[str, any]] = None,
    save_splits: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates reproducible video-level train, validation, and test splits.
    Respects official test set and avoids any frame/video leakage.

    Args:
        config: Configuration dictionary.
        save_splits: Whether to persist splits to CSV files.

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    if config is None:
        config = load_config()

    raw_dir = config["dataset"]["raw_dir"]
    splits_dir = config["dataset"]["splits_dir"]
    test_list_path = config["dataset"]["official_test_list"]
    seed = config["dataset"]["random_seed"]
    val_ratio = config["dataset"]["val_split"]  # e.g., 0.15 of total dataset

    all_videos_df = discover_all_videos(raw_dir)
    test_df = load_official_test_split(test_list_path, raw_dir)

    # Exclude test videos from candidate train/val pool using relative_path
    test_rel_paths = set(test_df["relative_path"])
    train_val_df = all_videos_df[~all_videos_df["relative_path"].isin(test_rel_paths)].copy()

    # Calculate validation fraction relative to remaining train_val set
    total_count = len(all_videos_df)
    target_val_count = int(total_count * val_ratio)
    val_fraction_of_remaining = target_val_count / len(train_val_df)

    # Stratified split by label to maintain balanced class representation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remaining,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verify zero overlap
    train_set = set(train_df["relative_path"])
    val_set = set(val_df["relative_path"])
    test_set = set(test_df["relative_path"])

    assert len(train_set.intersection(val_set)) == 0, "Leakage detected between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage detected between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage detected between val and test!"

    logger.info(
        f"Splits successfully created (Video-Level, Zero Leakage):\n"
        f"  Train: {len(train_df)} videos ({(train_df['label'] == LABEL_REAL).sum()} REAL, {(train_df['label'] == LABEL_FAKE).sum()} FAKE)\n"
        f"  Val:   {len(val_df)} videos ({(val_df['label'] == LABEL_REAL).sum()} REAL, {(val_df['label'] == LABEL_FAKE).sum()} FAKE)\n"
        f"  Test:  {len(test_df)} videos ({(test_df['label'] == LABEL_REAL).sum()} REAL, {(test_df['label'] == LABEL_FAKE).sum()} FAKE)"
    )

    if save_splits:
        os.makedirs(splits_dir, exist_ok=True)
        train_path = os.path.join(splits_dir, "train_videos.csv")
        val_path = os.path.join(splits_dir, "val_videos.csv")
        test_path = os.path.join(splits_dir, "test_videos.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Splits saved to {splits_dir}/ (train, val, test CSVs).")

    return train_df, val_df, test_df


def load_splits(splits_dir: str = "data/splits") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads saved train/val/test CSV splits.
    If not existing, generates them automatically.
    """
    train_path = os.path.join(splits_dir, "train_videos.csv")
    val_path = os.path.join(splits_dir, "val_videos.csv")
    test_path = os.path.join(splits_dir, "test_videos.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        logger.info("Splits not found on disk. Generating now...")
        return create_splits()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def calculate_class_weights(df: pd.DataFrame) -> Dict[int, float]:
    """
    Computes balanced class weights for training.

    Args:
        df: DataFrame containing a 'label' column.

    Returns:
        dict: {0: weight_real, 1: weight_fake}
    """
    labels = df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    logger.info(f"Computed class weights: {weight_dict}")
    return weight_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and verify dataset splits.")
    parser.add_argument("--create-splits", action="store_true", help="Generate train/val/test splits")
    args = parser.parse_args()

    cfg = load_config()
    create_splits(cfg, save_splits=True)
