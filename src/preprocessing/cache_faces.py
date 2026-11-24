"""
Pre-caching utility for Celeb-DF dataset face extraction.
Extracts and persists 224x224 face crops to data/processed/faces/
using thread-isolated VideoProcessor instances.
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.dataset.dataset_loader import load_splits
from src.preprocessing.video_processor import VideoProcessor

logger = setup_logger("FaceCacheUtility")

_thread_local = threading.local()


def get_thread_processor(config: dict) -> VideoProcessor:
    """Provides a thread-isolated VideoProcessor instance to guarantee thread safety."""
    if not hasattr(_thread_local, "processor"):
        _thread_local.processor = VideoProcessor(
            frames_per_video=config["preprocessing"]["frames_per_video"],
            target_size=tuple(config["preprocessing"]["image_size"]),
            face_method=config["preprocessing"]["face_detection"]["method"],
            margin_ratio=config["preprocessing"]["face_detection"]["margin_ratio"],
        )
    return _thread_local.processor


def process_single_video(args: Tuple[pd.Series, str, dict]) -> Tuple[str, int, bool]:
    """
    Processes a single video row and writes compressed npz face cache.

    Returns:
        (rel_path, num_faces, was_cached)
    """
    row, cache_dir, config = args
    rel_path = row["relative_path"]
    full_path = row["full_path"]
    safe_name = rel_path.replace(os.sep, "__").replace(".mp4", ".npz")
    cache_path = os.path.join(cache_dir, safe_name)

    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if "faces" in data and len(data["faces"]) > 0:
                return rel_path, len(data["faces"]), True
        except Exception:
            pass  # Recompute if corrupt

    processor = get_thread_processor(config)
    faces, meta = processor.process_video(full_path)

    if faces is not None and len(faces) > 0:
        np.savez_compressed(cache_path, faces=faces)
        return rel_path, len(faces), False
    else:
        # Fallback dummy zero-face array so video isn't repeatedly re-extracted
        dummy = np.zeros((1, config["preprocessing"]["image_size"][0], config["preprocessing"]["image_size"][1], 3), dtype=np.uint8)
        np.savez_compressed(cache_path, faces=dummy)
        return rel_path, 0, False


def run_caching(max_workers: int = 4, include_test: bool = False):
    config = load_config()
    cache_dir = os.path.join(config["dataset"]["processed_dir"], "faces")
    os.makedirs(cache_dir, exist_ok=True)

    train_df, val_df, test_df = load_splits(config["dataset"]["splits_dir"])
    dfs = [train_df, val_df]
    if include_test:
        dfs.append(test_df)
    dataset_df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Starting face caching for {len(dataset_df)} videos ({len(train_df)} train, {len(val_df)} val) using {max_workers} workers...")
    t0 = time.time()

    tasks = [(row, cache_dir, config) for _, row in dataset_df.iterrows()]
    total_faces = 0
    already_cached = 0
    newly_cached = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (rel_path, num_faces, was_cached) in enumerate(executor.map(process_single_video, tasks)):
            total_faces += num_faces
            if was_cached:
                already_cached += 1
            else:
                newly_cached += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(tasks):
                elapsed = time.time() - t0
                logger.info(f"Progress: {idx + 1}/{len(tasks)} videos ({already_cached} cached, {newly_cached} newly processed, {elapsed:.1f}s elapsed)")

    total_time = time.time() - t0
    logger.info(
        f"Face caching completed in {total_time:.2f}s!\n"
        f"  Total videos:     {len(tasks)}\n"
        f"  Total face crops: {total_faces}\n"
        f"  Cache directory:  {cache_dir}\n"
        f"  Cache files:      {len(os.listdir(cache_dir))}"
    )


if __name__ == "__main__":
    workers = 4
    if len(sys.argv) > 1:
        workers = int(sys.argv[1])
    run_caching(max_workers=workers)
