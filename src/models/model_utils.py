"""
Model utilities for checkpoint management, weights loading, and metrics tracking.
"""

import os
from typing import Optional
import tensorflow as tf
from src.utils.logging_utils import setup_logger

logger = setup_logger("ModelUtils")


def save_trained_model(model: tf.keras.Model, filepath: str) -> str:
    """
    Saves trained Keras model safely.

    Args:
        model: Trained model instance.
        filepath: Target filepath (e.g. .keras or .h5).

    Returns:
        Absolute path to saved model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    model.save(filepath)
    logger.info(f"Model successfully saved to: {filepath}")
    return os.path.abspath(filepath)


def load_trained_model(filepath: str) -> tf.keras.Model:
    """
    Loads saved Keras model from disk.

    Args:
        filepath: Path to saved model file.

    Returns:
        tf.keras.Model: Loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    logger.info(f"Loading trained model from: {filepath}")
    model = tf.keras.models.load_model(filepath)
    logger.info("Model loaded successfully.")
    return model
