"""
Configuration utility module for DeepFake Detection System.
Provides loading, validating, and accessing settings from config.yaml.
"""

import os
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads and validates the YAML configuration file.

    Args:
        config_path: Relative or absolute path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If essential configuration sections are missing.
    """
    if not os.path.isabs(config_path):
        # Resolve path relative to project root (assumed to be working directory or parent of src)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        resolved_path = os.path.join(project_root, config_path)
        if os.path.exists(resolved_path):
            config_path = resolved_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate essential sections
    required_sections = ["dataset", "preprocessing", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}' in {config_path}")

    return config


def get_model_save_path(config: Dict[str, Any]) -> str:
    """Helper to get full model save path."""
    model_dir = config["training"].get("model_dir", "models/trained_model")
    filename = config["training"].get("model_filename", "vgg16_deepfake_detector.keras")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)
