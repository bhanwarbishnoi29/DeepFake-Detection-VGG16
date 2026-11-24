"""
VGG16 Transfer Learning Model for Video DeepFake Detection.
Implements ImageNet pretrained feature extraction, classification head,
and configurable fine-tuning of top convolutional layers.
"""

import os
from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

from src.utils.logging_utils import setup_logger

logger = setup_logger("VGG16Model")


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0001,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Constructs the VGG16 Transfer Learning model.

    Architecture:
        Input(224, 224, 3)
        ↓
        VGG16 Base (include_top=False, weights='imagenet')
        ↓
        GlobalAveragePooling2D
        ↓
        Dense(dense_units, activation='relu')
        ↓
        Dropout(dropout_rate)
        ↓
        Dense(1, activation='sigmoid')

    Args:
        input_shape: Dimensions of input face images (224, 224, 3).
        weights: Weights to load ('imagenet' or None).
        dense_units: Number of units in fully connected classification layer.
        dropout_rate: Dropout regularization fraction.
        learning_rate: Adam optimizer learning rate.
        freeze_base: Whether to freeze VGG16 convolutional layers.

    Returns:
        tf.keras.Model: Compiled binary classification model.
    """
    inputs = layers.Input(shape=input_shape, name="input_face")

    local_weights = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "pretrained", "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    )

    if weights == "imagenet" and os.path.exists(local_weights):
        logger.info(f"Loading local ImageNet weights from: {local_weights}")
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )
        base_model.load_weights(local_weights)
    else:
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
        )

    base_model.trainable = not freeze_base
    if freeze_base:
        logger.info(f"Frozen all {len(base_model.layers)} layers of VGG16 base model.")
    else:
        logger.info("VGG16 base model layers set to trainable.")

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_probability", dtype="float32")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_DeepFake_Detector")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    logger.info("VGG16 Transfer Learning model built and compiled successfully.")
    return model


def setup_fine_tuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 4,
    fine_tune_lr: float = 0.00001,
) -> tf.keras.Model:
    """
    Configures fine-tuning by unfreezing the top N convolutional layers of VGG16
    and recompiling with a low learning rate.

    Args:
        model: Trained or built VGG16 model.
        unfreeze_layers: Number of top layers to unfreeze (e.g., 4 unfreezes block5_conv1..3 + pool).
        fine_tune_lr: Low learning rate for delicate fine-tuning (e.g., 1e-5).

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.
    """
    logger.info(f"Configuring fine-tuning: unfreezing top {unfreeze_layers} layers with LR={fine_tune_lr}")

    # Search for VGG16 convolutional layers
    # In functional model, layers are sequenced; count backwards from base
    # Find layers before classification head (global_average_pool)
    vgg_layers = []
    for layer in model.layers:
        if layer.name == "global_average_pool":
            break
        vgg_layers.append(layer)

    total_base_layers = len(vgg_layers)
    freeze_until = max(0, total_base_layers - unfreeze_layers)

    for i, layer in enumerate(vgg_layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
            logger.info(f"Unfrozen layer: {layer.name}")

    # Re-compile model with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    trainable_count = sum(len(l.trainable_weights) for l in model.layers)
    logger.info(f"Fine-tuning compiled. Trainable weight tensors count: {trainable_count}")
    return model


def get_model_summary_string(model: tf.keras.Model) -> str:
    """Returns a string representation of the model summary."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
