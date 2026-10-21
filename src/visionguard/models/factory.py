from __future__ import annotations

import tensorflow as tf


def binary_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc"),
    ]


def build_baseline(input_shape: tuple[int, int, int], learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="visionguard_baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=binary_metrics(),
    )
    return model


def build_transfer(
    input_shape: tuple[int, int, int],
    learning_rate: float,
    dropout: float,
    *,
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="visionguard_mobilenet_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=binary_metrics(),
    )
    return model, base


def configure_fine_tuning(
    model: tf.keras.Model,
    base: tf.keras.Model,
    *,
    unfreeze_layers: int,
    learning_rate: float,
) -> None:
    base.trainable = True
    cutoff = max(0, len(base.layers) - unfreeze_layers)
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    for layer in base.layers[cutoff:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=binary_metrics(),
    )
