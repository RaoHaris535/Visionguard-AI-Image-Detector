from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import tensorflow as tf


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]):
    contents = tf.io.read_file(path)
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    image = tf.cast(image, tf.float32)
    return image, label


def augmentation(seed: int) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.1, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="training_augmentation",
    )


def build_dataset(
    manifest: Path,
    data_root: Path,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    positive_class: str,
    training: bool,
    seed: int,
    preprocessing: Literal["scale_01", "mobilenet_v2"] = "scale_01",
) -> tf.data.Dataset:
    frame = pd.read_csv(manifest)
    paths = [str(data_root / relative) for relative in frame["path"]]
    labels = (frame["label"] == positive_class).astype("float32").tolist()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    dataset = dataset.batch(batch_size)
    if training:
        augmenter = augmentation(seed)
        dataset = dataset.map(
            lambda image, label: (augmenter(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if preprocessing == "scale_01":
        dataset = dataset.map(
            lambda image, label: (image / 255.0, label), num_parallel_calls=tf.data.AUTOTUNE
        )
    elif preprocessing == "mobilenet_v2":
        dataset = dataset.map(
            lambda image, label: (
                tf.keras.applications.mobilenet_v2.preprocess_input(image),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")
    return dataset.prefetch(tf.data.AUTOTUNE)
