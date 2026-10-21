from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from visionguard.config import load_yaml
from visionguard.data.pipeline import build_dataset
from visionguard.models.factory import binary_metrics
from visionguard.reproducibility import set_global_seed


class VisionGuardHyperModel(kt.HyperModel):
    def __init__(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        base = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )
        unfreeze = hp.Choice("unfreeze_layers", [0, 10, 20, 40])
        base.trainable = unfreeze > 0
        for layer in base.layers[: max(0, len(base.layers) - unfreeze)]:
            layer.trainable = False
        for layer in base.layers[-unfreeze:] if unfreeze else []:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        inputs = tf.keras.Input(self.input_shape)
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(hp.Choice("dropout", [0.2, 0.3, 0.5]))(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4, 3e-5])
            ),
            loss="binary_crossentropy",
            metrics=binary_metrics(),
        )
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune VisionGuard transfer-learning settings")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--max-trials", type=int, default=8)
    args = parser.parse_args()
    if args.max_trials < 8:
        raise ValueError("The assignment requires at least eight trials")
    config = load_yaml(args.config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    data = config["data"]
    kwargs = {
        "data_root": Path(data["root"]),
        "image_size": tuple(data["image_size"]),
        "batch_size": int(data["batch_size"]),
        "positive_class": config["project"]["positive_class"],
        "seed": seed,
        "preprocessing": "mobilenet_v2",
    }
    train_ds = build_dataset(Path(data["split_dir"]) / "train.csv", training=True, **kwargs)
    validation_ds = build_dataset(
        Path(data["split_dir"]) / "validation.csv", training=False, **kwargs
    )
    tuner = kt.RandomSearch(
        VisionGuardHyperModel((*data["image_size"], 3)),
        objective=kt.Objective("val_roc_auc", direction="max"),
        max_trials=args.max_trials,
        seed=seed,
        directory="artifacts/tuner",
        project_name="visionguard",
        overwrite=True,
    )
    tuner.search(
        train_ds,
        validation_data=validation_ds,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)],
        verbose=2,
    )
    best = tuner.get_best_hyperparameters(1)[0].values
    Path("artifacts/tuner_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    tuner.results_summary()


if __name__ == "__main__":
    main()
