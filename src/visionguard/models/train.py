from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from visionguard.config import load_yaml
from visionguard.data.pipeline import build_dataset
from visionguard.models.evaluation import choose_threshold, save_evaluation, save_training_curves
from visionguard.models.factory import build_baseline, build_transfer, configure_fine_tuning
from visionguard.reproducibility import set_global_seed
from visionguard.tracking.mlflow_tools import common_tags, configure_tracking


def callbacks(model_path: Path, patience: int, reduce_lr_patience: int):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=reduce_lr_patience, min_lr=1e-7
        ),
    ]


def _datasets(config: dict[str, Any], preprocessing: str):
    data = config["data"]
    seed = int(config["project"]["seed"])
    kwargs = {
        "data_root": Path(data["root"]),
        "image_size": tuple(data["image_size"]),
        "batch_size": int(data["batch_size"]),
        "positive_class": config["project"]["positive_class"],
        "seed": seed,
        "preprocessing": preprocessing,
    }
    split_dir = Path(data["split_dir"])
    return (
        build_dataset(split_dir / "train.csv", training=True, **kwargs),
        build_dataset(split_dir / "validation.csv", training=False, **kwargs),
        build_dataset(split_dir / "test.csv", training=False, **kwargs),
    )


def _labels_and_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset):
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset]).astype(int)
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    return labels, probabilities


def _log_history(history: tf.keras.callbacks.History, offset: int = 0) -> None:
    for index in range(len(history.history["loss"])):
        metrics = {key: float(values[index]) for key, values in history.history.items()}
        mlflow.log_metrics(metrics, step=index + offset)


def train_baseline(config: dict[str, Any]) -> dict[str, float]:
    data, training, outputs = config["data"], config["training"], config["outputs"]
    seed = int(config["project"]["seed"])
    train_ds, validation_ds, test_ds = _datasets(config, "scale_01")
    model = build_baseline((*data["image_size"], 3), training["baseline_learning_rate"])
    model_path = Path(outputs["model_dir"]) / "baseline.keras"
    configure_tracking(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name="baseline-cnn-manual"):
        mlflow.log_params(
            {
                "architecture": "custom_cnn",
                "image_size": str(data["image_size"]),
                "batch_size": data["batch_size"],
                "learning_rate": training["baseline_learning_rate"],
            }
        )
        mlflow.set_tags(common_tags(seed, "baseline_cnn"))
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=training["epochs"],
            callbacks=callbacks(model_path, training["patience"], training["reduce_lr_patience"]),
            verbose=2,
        )
        _log_history(history)
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        curve_path = Path(outputs["figure_dir"]) / "baseline_training_curves.png"
        save_training_curves(history.history, curve_path, best_epoch)
        validation_labels, validation_probabilities = _labels_and_probabilities(
            model, validation_ds
        )
        threshold, threshold_frame = choose_threshold(validation_labels, validation_probabilities)
        threshold_frame.to_csv(
            Path(outputs["artifact_dir"]) / "baseline_threshold_search.csv", index=False
        )
        test_labels, test_probabilities = _labels_and_probabilities(model, test_ds)
        metrics = save_evaluation(
            test_labels, test_probabilities, threshold, Path(outputs["figure_dir"]), "baseline"
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_artifact(str(curve_path), artifact_path="figures")
        mlflow.log_artifacts(outputs["figure_dir"], artifact_path="figures")
        mlflow.log_artifacts(data["split_dir"], artifact_path="splits")
        mlflow.tensorflow.log_model(model, name="model")
    return metrics


def train_transfer(config: dict[str, Any]) -> dict[str, float]:
    data, training, outputs = config["data"], config["training"], config["outputs"]
    seed = int(config["project"]["seed"])
    train_ds, validation_ds, test_ds = _datasets(config, "mobilenet_v2")
    model, base = build_transfer(
        (*data["image_size"], 3), training["transfer_learning_rate"], training["dropout"]
    )
    model_path = Path(outputs["model_dir"]) / "transfer.keras"
    configure_tracking(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name="mobilenet-v2-autolog"):
        mlflow.tensorflow.autolog(log_models=False)
        mlflow.set_tags(common_tags(seed, "mobilenet_v2"))
        head_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=training["epochs"],
            callbacks=callbacks(model_path, training["patience"], training["reduce_lr_patience"]),
            verbose=2,
        )
        configure_fine_tuning(
            model,
            base,
            unfreeze_layers=training["unfreeze_layers"],
            learning_rate=training["fine_tune_learning_rate"],
        )
        fine_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=training["epochs"],
            callbacks=callbacks(model_path, training["patience"], training["reduce_lr_patience"]),
            verbose=2,
        )
        combined = {
            key: head_history.history[key] + fine_history.history[key]
            for key in head_history.history
        }
        best_epoch = int(np.argmin(combined["val_loss"]) + 1)
        curve_path = Path(outputs["figure_dir"]) / "transfer_training_curves.png"
        save_training_curves(combined, curve_path, best_epoch)
        validation_labels, validation_probabilities = _labels_and_probabilities(
            model, validation_ds
        )
        threshold, threshold_frame = choose_threshold(validation_labels, validation_probabilities)
        threshold_frame.to_csv(
            Path(outputs["artifact_dir"]) / "transfer_threshold_search.csv", index=False
        )
        test_labels, test_probabilities = _labels_and_probabilities(model, test_ds)
        metrics = save_evaluation(
            test_labels, test_probabilities, threshold, Path(outputs["figure_dir"]), "transfer"
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_artifacts(outputs["figure_dir"], artifact_path="figures")
        mlflow.log_artifacts(data["split_dir"], artifact_path="splits")
        mlflow.tensorflow.log_model(
            model, name="model", registered_model_name="visionguard-classifier"
        )
    return metrics


def run(config_path: Path) -> dict[str, dict[str, float]]:
    config = load_yaml(config_path)
    set_global_seed(int(config["project"]["seed"]))
    Path(config["outputs"]["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["figure_dir"]).mkdir(parents=True, exist_ok=True)
    results = {"baseline": train_baseline(config), "transfer": train_transfer(config)}
    Path(config["outputs"]["artifact_dir"], "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both VisionGuard model families")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
