from __future__ import annotations

import argparse
import copy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from visionguard.config import load_yaml
from visionguard.data.pipeline import build_dataset
from visionguard.models.evaluation import choose_threshold, metrics_at_threshold
from visionguard.models.factory import build_transfer, configure_fine_tuning
from visionguard.reproducibility import set_global_seed
from visionguard.tracking.mlflow_tools import common_tags, configure_tracking


def dataset(config, split: str, training: bool):
    data = config["data"]
    return build_dataset(
        Path(data["split_dir"]) / f"{split}.csv",
        Path(data["root"]),
        image_size=tuple(data["image_size"]),
        batch_size=int(data["batch_size"]),
        positive_class=config["project"]["positive_class"],
        training=training,
        seed=int(config["project"]["seed"]),
        preprocessing="mobilenet_v2",
    )


def collect(model, input_data):
    labels = np.concatenate([labels.numpy() for _, labels in input_data]).astype(int)
    probabilities = model.predict(input_data, verbose=0).reshape(-1)
    return labels, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six controlled VisionGuard experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    base_config = load_yaml(args.config)
    matrix = load_yaml(args.matrix)["experiments"]
    seed = int(base_config["project"]["seed"])
    set_global_seed(seed)
    train_ds = dataset(base_config, "train", True)
    validation_ds = dataset(base_config, "validation", False)
    rows = []
    configure_tracking(base_config["tracking"]["experiment_name"])

    for definition in matrix:
        config = copy.deepcopy(base_config)
        model, backbone = build_transfer(
            (*config["data"]["image_size"], 3),
            definition["learning_rate"],
            definition["dropout"],
        )
        with mlflow.start_run(run_name=definition["name"]):
            mlflow.log_params(definition)
            mlflow.set_tags(common_tags(seed, "mobilenet_v2_controlled_matrix"))
            model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, verbose=2)
            if int(definition["unfreeze_layers"]) > 0:
                configure_fine_tuning(
                    model,
                    backbone,
                    unfreeze_layers=int(definition["unfreeze_layers"]),
                    learning_rate=float(definition["learning_rate"]) / 100,
                )
                model.fit(
                    train_ds,
                    validation_data=validation_ds,
                    epochs=max(2, args.epochs // 2),
                    verbose=2,
                )
            labels, probabilities = collect(model, validation_ds)
            threshold, _ = choose_threshold(labels, probabilities)
            metrics = metrics_at_threshold(labels, probabilities, threshold)
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            rows.append({"run": definition["name"], **definition, **metrics})

    output = Path("artifacts/experiment_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("f1", ascending=False).to_csv(output, index=False)
    print(pd.read_csv(output).to_string(index=False))


if __name__ == "__main__":
    main()
