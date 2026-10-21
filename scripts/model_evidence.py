from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from visionguard.models.factory import build_baseline, build_transfer


def layer_rows(model: tf.keras.Model):
    for layer in model.layers:
        yield {
            "layer": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(getattr(layer, "output", "unknown").shape),
            "parameters": layer.count_params(),
            "trainable": layer.trainable,
        }


def main() -> None:
    output = Path("artifacts/model_evidence")
    output.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline((224, 224, 3), 1e-3)
    transfer, base = build_transfer((224, 224, 3), 1e-3, 0.3, weights=None)
    for name, model in (("baseline", baseline), ("transfer", transfer)):
        rows = list(layer_rows(model))
        with (output / f"{name}_layers.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        lines: list[str] = []
        model.summary(print_fn=lines.append)
        (output / f"{name}_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (output / "frozen_trainable_layers.txt").write_text(
        "\n".join(layer.name for layer in base.layers if layer.trainable), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axis("off")
    axis.text(
        0.01,
        0.98,
        (output / "baseline_summary.txt").read_text(encoding="utf-8"),
        va="top",
        family="monospace",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig("reports/figures/baseline_model_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
