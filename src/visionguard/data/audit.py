from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir)
        label = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        row: dict[str, object] = {
            "path": str(relative),
            "label": label,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "is_corrupt": False,
            "error": None,
        }
        try:
            with Image.open(path) as image:
                image.load()
                row.update(
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    format=image.format,
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            row["is_corrupt"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    if not rows:
        raise ValueError(f"No supported image files found under {data_dir}")
    frame = pd.DataFrame(rows)
    frame["is_duplicate"] = frame.duplicated("sha256", keep=False)
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    readable = frame.loc[~frame["is_corrupt"]].copy()
    statistics: dict[str, object] = {
        "total_files": int(len(frame)),
        "corrupt_files": int(frame["is_corrupt"].sum()),
        "duplicate_files": int(frame["is_duplicate"].sum()),
        "duplicate_groups": int(readable.loc[readable["is_duplicate"], "sha256"].nunique()),
        "class_counts": {str(k): int(v) for k, v in Counter(frame["label"]).items()},
        "by_class": {},
    }
    for label, group in readable.groupby("label"):
        metrics: dict[str, dict[str, float]] = {}
        for column in ("width", "height", "file_size_bytes"):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            metrics[column] = {
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        statistics["by_class"][str(label)] = metrics  # type: ignore[index]
    return statistics


def save_eda(frame: pd.DataFrame, data_dir: Path, output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = frame.loc[~frame["is_corrupt"]].copy()
    sns.set_theme(style="whitegrid", context="notebook")

    labels = sorted(clean["label"].unique())
    samples_per_class = 6
    fig, axes = plt.subplots(len(labels), samples_per_class, figsize=(15, 3.2 * len(labels)))
    if len(labels) == 1:
        axes = [axes]
    for row_index, label in enumerate(labels):
        group = clean.loc[clean["label"] == label]
        sample = group.sample(min(samples_per_class, len(group)), random_state=seed)
        for column_index in range(samples_per_class):
            axis = axes[row_index][column_index]
            axis.axis("off")
            if column_index < len(sample):
                record = sample.iloc[column_index]
                with Image.open(data_dir / str(record["path"])) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(str(label).replace("_", " ").title(), fontsize=10)
    fig.suptitle("Representative images by class", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    long = clean.melt(
        id_vars=["label"],
        value_vars=["width", "height", "file_size_bytes"],
        var_name="measurement",
        value_name="value",
    )
    grid = sns.displot(
        data=long,
        x="value",
        hue="label",
        col="measurement",
        kind="hist",
        element="step",
        common_norm=False,
        facet_kws={"sharex": False, "sharey": False},
        height=4,
        aspect=1.05,
    )
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Image dimension and file-size distributions", y=1.05, fontweight="bold")
    grid.savefig(output_dir / "dimension_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def run(data_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = audit_dataset(data_dir)
    frame.to_csv(output_dir / "images.csv", index=False)
    summary = summarize(frame)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_eda(frame, data_dir, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the VisionGuard image dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/audit"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
