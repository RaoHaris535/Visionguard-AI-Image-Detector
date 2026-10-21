from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    audit_csv: Path,
    output_dir: Path,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and total less than one")
    frame = pd.read_csv(audit_csv)
    frame = frame.loc[(~frame["is_corrupt"].astype(bool)) & (~frame["is_duplicate"].astype(bool))]
    if frame["label"].nunique() != 2:
        raise ValueError("VisionGuard requires exactly two readable, deduplicated classes")

    train, remainder = train_test_split(
        frame,
        test_size=validation_fraction + test_fraction,
        stratify=frame["label"],
        random_state=seed,
    )
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test_fraction,
        stratify=remainder["label"],
        random_state=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"train": len(train), "validation": len(validation), "test": len(test)}
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        subset[["path", "label", "sha256"]].sort_values("path").to_csv(
            output_dir / f"{name}.csv", index=False
        )
    (output_dir / "metadata.json").write_text(
        json.dumps({"seed": seed, "counts": result}, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic stratified split manifests")
    parser.add_argument("--audit-csv", type=Path, default=Path("data/audit/images.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = create_splits(
        args.audit_csv,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
