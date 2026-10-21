from pathlib import Path

import pandas as pd

from visionguard.data.splits import create_splits


def test_splits_are_disjoint_and_complete(tmp_path: Path):
    rows = []
    for label in ("real", "ai_generated"):
        for index in range(20):
            rows.append(
                {
                    "path": f"{label}/{index}.jpg",
                    "label": label,
                    "sha256": f"{label}-{index}",
                    "is_corrupt": False,
                    "is_duplicate": False,
                }
            )
    audit_csv = tmp_path / "audit.csv"
    pd.DataFrame(rows).to_csv(audit_csv, index=False)
    output = tmp_path / "splits"
    counts = create_splits(audit_csv, output, seed=42)
    sets = {name: set(pd.read_csv(output / f"{name}.csv")["path"]) for name in counts}
    assert sum(counts.values()) == 40
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
