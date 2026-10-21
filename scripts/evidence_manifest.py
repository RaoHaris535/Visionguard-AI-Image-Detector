from __future__ import annotations

import json
from pathlib import Path

REQUIRED = {
    "1.1": ["screenshots/environment-versions.png"],
    "1.2": ["../data/audit/images.csv", "../data/audit/summary.json"],
    "1.3": ["figures/sample_grid.png", "figures/dimension_histograms.png"],
    "2.4": ["figures/baseline_training_curves.png", "figures/transfer_training_curves.png"],
    "2.5": ["figures/baseline_confusion_matrix.png", "figures/transfer_confusion_matrix.png"],
    "3.5": ["../artifacts/model_comparison.json"],
    "4.2": ["screenshots/upload-validation.png"],
    "6.3": ["screenshots/compose-health.png"],
    "7": ["screenshots/cloud-api-postman.png"],
    "8": ["screenshots/flask-remote-api.png", "screenshots/history.png"],
}


def main() -> None:
    report_root = Path("reports")
    result = {
        task: [
            {"path": str(report_root / relative), "present": (report_root / relative).exists()}
            for relative in paths
        ]
        for task, paths in REQUIRED.items()
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/evidence_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    missing = [
        entry["path"] for entries in result.values() for entry in entries if not entry["present"]
    ]
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(f"Missing {len(missing)} evidence files")


if __name__ == "__main__":
    main()
