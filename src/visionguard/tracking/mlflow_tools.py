from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient


def configure_tracking(experiment_name: str) -> None:
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "uncommitted"


def dvc_data_version(pointer: Path = Path("data/raw.dvc")) -> str:
    if not pointer.exists():
        return "untracked"
    for line in pointer.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("md5:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def common_tags(seed: int, model_family: str) -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "dvc_data_version": dvc_data_version(),
        "seed": str(seed),
        "model_family": model_family,
    }


def register_and_alias(model_uri: str, name: str, alias: str = "champion") -> int:
    result = mlflow.register_model(model_uri=model_uri, name=name)
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, result.version)
    return int(result.version)
