from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class AppSettings:
    secret_key: str
    model_path: Path
    model_version: str
    decision_threshold: float
    max_upload_bytes: int
    database_url: str
    inference_mode: str
    inference_api_url: str | None
    inference_api_key: str | None

    @classmethod
    def from_env(cls) -> AppSettings:
        load_dotenv()
        mode = os.getenv("INFERENCE_MODE", "local").lower()
        if mode not in {"local", "remote"}:
            raise ValueError("INFERENCE_MODE must be 'local' or 'remote'")
        threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))
        if not 0.0 < threshold < 1.0:
            raise ValueError("DECISION_THRESHOLD must be between 0 and 1")
        return cls(
            secret_key=os.getenv("SECRET_KEY", "development-only-change-me"),
            model_path=Path(os.getenv("MODEL_PATH", "models/champion.keras")),
            model_version=os.getenv("MODEL_VERSION", "unregistered"),
            decision_threshold=threshold,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024,
            database_url=os.getenv("DATABASE_URL", "sqlite:///visionguard.sqlite3"),
            inference_mode=mode,
            inference_api_url=os.getenv("INFERENCE_API_URL"),
            inference_api_key=os.getenv("INFERENCE_API_KEY"),
        )
