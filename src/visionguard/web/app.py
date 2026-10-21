from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from visionguard.config import AppSettings
from visionguard.inference.predictor import InvalidImageError, Prediction, Predictor
from visionguard.web.database import HistoryStore

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _remote_prediction(settings: AppSettings, payload: bytes, filename: str) -> Prediction:
    if not settings.inference_api_url or not settings.inference_api_key:
        raise RuntimeError("Remote inference requires INFERENCE_API_URL and INFERENCE_API_KEY")
    response = requests.post(
        f"{settings.inference_api_url.rstrip('/')}/predict",
        files={"image": (filename, payload, "application/octet-stream")},
        headers={"X-API-Key": settings.inference_api_key},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    return Prediction(
        label=result["label"],
        confidence=float(result["confidence"]),
        ai_probability=float(result["ai_probability"]),
        threshold=float(result["threshold"]),
    )


def create_app(
    settings: AppSettings | None = None, predictor_override: Predictor | None = None
) -> Flask:
    settings = settings or AppSettings.from_env()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
    )
    store = HistoryStore(settings.database_url)
    store.initialize()

    predictor: Predictor | None = predictor_override
    if settings.inference_mode == "local" and predictor is None:
        predictor = Predictor(settings.model_path, threshold=settings.decision_threshold)

    @app.get("/")
    def index():
        return render_template(
            "index.html", max_upload_mb=settings.max_upload_bytes // (1024 * 1024)
        )

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template("error.html", message="Choose an image before submitting."), 400
        safe_name = secure_filename(uploaded.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return render_template("error.html", message="Use a JPG, PNG, WEBP, or BMP image."), 400
        payload = uploaded.read()
        if not payload:
            return render_template("error.html", message="The uploaded file is empty."), 400
        try:
            result = (
                predictor.predict(payload)
                if predictor is not None
                else _remote_prediction(settings, payload, safe_name)
            )
        except InvalidImageError as exc:
            return render_template("error.html", message=str(exc)), 400
        except requests.RequestException:
            app.logger.exception("Remote inference request failed")
            return render_template(
                "error.html", message="The inference service is temporarily unavailable."
            ), 503

        store.add(
            original_filename=safe_name,
            image_sha256=hashlib.sha256(payload).hexdigest(),
            verdict=result.label,
            confidence=result.confidence,
            ai_probability=result.ai_probability,
            threshold=result.threshold,
            model_version=settings.model_version,
            inference_source=settings.inference_mode,
        )
        return render_template(
            "result.html", result=result, filename=safe_name, model_version=settings.model_version
        )

    @app.get("/history")
    def history():
        return render_template("history.html", records=store.recent())

    @app.get("/health")
    def health():
        database_ok = store.healthcheck()
        model_ok = predictor is not None if settings.inference_mode == "local" else True
        status = 200 if database_ok and model_ok else 503
        return {
            "status": "healthy" if status == 200 else "unhealthy",
            "database": "available" if database_ok else "unavailable",
            "inference_mode": settings.inference_mode,
            "model_loaded": model_ok,
            "model_version": settings.model_version,
        }, status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return render_template(
            "error.html",
            message=f"The image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        logging.exception("Unhandled application error", exc_info=error)
        return render_template("error.html", message="An unexpected error occurred."), 500

    return app


def main() -> None:
    from waitress import serve

    serve(create_app(), host="0.0.0.0", port=8000, threads=4)


if __name__ == "__main__":
    main()
