from __future__ import annotations

import hmac
import os
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from visionguard.inference.predictor import InvalidImageError, Predictor


def create_api(predictor: Predictor | None = None) -> Flask:
    app = Flask(__name__)
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = max_bytes
    api_key = os.getenv("INFERENCE_API_KEY")
    if predictor is None:
        predictor = Predictor(
            Path(os.getenv("MODEL_PATH", "models/champion.keras")),
            threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
        )

    @app.before_request
    def authenticate():
        if request.path == "/health":
            return None
        supplied = request.headers.get("X-API-Key", "")
        if not api_key or not hmac.compare_digest(supplied, api_key):
            return jsonify(error="Unauthorized"), 401
        return None

    @app.get("/health")
    def health():
        return {"status": "healthy", "model_loaded": True}

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(error="A multipart image field is required."), 400
        try:
            result = predictor.predict(uploaded.read())
        except InvalidImageError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(result.to_dict())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error=f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit."), 413

    return app


def main() -> None:
    from waitress import serve

    serve(create_api(), host="0.0.0.0", port=int(os.getenv("PORT", "8080")), threads=4)


if __name__ == "__main__":
    main()
