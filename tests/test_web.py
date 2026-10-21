from io import BytesIO
from pathlib import Path

from PIL import Image

from visionguard.config import AppSettings
from visionguard.inference.predictor import Prediction
from visionguard.web.app import create_app


class FakePredictor:
    def predict(self, payload: bytes) -> Prediction:
        assert payload
        return Prediction("Real", 0.88, 0.12, 0.5)


def image_bytes() -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", (16, 16), "navy").save(stream, format="PNG")
    stream.seek(0)
    return stream


def client(tmp_path: Path):
    settings = AppSettings(
        secret_key="test",
        model_path=tmp_path / "unused.keras",
        model_version="test-v1",
        decision_threshold=0.5,
        max_upload_bytes=1024 * 1024,
        database_url=f"sqlite:///{tmp_path / 'history.sqlite3'}",
        inference_mode="local",
        inference_api_url=None,
        inference_api_key=None,
    )
    app = create_app(settings, predictor_override=FakePredictor())
    app.config["TESTING"] = True
    return app.test_client()


def test_missing_upload_returns_friendly_error(tmp_path):
    response = client(tmp_path).post("/predict", data={})
    assert response.status_code == 400
    assert b"Choose an image" in response.data


def test_wrong_extension_is_rejected(tmp_path):
    response = client(tmp_path).post(
        "/predict",
        data={"image": (BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"JPG, PNG" in response.data


def test_prediction_is_recorded_in_history(tmp_path):
    test_client = client(tmp_path)
    response = test_client.post(
        "/predict",
        data={"image": (image_bytes(), "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"88.0%" in response.data
    history = test_client.get("/history")
    assert b"sample.png" in history.data
    assert b"test-v1" in history.data
