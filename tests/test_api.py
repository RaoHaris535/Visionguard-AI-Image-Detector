from dataclasses import dataclass
from io import BytesIO

import pytest

from visionguard.api.app import create_api
from visionguard.inference.predictor import Prediction


@dataclass
class FakePredictor:
    def predict(self, payload: bytes) -> Prediction:
        assert payload
        return Prediction("AI generated", 0.91, 0.91, 0.6)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    app = create_api(FakePredictor())
    app.config["TESTING"] = True
    return app.test_client()


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_requires_api_key(client):
    response = client.post("/predict")
    assert response.status_code == 401


def test_predict_returns_structured_result(client):
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        data={"image": (BytesIO(b"payload"), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["label"] == "AI generated"
