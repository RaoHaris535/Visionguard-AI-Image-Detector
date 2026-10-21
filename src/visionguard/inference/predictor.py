from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded into a supported RGB image."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    ai_probability: float
    threshold: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class Predictor:
    """Load a Keras model once and reuse it for every request."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        image_size: tuple[int, int] = (224, 224),
        preprocessing: str = "mobilenet_v2",
    ) -> None:
        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.image_size = image_size
        self.preprocessing = preprocessing

    def preprocess(self, payload: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            tensor = self.tf.io.decode_image(payload, channels=3, expand_animations=False)
            tensor.set_shape([None, None, 3])
            tensor = self.tf.image.resize_with_pad(
                tensor,
                target_height=self.image_size[0],
                target_width=self.image_size[1],
            )
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            self.tf.errors.InvalidArgumentError,
        ) as exc:
            raise InvalidImageError("The uploaded file is not a decodable image.") from exc
        if self.preprocessing == "mobilenet_v2":
            tensor = self.tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
        elif self.preprocessing == "scale_01":
            tensor = tensor / 255.0
        else:
            raise ValueError(f"Unsupported preprocessing mode: {self.preprocessing}")
        return np.expand_dims(tensor.numpy(), axis=0)

    def predict(self, payload: bytes) -> Prediction:
        probability = float(self.model.predict(self.preprocess(payload), verbose=0).reshape(-1)[0])
        is_ai = probability >= self.threshold
        return Prediction(
            label="AI generated" if is_ai else "Real",
            confidence=probability if is_ai else 1.0 - probability,
            ai_probability=probability,
            threshold=self.threshold,
        )
