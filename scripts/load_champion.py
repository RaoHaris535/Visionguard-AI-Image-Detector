from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the champion model by MLflow alias")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-name", default="visionguard-classifier")
    args = parser.parse_args()
    model = mlflow.tensorflow.load_model(f"models:/{args.model_name}@champion")
    with Image.open(args.image) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    probability = float(model.predict(array[None, ...], verbose=0).reshape(-1)[0])
    print(
        {"label": "AI generated" if probability >= 0.5 else "Real", "ai_probability": probability}
    )


if __name__ == "__main__":
    main()
