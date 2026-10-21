from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf


def run(learning_rate: float, epochs: int) -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:5000].astype("float32") / 255.0
    y_train = y_train[:5000]
    x_test = x_test[:1000].astype("float32") / 255.0
    y_test = y_test[:1000]
    model = tf.keras.Sequential(
        [
            tf.keras.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    mlflow.set_experiment("visionguard-tool-lab-mlflow")
    with mlflow.start_run(run_name=f"toy-lr-{learning_rate:g}"):
        mlflow.log_params({"learning_rate": learning_rate, "epochs": epochs, "units": 64})
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, verbose=2)
        for step in range(epochs):
            mlflow.log_metrics(
                {key: float(values[step]) for key, values in history.history.items()}, step=step
            )
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": accuracy})
        figure_path = Path("artifacts/toy_mlflow_curve.png")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.plot(history.history["accuracy"], label="training")
        plt.plot(history.history["val_accuracy"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160)
        plt.close()
        mlflow.log_artifact(str(figure_path), artifact_path="figures")
        mlflow.tensorflow.log_model(model, name="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    run(args.learning_rate, args.epochs)
