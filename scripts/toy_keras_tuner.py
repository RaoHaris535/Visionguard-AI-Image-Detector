from __future__ import annotations

import keras_tuner as kt
import tensorflow as tf


def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.Choice("units", [32, 64, 128]), activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:10000].astype("float32") / 255.0
    y_train = y_train[:10000]
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=6,
        seed=42,
        directory="artifacts/tuner",
        project_name="fashion_mnist_tool_lab",
        overwrite=True,
    )
    tuner.search(x_train, y_train, validation_split=0.2, epochs=5, verbose=2)
    tuner.results_summary()
    print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
