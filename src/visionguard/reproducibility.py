from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds before TensorFlow creates any random state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    with suppress(AttributeError, RuntimeError):
        tf.config.experimental.enable_op_determinism()
