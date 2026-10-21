import numpy as np

from visionguard.models.evaluation import choose_threshold, metrics_at_threshold


def test_threshold_search_returns_valid_threshold():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, frame = choose_threshold(labels, probabilities)
    assert 0 < threshold < 1
    assert frame.loc[frame["threshold"] == threshold, "f1"].iloc[0] == 1.0


def test_metrics_use_selected_threshold():
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.2, 0.8, 0.7, 0.1])
    metrics = metrics_at_threshold(labels, probabilities, 0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
