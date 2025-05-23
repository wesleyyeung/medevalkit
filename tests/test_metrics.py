import numpy as np
from medevalkit.metrics import (
    binary_classification_metrics,
    regression_metrics
)

def test_binary_classification_metrics_structure():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
    results = binary_classification_metrics(y_true, y_prob)
    expected_keys = {"auc", "accuracy", "precision", "recall", "f1"}

    assert isinstance(results, dict)
    assert expected_keys.issubset(results.keys())

def test_regression_metrics_structure():
    y_true = np.array([3.2, 4.1, 5.0])
    y_pred = np.array([2.9, 4.2, 5.1])
    results = regression_metrics(y_true, y_pred)
    expected_keys = {"r2", "mse", "mae"}

    assert isinstance(results, dict)
    assert expected_keys.issubset(results.keys())
