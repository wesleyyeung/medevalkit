import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evalkit.calibration import MulticlassCalibration

def test_multiclass_calibration_outputs():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)

    mc = MulticlassCalibration(y_test, y_prob)
    ece = mc.expected_calibration_error()
    curves = mc.one_vs_rest_curves()

    assert isinstance(ece, float)
    assert isinstance(curves, dict)
    assert all("calibration_curve" in c and "brier_score" in c for c in curves.values())
