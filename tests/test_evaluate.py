import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from medevalkit.evaluate import Evaluate

def test_evaluate_binary_classification_report():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    evaluator = Evaluate(model, X_test_scaled, y_test, classification=True)
    report = evaluator.generate_report()

    assert "text_report" in report
    assert isinstance(report["text_report"], str)

def test_evaluate_regression_report():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    evaluator = Evaluate(model, X_test, y_test, classification=False)
    report = evaluator.generate_report()

    assert "text_report" in report
    assert isinstance(report["text_report"], str)
