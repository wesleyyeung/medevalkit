import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from evalkit.compare import ModelComparer, delong_test

def test_delong_test_statistic_validity():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    y_score1 = np.random.rand(100)
    y_score2 = np.random.rand(100)

    p, auc1, auc2 = delong_test(y_true, y_score1, y_score2)
    assert 0 <= auc1 <= 1
    assert 0 <= auc2 <= 1
    assert 0 <= p <= 1

def test_model_comparer_bootstrap_and_delong():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
    clf1 = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    clf2 = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    comparer = ModelComparer({"LogReg": clf1, "RF": clf2}, X_test, y_test)

    result_bootstrap = comparer.compare_auc(method="bootstrap", parametric=True, n_resamples=100)
    result_delong = comparer.compare_auc(method="delong")

    assert isinstance(result_bootstrap, dict)
    assert isinstance(result_delong, dict)
    assert len(result_bootstrap) == len(result_delong)
