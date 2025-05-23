import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from medevalkit.simulation import EvaluateWithSimulation

def test_simulated_evaluation_outputs_auc():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    evaluator = EvaluateWithSimulation(model, X_test, y_test, classification=True)
    df = evaluator.get_metrics_df()
    assert 'auc' in df.columns
    assert df['auc'].iloc[0] > 0.5
