import numpy as np
from sklearn.metrics import confusion_matrix
from evalkit.threshold import ThresholdAnalysis, DecisionCurveAnalysis

def test_threshold_analysis_outputs_expected_metrics():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.76, 0.33, 0.88, 0.25, 0.67, 0.59])

    ta = ThresholdAnalysis(y_true, y_prob)
    summary = ta.compute_summary()

    assert isinstance(summary, dict)
    assert 0.5 in summary
    for metric in ['sensitivity', 'specificity', 'ppv', 'npv']:
        assert metric in summary[0.5]

def test_decision_curve_net_benefit_calculation():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.65, 0.3, 0.7, 0.1, 0.9, 0.4, 0.25, 0.6])

    dca = DecisionCurveAnalysis(y_true, y_prob)
    df = dca.compute_net_benefit()

    assert 'threshold' in df.columns
    assert 'net_benefit' in df.columns
    assert len(df) > 0
