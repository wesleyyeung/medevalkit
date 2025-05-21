# EvalKit

**EvalKit** is a modular and extensible Python toolkit for evaluating machine learning models, especially in healthcare applications. It provides unified APIs for computing metrics, calibration curves, bootstrap confidence intervals, and plotting diagnostic curves.

## Features

- Binary and multiclass classification support
- Threshold optimization
- Calibration error estimation
- Bootstrap confidence intervals
- ROC, PR, and calibration plotting
- Simulate results at different incidence rates

## Installation

```bash
pip evalkit-beta
```

## Usage Example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from evalkit.bootstrap import Bootstrapper
from evalkit.calibration import MulticlassCalibration
from evalkit.threshold import ThresholdOptimizer
from evalkit.evaluate import Evaluate
from evalkit.fairness import FairnessMetrics
from evalkit.compare import ModelComparer
from evalkit._plots import (
    plot_multiple_roc_curves_with_comparison, plot_auc_bar_chart_with_error_bars
)


from evalkit.threshold import ThresholdAnalysis, DecisionCurveAnalysis
from evalkit._plots import (
    plot_roc_curve,
    plot_calibration_curve,
    plot_threshold_metrics,
    plot_decision_curve
)

# 1. Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Train models
clf = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)
nb = BernoulliNB()

clf.fit(X_train, y_train)
rf.fit(X_train, y_train)
nb.fit(X_train, y_train)

## START HERE IF YOU HAVE A PRETRAINED MODEL ##

# 3. Threshold optimization
opt = ThresholdOptimizer(y_true=y_test, y_pred_prob=clf.predict_proba(X_test)[:, 1])
threshold = opt.optimize_youden()

# 4. Evaluate
evaluator = Evaluate(clf, X_test, y_test, classification=True, threshold=threshold)
report = evaluator.generate_report(bootstrap=True)
print(report['text_report'])

# 5. ROC and Calibration Plots
y_prob = clf.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_prob)
plot_calibration_curve(y_test, y_prob)

# 6. Threshold analysis
ta = ThresholdAnalysis(y_test, y_prob)
ta_results = ta.compute()
plot_threshold_metrics(ta_results)

# 7. Decision curve
dca = DecisionCurveAnalysis(y_test, y_prob)
dca_df = dca.compute()
plot_decision_curve(dca_df)

# 8. Fairness analysis
sex = np.array(['male' if i % 2 == 0 else 'female' for i in range(len(y_test))]) #dummy example, use real boolean indices
subgroup_indices = {
    'sex': {
        'male': np.where(sex == 'male')[0],
        'female': np.where(sex == 'female')[0]
    }
}
fair = FairnessMetrics(clf, X_test, y_test, subgroup_indices)
fair_metrics, fair_gaps = fair.compute_fairness_metrics(bootstrap=True)
print(fair_metrics)
print(fair_gaps)

# 9. Pairwise comparisons
model_dict = {"LogReg": clf, "RF": rf, "NB": nb}
comparer = ModelComparer(model_dict, X_test, y_test)

# 10. Bootstrap comparison
print("Bootstrap AUC Comparison:")
bootstrap_result = comparer.compare_auc(method='bootstrap', parametric=True, n_resamples=1000)
for k, v in bootstrap_result.items():
    print(f"{k}: AUC1={v['AUC1']:.3f}, AUC2={v['AUC2']:.3f}, p-value={v['p_value']:.5f}")

# 11. DeLong comparison
print("\nDeLong AUC Comparison:")
delong_result = comparer.compare_auc(method='delong')
for k, v in delong_result.items():
    print(f"{k}: AUC1={v['AUC1']:.3f}, AUC2={v['AUC2']:.3f}, p-value={v['p_value']:.5f}")

# 12. Plot multiple ROC curves
model_probs = {
    "LogReg": clf.predict_proba(X_test)[:, 1],
    "RF": rf.predict_proba(X_test)[:, 1],
    "NB": nb.predict_proba(X_test)[:, 1],
}
plot_multiple_roc_curves_with_comparison(
    y_true=y_test,
    model_probs=model_probs,
    method="bootstrap",
    show_pvalues_on_plot=True
)

# 13. Extract p-values for plotting
pval_dict = {}
for k, v in bootstrap_result.items():
    m1, m2 = k.split(" vs ")
    pval_dict[(m1, m2)] = v["p_value"]

# 14. Plot pairwise comparison bar chart
bs = Bootstrapper(n_resamples=1000)
auc_dict = {
    "LogReg": bs.bootstrap(metric_fn=roc_auc_score, y_true=y_test, y_pred=clf.predict_proba(X_test)[:, 1]),
    "RF": bs.bootstrap(metric_fn=roc_auc_score, y_true=y_test, y_pred=rf.predict_proba(X_test)[:, 1]),
    "NB": bs.bootstrap(metric_fn=roc_auc_score, y_true=y_test, y_pred=nb.predict_proba(X_test)[:, 1]),
}
plot_auc_bar_chart_with_error_bars(
    auc_dict=auc_dict,
    pval_dict=pval_dict,
    title="AUC Comparison with 95% Confidence Intervals and Pairwise p-values"
```

## GitHub Repo

https://github.com/wesleyyeung/evalkit

## License

This project is licensed under the MIT License.
