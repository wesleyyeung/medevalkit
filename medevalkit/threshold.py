"""Threshold optimization strategies for classification models."""

from typing import Dict
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.preprocessing import label_binarize
from .metrics import BinaryClassificationMetrics

class ThresholdAnalysis:
    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob

    def compute(self, step: float = 0.01) -> dict:
        output_dict = {}
        threshold = 1.0
        while threshold >= 0:
            cm = BinaryClassificationMetrics(y_true=self.y_true, y_pred_prob=self.y_pred_prob, threshold=threshold)
            output_dict[threshold] = cm.compute_static(bootstrap=False) if hasattr(cm, 'compute_static') else cm.compute()
            threshold -= step
        return output_dict

class DecisionCurveAnalysis:
    def __init__(self, y_true: ArrayLike, y_prob: ArrayLike, thresholds: ArrayLike = np.linspace(0.01, 0.99, 99)):
        self.y_true = y_true
        self.y_prob = y_prob
        self.thresholds = thresholds

    def compute(self) -> pd.DataFrame:
        results = []
        n = len(self.y_true)
        for thresh in self.thresholds:
            tp = np.sum((self.y_prob >= thresh) & (self.y_true == 1))
            fp = np.sum((self.y_prob >= thresh) & (self.y_true == 0))
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            results.append({'threshold': thresh, 'net_benefit': net_benefit})
        return pd.DataFrame(results)

class ThresholdOptimizer:
    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob

    def optimize_youden(self) -> float:
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        j_scores = tpr - fpr
        return thresholds[np.argmax(j_scores)]

    def optimize_f1(self) -> float:
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1_scores)]

    def optimize_closest_to_perfect(self) -> float:
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        return thresholds[np.argmin(dist)]

    def optimize_fixed_sensitivity(self, target_sens=0.9) -> float:
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        idx = np.where(tpr >= target_sens)[0]
        return thresholds[idx[0]] if len(idx) > 0 else thresholds[-1]

    def optimize_decision_curve(self) -> float:
        dca = DecisionCurveAnalysis(self.y_true, self.y_pred_prob)
        dca_df = dca.compute()
        return dca_df.loc[dca_df['net_benefit'].idxmax(), 'threshold']

class MulticlassThresholdOptimizer:
    """Threshold optimization for multiclass classifiers using one-vs-rest approach.

    Each class is treated as a separate binary classification problem, and
    thresholds are optimized independently using specified strategy.

    Attributes:
        y_true (np.ndarray): True labels (integer-encoded).
        y_pred_prob (np.ndarray): Predicted probabilities, shape (n_samples, n_classes).
        classes (np.ndarray): Unique class labels.
        thresholds_ (dict): Mapping from class label to optimal threshold.
    """

    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike):
        """Initializes the optimizer.

        Args:
            y_true (ArrayLike): True labels (integers, shape (n_samples,)).
            y_pred_prob (ArrayLike): Predicted probabilities (shape (n_samples, n_classes)).
        """
        self.y_true = np.asarray(y_true)
        self.y_pred_prob = np.asarray(y_pred_prob)
        self.classes = np.unique(self.y_true)
        self.n_classes = len(self.classes)
        self.thresholds_: Dict[int, float] = {}

        assert self.y_pred_prob.shape[1] == self.n_classes, \
            "Shape mismatch: y_pred_prob must have shape (n_samples, n_classes)"

    def optimize_per_class(self, method: str = "youden", **kwargs) -> Dict[int, float]:
        """Optimizes thresholds for each class using a specified method.

        Args:
            method (str): Optimization strategy. One of:
                - "youden"
                - "f1"
                - "closest_to_perfect"
                - "fixed_sensitivity"
                - "decision_curve"
            **kwargs: Additional keyword arguments for the optimization method.

        Returns:
            Dict[int, float]: Dictionary mapping class label to threshold.
        """
        self.thresholds_ = {}

        for class_index, class_label in enumerate(self.classes):
            # One-vs-rest binarization
            y_bin = (self.y_true == class_label).astype(int)
            y_prob = self.y_pred_prob[:, class_index]

            # Use the binary optimizer
            opt = ThresholdOptimizer(y_bin, y_prob)

            if method == "youden":
                threshold = opt.optimize_youden()
            elif method == "f1":
                threshold = opt.optimize_f1()
            elif method == "closest_to_perfect":
                threshold = opt.optimize_closest_to_perfect()
            elif method == "fixed_sensitivity":
                threshold = opt.optimize_fixed_sensitivity(**kwargs)
            elif method == "decision_curve":
                threshold = opt.optimize_decision_curve()
            else:
                raise ValueError(f"Unknown method: {method}")

            self.thresholds_[class_label] = threshold

        return self.thresholds_

    def predict(self, y_pred_prob: ArrayLike = None) -> np.ndarray:
        """Generates multiclass predictions using optimized thresholds.

        Args:
            y_pred_prob (ArrayLike, optional): Predicted probabilities to use.
                If None, will use the instance's stored probabilities.

        Returns:
            np.ndarray: Array of predicted class labels.
        """
        if not self.thresholds_:
            raise RuntimeError("Thresholds not computed. Call optimize_per_class() first.")

        y_pred_prob = self.y_pred_prob if y_pred_prob is None else np.asarray(y_pred_prob)
        decisions = np.full(y_pred_prob.shape[0], fill_value=-1, dtype=int)

        for i in range(y_pred_prob.shape[0]):
            passed = [
                cls for cls in self.classes
                if y_pred_prob[i, cls] >= self.thresholds_[cls]
            ]
            decisions[i] = passed[0] if passed else np.argmax(y_pred_prob[i])

        return decisions