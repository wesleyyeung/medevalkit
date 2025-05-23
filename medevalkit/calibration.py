"""Calibration metrics and plotting for classification models."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from .bootstrap import Bootstrapper

class BaseCalibration:
    
    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike, n_bins: int = 10):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.n_bins = n_bins

class BinaryCalibration(BaseCalibration):

    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike, n_bins: int = 10):
        super().__init__(y_true = y_true, y_pred_prob = y_pred_prob, n_bins = n_bins)

    def compute(self, bootstrap: bool = True, **kwargs) -> dict:
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_pred_prob, n_bins=self.n_bins)
        output = {
            'brier_score': brier_score_loss(self.y_true, self.y_pred_prob),
            'calibration_curve': (prob_pred, prob_true)
        }
        if bootstrap:
            bs = Bootstrapper(**kwargs)
            brier, brier_lower, brier_upper = bs.bootstrap_ci(brier_score_loss,self.y_true,self.y_pred_prob)
            output.update({
                'brier_score': brier,
                'brier_score_upper': brier_upper,
                'brier_score_lower': brier_lower,
                'calibration_curve': {'prob_pred':prob_pred,'prob_true':prob_true}
            })
        
        return output

class MulticlassCalibration(BaseCalibration):

    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike, n_bins: int = 10):
        super().__init__(y_true = y_true, y_pred_prob = y_pred_prob, n_bins = n_bins)
        self.n_classes = y_pred_prob.shape[1]

    def one_vs_rest_curves(self, bootstrap: bool = True, **kwargs) -> dict: 
        """
        Returns calibration curves and Brier scores for each class in a one-vs-rest fashion.
        Output: {
            class_index: {
                'prob_true': [...],
                'prob_pred': [...],
                'brier_score': float
            },
            ...
        }
        """
        curves = {}
        for k in range(self.n_classes):
            y_true_bin = (self.y_true == k).astype(int)
            prob_k = self.y_pred_prob[:, k]
            prob_true, prob_pred = calibration_curve(y_true_bin, prob_k, n_bins=self.n_bins)
            brier = brier_score_loss(y_true_bin, prob_k)
            curves[k] = {
                'brier_score': brier,
                'calibration_curve': {'prob_pred':prob_pred,'prob_true':prob_true}
            }
            if bootstrap:
                bs = Bootstrapper(**kwargs)
                brier, brier_lower, brier_upper = bs.bootstrap_ci(brier_score_loss, y_true_bin, prob_k)
                curves[k].update({
                'brier_score': brier,
                'brier_score_upper': brier_upper,
                'brier_score_lower': brier_lower
            })
        return curves

    @staticmethod
    def _expected_calibration_error(y_true: ArrayLike, y_pred_prob: ArrayLike, n_bins = 10) -> float:
        """
        Computes Expected Calibration Error (ECE) for multiclass classification.
        """
        y_pred = np.argmax(y_pred_prob, axis=1)
        confidences = np.max(y_pred_prob, axis=1)
        accuracies = (y_pred == y_true).astype(int)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                acc = np.mean(accuracies[bin_mask])
                conf = np.mean(confidences[bin_mask])
                ece += (bin_size / len(y_true)) * abs(acc - conf)
        return ece
    
    def expected_calibration_error(self,bootstrap: bool = True, **kwargs):
        output = {
            'expected_calibration_error': self._expected_calibration_error(self.y_true,self.y_pred_prob)
        }
        if bootstrap:
            bs = Bootstrapper(**kwargs)
            ece, ece_lower, ece_upper = bs.bootstrap_ci(self._expected_calibration_error,self.y_true,self.y_pred_prob)
            output.update({
                'expected_calibration_error': ece,
                'expected_calibration_error_upper': ece_upper,
                'expected_calibration_error_lower': ece_lower
            })
            
        return output