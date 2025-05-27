"""Metric utilities for classification and regression tasks."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, f1_score,
    mean_squared_error, mean_absolute_error, precision_score, recall_score
)
from .bootstrap import Bootstrapper

class BaseClassificationMetrics:
    def __init__(self, y_true: ArrayLike, y_pred_prob: ArrayLike, threshold: float = 0.5):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.threshold = threshold

    def compute(self):
        raise NotImplementedError("Must implement in subclass")

class BinaryClassificationMetrics(BaseClassificationMetrics):
    def __init__(self, y_true, y_pred_prob, threshold=0.5):
        super().__init__(y_true, y_pred_prob, threshold)
        self.y_pred = (y_pred_prob >= threshold).astype(int)

    @staticmethod
    def sensitivity(y_true,y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    
    @staticmethod
    def specificity(y_true,y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    
    @staticmethod
    def ppv(y_true,y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def npv(y_true,y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0

    @staticmethod
    def accuracy(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (tp + tn) / (tp + tn + fp + fn)

    def compute(self, bootstrap: bool = True, n_resamples: int = 1000) -> dict:
        if bootstrap:
            bs = Bootstrapper(n_resamples=n_resamples)
            auc, auc_lower, auc_upper = bs.bootstrap_ci(roc_auc_score,y_true=self.y_true,y_pred=self.y_pred_prob)
            auprc, auprc_lower, auprc_upper =  bs.bootstrap_ci(average_precision_score,y_true=self.y_true,y_pred=self.y_pred_prob)
            f1, f1_lower, f1_upper = bs.bootstrap_ci(f1_score,y_true=self.y_true,y_pred=self.y_pred)
            sens, sens_lower, sens_upper = bs.bootstrap_ci(self.sensitivity,y_true=self.y_true,y_pred=self.y_pred)
            spec, spec_lower, spec_upper = bs.bootstrap_ci(self.specificity,y_true=self.y_true,y_pred=self.y_pred)
            ppv, ppv_lower, ppv_upper = bs.bootstrap_ci(self.ppv,y_true=self.y_true,y_pred=self.y_pred)
            npv, npv_lower,  npv_upper = bs.bootstrap_ci(self.npv,y_true=self.y_true,y_pred=self.y_pred)
            acc, acc_lower, acc_upper = bs.bootstrap_ci(self.accuracy,y_true=self.y_true,y_pred=self.y_pred)
            return {
                'auc': auc,
                'auc_upper': auc_upper,
                'auc_lower': auc_lower,
                'auprc': auprc,
                'auprc_upper': auprc_upper, 
                'auprc_lower': auprc_lower,
                'f1': f1,
                'f1_upper': f1_upper,
                'f1_lower': f1_lower,
                'sensitivity': sens,
                'sensitivity_upper': sens_upper,
                'sensitivity_lower': sens_lower,
                'specificity': spec,
                'specificity_upper': spec_upper,
                'specificity_lower': spec_lower,
                'ppv': ppv,
                'ppv_upper': ppv_upper,
                'ppv_lower': ppv_lower,
                'npv': npv,
                'npv_upper': npv_upper,
                'npv_lower': npv_lower,
                'accuracy': acc,
                'accuracy_upper': acc_upper,
                'accuracy_lower': acc_lower
            }
        else:
            tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
            return {
                'auc': roc_auc_score(self.y_true, self.y_pred_prob),
                'auprc': average_precision_score(self.y_true, self.y_pred_prob),
                'f1': f1_score(self.y_true, self.y_pred),
                'sensitivity': tp / (tp + fn),
                'specificity': tn / (tn + fp),
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn)
            }
        
    def compute_static(self, bootstrap: bool = True, n_resamples: int = 1000) -> dict:
            if bootstrap:
                bs = Bootstrapper(n_resamples)
                f1, f1_lower, f1_upper = bs.bootstrap_ci(f1_score,y_true=self.y_true,y_pred=self.y_pred)
                sens, sens_lower, sens_upper = bs.bootstrap_ci(self.sensitivity,y_true=self.y_true,y_pred=self.y_pred)
                spec, spec_lower, spec_upper = bs.bootstrap_ci(self.specificity,y_true=self.y_true,y_pred=self.y_pred)
                ppv, ppv_lower, ppv_upper = bs.bootstrap_ci(self.ppv,y_true=self.y_true,y_pred=self.y_pred)
                npv, npv_lower,  npv_upper = bs.bootstrap_ci(self.npv,y_true=self.y_true,y_pred=self.y_pred)
                acc, acc_lower, acc_upper = bs.bootstrap_ci(self.accuracy,y_true=self.y_true,y_pred=self.y_pred)
                return {
                    'f1': f1,
                    'f1_upper': f1_upper,
                    'f1_lower': f1_lower,
                    'sensitivity': sens,
                    'sensitivity_upper': sens_upper,
                    'sensitivity_lower': sens_lower,
                    'specificity': spec,
                    'specificity_upper': spec_upper,
                    'specificity_lower': spec_lower,
                    'ppv': ppv,
                    'ppv_upper': ppv_upper,
                    'ppv_lower': ppv_lower,
                    'npv': npv,
                    'npv_upper': npv_upper,
                    'npv_lower': npv_lower,
                    'accuracy': acc,
                    'accuracy_upper': acc_upper,
                    'accuracy_lower': acc_lower
                }
            else:     
                tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
                return {
                    'f1': f1_score(self.y_true,self.y_pred),
                    'sensitivity': tp / (tp + fn),
                    'specificity': tn / (tn + fp),
                    'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                    'accuracy': (tp + tn) / (tp + tn + fp + fn)
                }

class MulticlassClassificationMetrics(BaseClassificationMetrics):
    def __init__(self, y_true, y_pred_prob, average: str = 'macro'):
        super().__init__(y_true, y_pred_prob)
        self.y_pred = np.argmax(y_pred_prob, axis=1)
        self.average = average

    def compute(self, bootstrap: bool = False, n_resamples: int = 1000) -> dict:
        output = {
            f'auc ({self.average})': roc_auc_score(self.y_true, self.y_pred_prob, multi_class='ovr', average=self.average),
            f'precision ({self.average})': precision_score(self.y_true, self.y_pred, average=self.average),
            f'recall ({self.average})': recall_score(self.y_true, self.y_pred, average=self.average)
        }
        if bootstrap:
            bs = Bootstrapper(n_resamples=n_resamples)
            auc, auc_lower, auc_upper = bs.bootstrap_ci(roc_auc_score,self.y_true, self.y_pred_prob,multi_class='ovr', average=self.average)
            f1, f1_lower, f1_upper = bs.bootstrap_ci(f1_score,y_true=self.y_true,y_pred=self.y_pred,average=self.average)
            precision, precision_lower, precision_upper = bs.bootstrap_ci(precision_score, self.y_true, self.y_pred, average=self.average)
            recall, recall_lower, recall_upper = bs.bootstrap_ci(recall_score, self.y_true, self.y_pred, average=self.average)
            output.update({
                f'auc ({self.average})': auc,
                f'auc ({self.average})_upper': auc_upper,
                f'auc ({self.average})_lower': auc_lower,
                f'f1 ({self.average})': f1,
                f'f1 ({self.average})_upper': f1_upper,
                f'f1 ({self.average})_lower': f1_lower,
                f'precision ({self.average})': precision,
                f'precision ({self.average})_upper': precision_upper,
                f'precision ({self.average})_lower': precision_lower,
                f'recall ({self.average})': recall,
                f'recall ({self.average})_upper': recall_upper,
                f'recall ({self.average})_lower': recall_lower
            })
        return output

class RegressionMetrics:
 
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def compute(self, bootstrap: bool = True, **kwargs) -> dict: 
        if bootstrap:
            bs = Bootstrapper(**kwargs)
            mse, mse_lower, mse_upper = bs.bootstrap_ci(mean_squared_error,self.y_true,self.y_pred)
            mae, mae_lower, mae_upper = bs.bootstrap_ci(mean_absolute_error,self.y_true,self.y_pred)
            return {
                'mse': mse,
                'mse_upper': mse_upper,
                'mse_lower': mse_lower,
                'mae': mae,
                'mae_upper': mae_upper,
                'mae_lower': mae_lower
            }
        else:
            return {
                'mse': mean_squared_error(self.y_true, self.y_pred),
                'mae': mean_absolute_error(self.y_true, self.y_pred),
            }
