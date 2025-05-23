"""Tools to compare multiple models or predictions."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ttest_ind, mannwhitneyu, norm
from sklearn.metrics import roc_auc_score
from .bootstrap import Bootstrapper

def _delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    label_ordered = ground_truth[order]
    n1 = np.sum(label_ordered == 1)
    n2 = np.sum(label_ordered == 0)
    pos_scores = predictions[ground_truth == 1]
    neg_scores = predictions[ground_truth == 0]

    aucs = []
    v01 = []
    v10 = []
    for i in range(n1):
        for j in range(n2):
            if pos_scores[i] > neg_scores[j]:
                aucs.append(1)
            elif pos_scores[i] == neg_scores[j]:
                aucs.append(0.5)
            else:
                aucs.append(0)
    auc = np.mean(aucs)

    for r in range(n1):
        v01.append(np.mean([1 if pos_scores[r] > x else 0.5 if pos_scores[r] == x else 0 for x in neg_scores]))
    for r in range(n2):
        v10.append(np.mean([1 if x > neg_scores[r] else 0.5 if x == neg_scores[r] else 0 for x in pos_scores]))

    s01 = np.var(v01) / n1
    s10 = np.var(v10) / n2
    return auc, s01 + s10

def delong_test(y_true, pred1, pred2):
    auc1, var1 = _delong_roc_variance(y_true, pred1)
    auc2, var2 = _delong_roc_variance(y_true, pred2)
    se_diff = np.sqrt(var1 + var2)
    z_score = (auc1 - auc2) / se_diff
    p_value = 2 * norm.sf(abs(z_score))
    return p_value, auc1, auc2

class ModelComparer:
    def __init__(self, model_dict, x, y):
        self.model_dict = model_dict
        self.x = x
        self.y = y

    @staticmethod
    def parametric(array_a: ArrayLike, array_b: ArrayLike) -> float:
        return ttest_ind(array_a, array_b, equal_var=False).pvalue

    @staticmethod
    def nonparametric(array_a: ArrayLike, array_b: ArrayLike) -> float:
        return mannwhitneyu(array_a, array_b).pvalue

    def compare_auc(self, method: str = 'bootstrap', parametric: bool = True, n_resamples: int = None) -> dict:
    
        if parametric:
            test_func = self.parametric
        else:
            test_func = self.nonparametric

        output = {}
        model_names = list(self.model_dict.keys())

        if method == 'bootstrap':
            ##Implement pairwise t-tests using bootstrapped resamples
            aucs = {}
            bs = Bootstrapper(n_resamples=n_resamples)
            for name, model in self.model_dict.items():
                y_pred_prob = model.predict_proba(self.x)[:, 1]
                aucs[name] = bs.bootstrap(metric_fn=roc_auc_score, y_true= self.y, y_pred = y_pred_prob)

            if len(self.model_dict.keys()) < 2:
                raise ValueError('Cannot make pairwise comparisons on a single model!')
            elif len(self.model_dict.keys()) >= 2:
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                            name1, name2 = model_names[i], model_names[j]
                            auc1 = np.median(aucs[name1])
                            auc2 = np.median(aucs[name2])
                            p = test_func(aucs[name1],aucs[name2])
                            output[f"{name1} vs {name2}"] = {
                                "AUC1": auc1,
                                "AUC2": auc2,
                                "p_value": p
                            }

        elif method == 'delong':
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    pred1 = self.model_dict[name1].predict_proba(self.x)[:, 1]
                    pred2 = self.model_dict[name2].predict_proba(self.x)[:, 1]
                    p, auc1, auc2 = delong_test(self.y, pred1, pred2)
                    output[f"{name1} vs {name2}"] = {
                        "AUC1": auc1,
                        "AUC2": auc2,
                        "p_value": p
                    }
        else:
            raise ValueError(f'Input parameter method: "{method}" must be either "bootstrap", or "delong"')
        
        return output
