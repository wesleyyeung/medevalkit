"""Bootstrap resampling utilities for confidence interval estimation."""

import types
import numpy as np
from numpy.typing import ArrayLike

class Bootstrapper:

    def __init__(self, n_resamples: int = 1000, random_state : int = 1):
        self.n_resamples = n_resamples
        self.random_state = np.random.RandomState(random_state)
    
    def bootstrap(self, metric_fn: types.FunctionType, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> list:
        bootstrapped_scores = []
        n_samples = len(y_true)
        while len(bootstrapped_scores) <= self.n_resamples:
            indices = self.random_state.choice(n_samples, n_samples, replace=True)
            try:
                score = metric_fn(y_true[indices], y_pred[indices], **kwargs)
                bootstrapped_scores.append(score)
            except:
                continue
                
        return bootstrapped_scores

    def bootstrap_ci(self, metric_fn: types.FunctionType, y_true: ArrayLike, y_pred: ArrayLike, alpha: float = 0.05, **kwargs) -> tuple:
        bootstrapped_scores = self.bootstrap(metric_fn = metric_fn, y_true = y_true, y_pred = y_pred, **kwargs)
        lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
        upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
        return np.mean(bootstrapped_scores), lower, upper
