"""Simulation tools for synthetic data or perturbation experiments."""

import math
import numpy as np
from numpy.typing import ArrayLike

class PopulationSimulator:
    """
    Create a set of virtual cohorts of a prespecified incidence rate using bootstrap resampling 
    """
    def __init__(self, y: ArrayLike, n_resamples: int, random_state=None):
        try:
            self.y = y.tolist()
        except:
            self.y = y

        labels = set(np.unique(y))
        if labels != {0, 1}:
            raise ValueError("Simulator only supports binary classification with labels 0 and 1.")
        self.incidence_rate = sum(y)/len(y)
        self.n_resamples = n_resamples
        self.random_state = np.random.RandomState(random_state)

    def simulate(self, incidence_rate_list: list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]) -> tuple:
        positive_cases = [idx for idx,y_label in enumerate(self.y) if y_label == 1]
        negative_cases = [idx for idx,y_label in enumerate(self.y) if y_label == 0]
        
        simulation_dict = {}
        incidence_rate_list = sorted(list(set(incidence_rate_list + [self.incidence_rate])))

        for i in incidence_rate_list:
            resampled_pos = []
            resampled_neg = []
            for _ in range(self.n_resamples):
                #randomly resample from y such that incidence matches i
                n_pos = max(math.ceil(i * len(self.y)),1)
                n_neg = max(math.ceil((1-i) * len(self.y)),1)
                resampled_pos += [self.random_state.choice(positive_cases, n_pos, replace=True)]
                resampled_neg += [self.random_state.choice(negative_cases, n_neg, replace=True)]
            #concat arrays
            resampled_pos = np.concat(resampled_pos).tolist()
            resampled_neg = np.concat(resampled_neg).tolist()
            #combined arrays
            combined = resampled_pos + resampled_neg
            #resample to correct length - n_resamples * len(y)
            combined = self.random_state.choice(combined, self.n_resamples * len(self.y), replace=True)
            assert len(combined) == self.n_resamples * len(self.y)
            assert len(combined) 
            simulation_dict[i] = combined

        return simulation_dict