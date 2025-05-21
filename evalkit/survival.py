import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from typing import Tuple, List, Optional, Dict
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from lifelines import KaplanMeierFitter


class LandmarkSurvivalEvaluator:
    """Evaluate a classifier using landmarking survival approach."""

    def __init__(self, model, prediction_window: float):
        """
        Args:
            model: A trained classifier with predict_proba method.
            prediction_window (float): Landmark time window for defining events.
        """
        self.model = model
        self.prediction_window = prediction_window
        self._train_X = None
        self._train_durations = None
        self._train_events = None

    def fit(self, X_train: np.ndarray, durations: np.ndarray, events: np.ndarray):
        """
        Store training survival data for dynamic AUC evaluation.

        Args:
            X_train: Feature matrix used to train the model.
            durations: Follow-up times.
            events: Event indicators (1=event, 0=censored).
        """
        self._train_X = np.asarray(X_train)
        self._train_durations = np.asarray(durations)
        self._train_events = np.asarray(events)

    def transform_labels(self, durations: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert survival data into binary labels and a mask indicating valid samples.

        Args:
            durations: Follow-up times.
            events: Event indicators.

        Returns:
            label: Binary event within prediction window.
            mask: Whether subject is still at risk at prediction window.
        """
        durations = np.asarray(durations)
        events = np.asarray(events)
        label = (durations <= self.prediction_window) & (events == 1)
        mask = durations >= self.prediction_window
        return label.astype(int), mask

    def evaluate_dynamic_auc(self, X, durations, events, times: list, n_resamples=1000, seed=1):
        """
        Computes time-dependent AUCs via cumulative_dynamic_auc with test set bootstrapping.
        """
        if self._train_X is None:
            raise ValueError("Must call .fit() with training data before evaluating dynamic AUC.")

        train_surv = Surv.from_arrays(self._train_events == 1, self._train_durations)
        test_pred = self.model.predict_proba(X)[:, 1]
        rng = np.random.default_rng(seed)

        results = {t: [] for t in times}
        successful = 0

        for _ in range(n_resamples):
            try:
                # Bootstrap resample test set
                X_b, d_b, e_b, y_b = resample(X, durations, events, test_pred, random_state=rng.integers(1e9))
                if len(np.unique(y_b)) < 2:
                    continue  # skip if predictions lack variance

                surv_b = Surv.from_arrays(e_b == 1, d_b)
                aucs, _ = cumulative_dynamic_auc(train_surv, surv_b, y_b, times=times)

                for i, t in enumerate(times):
                    results[t].append(aucs[i])
                successful += 1
            except Exception:
                for t in times:
                    results[t].append(np.nan)

        if successful == 0:
            raise RuntimeError("All bootstrap iterations failed.")

        df = pd.DataFrame.from_dict(results, orient='index')
        summary = df.apply(lambda x: pd.Series({
            "mean": np.nanmean(x),
            "lower": np.nanpercentile(x, 2.5),
            "upper": np.nanpercentile(x, 97.5)
        }), axis=1)

        summary.index.name = "time"
        return summary.reset_index()

    def evaluate_multiple_windows(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        window_list: List[float],
        times: List[float],
        n_resamples: int = 1000
    ) -> Dict[float, pd.DataFrame]:
        """
        Evaluate dynamic AUCs across multiple prediction windows.

        Args:
            X: Test feature matrix.
            durations: Test follow-up times.
            events: Event indicators.
            window_list: List of prediction windows to evaluate.
            times: Time points for AUC evaluation.
            n_resamples: Bootstrap iterations.

        Returns:
            Dictionary of {window: AUC DataFrame}.
        """
        results = {}
        for w in window_list:
            self.prediction_window = w
            results[w] = self.evaluate_dynamic_auc(X, durations, events, times, n_resamples)
        return results

    def stratified_kaplan_by_risk(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        probs: np.ndarray,
        n_bins: int = 4,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot Kaplan-Meier survival curves stratified by predicted risk.

        Args:
            durations: Follow-up times.
            events: Event indicators.
            probs: Predicted probabilities.
            n_bins: Number of risk bins.
            ax: Optional matplotlib axis.

        Returns:
            The matplotlib axis with the plot.
        """
        df = pd.DataFrame({
            'risk_score': probs,
            'duration': durations,
            'event': events
        })
        df['bin'] = pd.qcut(df['risk_score'], n_bins, labels=False)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        for b in sorted(df['bin'].unique()):
            mask = df['bin'] == b
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, 'duration'], df.loc[mask, 'event'])
            kmf.plot_survival_function(ax=ax, label=f'Q{b + 1}')

        ax.set_title("Kaplan-Meier Curves by Risk Score Quartile")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()
        ax.grid(True)
        return ax