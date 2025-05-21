from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .metrics import BinaryClassificationMetrics, MulticlassClassificationMetrics


class FairnessMetrics:
    def __init__(
        self,
        clf,
        x: ArrayLike,
        y: ArrayLike,
        subgroup_indices: dict,
        threshold: float = 0.5,
        average: str = "macro",
    ):
        self.clf = clf
        self.x = x
        self.y_true = y
        self.y_pred = clf.predict(x)
        self.y_prob = clf.predict_proba(x) if hasattr(clf, "predict_proba") else None
        self.subgroup_indices = subgroup_indices
        self.threshold = threshold
        self.average = average
        self.n_classes = len(np.unique(y))

    def compute_fairness_metrics(self, bootstrap: bool = False, n_resamples: int = 1000):
        results = []
        gap_dict = {}

        # Decide metric class
        multiclass = self.n_classes > 2

        for feature, groups in self.subgroup_indices.items():
            metric_collect = {"TPR": [], "PPV": [], "FPR": []}  # for gap calculation

            for group_name, indices in groups.items():
                if len(indices) == 0:
                    continue

                y_g = self.y_true[indices]
                y_prob_g = self.y_prob[indices]

                if multiclass:
                    metric_obj = MulticlassClassificationMetrics(
                        y_g, y_prob_g, average=self.average
                    )
                    res = metric_obj.compute(bootstrap=bootstrap, n_resamples=n_resamples)

                    results.append({
                        "feature": feature,
                        "group": group_name,
                        "n": len(indices),
                        "TPR": res.get(f"recall ({self.average})"),
                        "TPR_lower": res.get(f"recall ({self.average})_lower"),
                        "TPR_upper": res.get(f"recall ({self.average})_upper"),
                        "PPV": res.get(f"precision ({self.average})"),
                        "PPV_lower": res.get(f"precision ({self.average})_lower"),
                        "PPV_upper": res.get(f"precision ({self.average})_upper"),
                        "FPR": np.nan  # Optional: not straightforward to define FPR for multiclass
                    })

                    metric_collect["TPR"].append(res.get(f"recall ({self.average})"))
                    metric_collect["PPV"].append(res.get(f"precision ({self.average})"))

                else:
                    y_prob_g_bin = y_prob_g[:, 1] if y_prob_g.ndim > 1 else y_prob_g
                    metric_obj = BinaryClassificationMetrics(
                        y_true=y_g,
                        y_pred_prob=y_prob_g_bin,
                        threshold=self.threshold
                    )
                    res = metric_obj.compute(bootstrap=bootstrap, n_resamples=n_resamples)

                    tpr = res["sensitivity"]
                    tpr_lower = res.get("sensitivity_lower")
                    tpr_upper = res.get("sensitivity_upper")
                    fpr = 1 - res["specificity"]
                    fpr_lower = 1 - res.get("specificity_upper") if "specificity_upper" in res else None
                    fpr_upper = 1 - res.get("specificity_lower") if "specificity_lower" in res else None
                    ppv = res["ppv"]
                    ppv_lower = res.get("ppv_lower")
                    ppv_upper = res.get("ppv_upper")

                    results.append({
                        "feature": feature,
                        "group": group_name,
                        "n": len(indices),
                        "TPR": tpr,
                        "TPR_lower": tpr_lower,
                        "TPR_upper": tpr_upper,
                        "FPR": fpr,
                        "FPR_lower": fpr_lower,
                        "FPR_upper": fpr_upper,
                        "PPV": ppv,
                        "PPV_lower": ppv_lower,
                        "PPV_upper": ppv_upper
                    })

                    metric_collect["TPR"].append(tpr)
                    metric_collect["FPR"].append(fpr)
                    metric_collect["PPV"].append(ppv)

            gap_dict[feature] = {
                "TPR_gap": np.nanmax(metric_collect["TPR"]) - np.nanmin(metric_collect["TPR"]),
                "PPV_gap": np.nanmax(metric_collect["PPV"]) - np.nanmin(metric_collect["PPV"]),
                "FPR_gap": (
                    np.nanmax(metric_collect["FPR"]) - np.nanmin(metric_collect["FPR"])
                    if not multiclass else np.nan
                ),
            }

        metrics_df = pd.DataFrame(results)
        gap_df = pd.DataFrame([
            {"feature": feature, **gaps} for feature, gaps in gap_dict.items()
        ])

        self.metrics_df = metrics_df
        self.gap_df = gap_df

        return metrics_df, gap_df