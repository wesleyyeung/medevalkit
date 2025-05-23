"""High-level functions to evaluate prediction results."""

import copy
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from .metrics import BinaryClassificationMetrics, MulticlassClassificationMetrics, RegressionMetrics
from .calibration import BinaryCalibration, MulticlassCalibration
from .simulation import PopulationSimulator

class Evaluate:
    def __init__(self, model, x: ArrayLike, y: ArrayLike, classification: bool = True, threshold: float = 0.5):
        self.model = model
        self.x = x
        self.y = y
        if classification:
            self.y_pred_prob = model.predict_proba(x)
        self.y_pred = model.predict(x)
        self.classification = classification
        self.results = None
        self.threshold = threshold

    @staticmethod
    def try_round(input,digits):
        try: 
            return round(input,digits)
        except:
            return input

    def construct_text_report(self, metrics: dict, bootstrap: bool, n_resamples: int) -> dict:
        
        for metric_category, metric_dict in metrics.items():
            #Get unique list of metrics in each metric dict with upper and lower confidence intervals
            metric_with_95ci_list = []
            for k, v in metric_dict.items():
                if '_upper' in k or '_lower' in k:
                    metric_with_95ci_list += [k.replace('_upper','').replace('_lower','')]
            metric_with_95ci_list = list(set(metric_with_95ci_list))

            _metrics = copy.deepcopy(metrics)

            #Modify each of these metrics to be of format: val (95% CI: upper - lower)
            for metric_with_95ci in metric_with_95ci_list:
                val = np.round(metrics[metric_category][metric_with_95ci],3)
                upper = np.round(metrics[metric_category][metric_with_95ci+'_upper'],3)
                lower = np.round(metrics[metric_category][metric_with_95ci+'_lower'],3)
                _metrics[metric_category][metric_with_95ci] = f"{val} (95% CI {lower}-{upper})"
                #Clean up
                _metrics[metric_category].pop(metric_with_95ci+'_upper',None)
                _metrics[metric_category].pop(metric_with_95ci+'_lower',None)

        _clf_metrics = _metrics.get('clf_metrics',{'Not Applicable':np.nan})
        _calib_metrics = _metrics.get('calib_metrics',{'Not Applicable':np.nan})
        _reg_metrics = _metrics.get('reg_metrics',{'Not Applicable':np.nan})

        report = """
==== Report ====
Bootstrap: {bootstrap}
Resamples: {n_resamples}

-- Classification Metrics --
{clf}

-- Calibration Metrics --
{calib}

-- Regression Metrics --
{reg}
                            """.format(bootstrap = bootstrap,
                                       n_resamples = n_resamples,
                                       clf="\n".join([f"{k}: {self.try_round(v,3)}" for k, v in _clf_metrics.items()]),
                                       calib="\n".join([f"{k}: {self.try_round(v,3)}" for k, v in _calib_metrics.items() if 'calibration_curve' not in k]),
                                       reg="\n".join([f"{k}: {self.try_round(v,3)}" for k, v in _reg_metrics.items()]))
        return {'text_report': report}

    def generate_report(self, indices: list = None, multiclass_method: str = 'ovr', calibration_bins: int = 10, bootstrap: bool = True, n_resamples: int = 1000, **kwargs) -> dict:
        if indices is None:
            indices = range(len(self.y))

        clf_metrics = {}
        calib_metrics = {}
        reg_metrics = {}

        classes = list(set(self.y))
        n_classes = len(classes)

        if self.classification:
            if n_classes > 2:
                clf_metrics = MulticlassClassificationMetrics(self.y[indices], self.y_pred_prob[indices,:]).compute(bootstrap = bootstrap, n_resamples = n_resamples, **kwargs)
                mcc = MulticlassCalibration(self.y[indices], self.y_pred_prob[indices], n_bins = calibration_bins)
                if multiclass_method == 'ovr':
                    ovr_calib_metrics = mcc.one_vs_rest_curves(bootstrap = bootstrap, n_resamples = n_resamples, **kwargs)
                    for class_label in ovr_calib_metrics.keys():
                        for k, v in ovr_calib_metrics[class_label].items():
                            calib_metrics[f'{class_label}_{k}'] = v
                elif multiclass_method == 'ece':
                    calib_metrics = mcc.expected_calibration_error(bootstrap = bootstrap, n_resamples = n_resamples, **kwargs)
                else:
                    raise ValueError(f"Parameter `multiclass_method` must be one of ['ovr','ece'], not {multiclass_method}!") 
            elif n_classes <= 2:
                pred_prob = self.y_pred_prob[indices, 1]
                clf_metrics = BinaryClassificationMetrics(self.y[indices], pred_prob).compute(bootstrap = bootstrap, n_resamples = n_resamples, **kwargs)
                calib_metrics = BinaryCalibration(self.y[indices], pred_prob, n_bins = calibration_bins).compute(bootstrap = bootstrap, n_resamples = n_resamples, **kwargs)
            output = {'clf_metrics':clf_metrics,'calib_metrics':calib_metrics}
            report = self.construct_text_report(output,bootstrap,n_resamples)
            output.update(report)
        else:
            reg_metrics = RegressionMetrics(self.y[indices], self.y_pred[indices]).compute()
            output = {'reg_metrics':reg_metrics}
            report = self.construct_text_report(output,bootstrap,n_resamples)
            output.update(report)

        self.results = output
        return output 
    
class EvaluateWithSimulation(Evaluate):
    def __init__(self, model, x: ArrayLike, y: ArrayLike, classification: bool = True, threshold: float = 0.5):
        super().__init__(model=model,x=x,y=y,classification=classification,threshold=threshold)
        self.results = None
    
    def run_simulation(self, incidence_rate_list: list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], 
                       calibration_bins: int = 10, bootstrap: bool = True, n_resamples: int = 1000, random_state: int = 1) -> dict:
        sim = PopulationSimulator(self.y, n_resamples, random_state)
        self.sim_dict = sim.simulate(incidence_rate_list)
        self.sim_results = {}
        for incidence, indices in self.sim_dict.items():
            metrics_list = []
            x_sim = self.x[indices,:]
            y_sim = self.y[indices]
            y_prob = self.model.predict_proba(x_sim)[:, 1]
            y_pred = (y_prob >= self.threshold).astype(int)

            metrics = self.generate_report( indices = indices, calibration_bins = calibration_bins, bootstrap = bootstrap, n_resamples = n_resamples)
            combined_metrics = {}
            for k,v in metrics.items():
                if not isinstance(v, dict):
                    v = {k: v}
                combined_metrics.update(v)
            metrics_list.append(combined_metrics)

            # Aggregate across replicates (mean and 95% CI)
            summary = {}
            for metric in metrics_list[0].keys():
                if 'upper' not in metric and 'lower' not in metric and 'text' not in metric and 'curve' not in metric:
                    values = [m[metric] for m in metrics_list]
                    try:
                        mean = np.mean(values)
                    except:
                        print(metric)
                        print(values)
                    ci_low, ci_high = np.percentile(values, [2.5, 97.5])
                    summary[metric] = f"{mean:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})"

            self.sim_results[incidence] = summary
    
    def generate_metrics(self):
        output = pd.DataFrame.from_dict(self.sim_results, orient="index").T
        output.columns = [np.round(col,2) for col in output.columns]
        return output