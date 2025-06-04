# Re-importing necessary libraries after code execution environment reset
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def compute_sens_spec_at_thresholds(results_df, pos_label=1, target_specs=[0.95], target_senss=[0.95]):
    """
    Compute sensitivity at given specificity thresholds, and vice versa.

    Parameters:
        results_df (pd.DataFrame): Contains 'pred_prob' and 'y_true'
        pos_label (int): Positive class label
        target_specs (List[float]): List of target specificity points
        target_senss (List[float]): List of target sensitivity points

    Returns:
        pd.DataFrame: Table of sensitivity at specificity and specificity at sensitivity
    """
    thresholds = np.linspace(0, 1, 1000)
    records = []

    for thresh in thresholds:
        preds = (results_df['pred_prob'] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(results_df['y_true'], preds, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        records.append((thresh, sens, spec))

    metrics_df = pd.DataFrame(records, columns=['threshold', 'sensitivity', 'specificity'])

    results = []

    # Sensitivity at given specificity
    for spec_target in target_specs:
        subset = metrics_df[metrics_df['specificity'] >= spec_target]
        if not subset.empty:
            best_row = subset.sort_values('specificity').head(1).iloc[0]
            results.append({
                'metric': f'sensitivity_at_{int(spec_target*100)}%_specificity',
                'value': best_row['sensitivity'],
                'threshold': best_row['threshold']
            })

    # Specificity at given sensitivity
    for sens_target in target_senss:
        subset = metrics_df[metrics_df['sensitivity'] >= sens_target]
        if not subset.empty:
            best_row = subset.sort_values('sensitivity').head(1).iloc[0]
            results.append({
                'metric': f'specificity_at_{int(sens_target*100)}%_sensitivity',
                'value': best_row['specificity'],
                'threshold': best_row['threshold']
            })

    return pd.DataFrame(results)

def plot_trade_off(result_df):
    """
    Plot sensitivity at specificity and specificity at sensitivity trade-offs.

    Parameters:
        result_df (pd.DataFrame): DataFrame containing 'metric', 'value', and 'threshold'
    """
    # Plotting with deep blue color
    plt.figure(figsize=(10, 6))
    bars = plt.barh(result_df['metric'], result_df['value'], color='darkblue')
    plt.xlabel('Score')
    plt.title('Sensitivity at Specificity and Specificity at Sensitivity (Updated Thresholds - Blue)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate bars with value and threshold
    for i, bar in enumerate(bars):
        score = result_df['value'][i]
        threshold = result_df['threshold'][i]
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f} @ {threshold:.2f}", va='center', color='black')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_save_path', type=str, default="DeepECG/performance_test/saved_results/1_year_ecg_meta_td_all_lead.csv")
    args = parser.parse_args()

    results_df = pd.read_csv(args.result_save_path)
    result_table = compute_sens_spec_at_thresholds(
        results_df, target_specs=[0.9, 0.95, 0.99], target_senss=[0.9, 0.95, 0.99]
    )
    plot_trade_off(result_table)

