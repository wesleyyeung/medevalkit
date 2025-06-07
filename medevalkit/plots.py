from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from .compare import ModelComparer, delong_test
from .bootstrap import Bootstrapper

# Set custom style sheet for all plots
plt.style.use('medevalkit/custom.mplstyle')

def plot_roc_curve(y_true, y_prob, label='Model', save_path=None):
    """
    Plots a Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        label (str): Label for the model in the legend.
        save_path (str, optional): Path to save the plot (e.g., 'roc_curve.png').
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7)) # Increased figure size
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box') # Ensure 1:1 aspect ratio
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiclass_roc(y_true, y_score, class_labels=None, save_path=None):
    """
    Plot ROC curves for multiclass classification using One-vs-Rest.

    Args:
        y_true (array-like): True class labels (n_samples,).
        y_score (array-like): Predicted probabilities (n_samples, n_classes).
        class_labels (list, optional): List of class names.
        save_path (str, optional): Path to save the plot.
    """
    n_classes = y_score.shape[1]
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(n_classes)]

    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    plt.figure(figsize=(9, 8)) # Larger figure for multiple curves
    colors = sns.color_palette("deep", n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multiclass ROC Curve (One-vs-Rest)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_calibration_curve(y_true, y_prob, n_bins=10, label='Model', save_path=None):
    """
    Plots a calibration curve (reliability diagram).

    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to use for calibration.
        label (str): Label for the model in the legend.
        save_path (str, optional): Path to save the plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label=label, color='darkgreen')
    
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives (True Probability)', fontsize=14)
    plt.title('Calibration Curve', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiclass_calibration(calibration_obj, class_labels=None, save_path=None):
    """
    Plots one-vs-rest calibration curves from a MulticlassCalibration instance.

    Args:
        calibration_obj: An instance of MulticlassCalibration (assumed to have one_vs_rest_curves method).
        class_labels (list, optional): Optional list of class names.
        save_path (str, optional): Path to save the plot.
    """
    curves = calibration_obj.one_vs_rest_curves()
    n_classes = len(curves)
    
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(n_classes)]

    plt.figure(figsize=(9, 8))
    colors = sns.color_palette("deep", n_classes)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated', alpha=0.7)

    for k, result in curves.items():
        prob_pred = result["calibration_curve"]["prob_pred"]
        prob_true = result["calibration_curve"]["prob_true"]
        brier = result["brier_score"]
        label = f"{class_labels[k]} (Brier: {brier:.3f})" # Format Brier score
        plt.plot(prob_pred, prob_true, marker="o", markersize=6, lw=2, label=label, color=colors[k])

    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives", fontsize=14)
    plt.title("Multiclass Calibration Curves (One-vs-Rest)", fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12) # External legend
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout() # Adjust for external legend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_curve(dca_df, label='Model', save_path=None):
    """
    Plots a Decision Curve Analysis (DCA) curve.

    Args:
        dca_df (pd.DataFrame): DataFrame with 'threshold' and 'net_benefit' columns.
        label (str): Label for the model in the legend.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(dca_df['threshold'], dca_df['net_benefit'], label=label, lw=2, color='purple')
    plt.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.8, label='Treat All/Treat None')
    
    # Optional: Plot 'treat all' and 'treat none' if available in dca_df or calculable
    if 'net_benefit_all' in dca_df.columns:
        plt.plot(dca_df['threshold'], dca_df['net_benefit_all'], linestyle=':', color='red', lw=1.5, label='Treat All')
    if 'net_benefit_none' in dca_df.columns:
        plt.plot(dca_df['threshold'], dca_df['net_benefit_none'], linestyle=':', color='blue', lw=1.5, label='Treat None')

    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title('Decision Curve Analysis', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([0, 1])
    # Adjust y-axis limits to better fit data and include 0
    min_nb = dca_df['net_benefit'].min()
    max_nb = dca_df['net_benefit'].max()
    plt.ylim(min(0, min_nb - 0.05), max(0.1, max_nb + 0.05)) # Extend limits slightly
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_threshold_metrics(threshold_dict, metrics=('sensitivity', 'specificity', 'ppv', 'npv'), save_path=None):
    """
    Plots various classification metrics across different probability thresholds.

    Args:
        threshold_dict (dict): Dictionary with thresholds as keys and a dict of metrics as values.
        metrics (tuple): Metrics to plot (e.g., 'sensitivity', 'specificity').
        save_path (str, optional): Path to save the plot.
    """
    df = pd.DataFrame.from_dict(threshold_dict, orient='index').sort_index(ascending=True)
    
    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(metrics))
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            plt.plot(df.index, df[metric], label=metric.replace('_', ' ').title(), lw=2, color=colors[i])
            
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title('Threshold Analysis Metrics', fontsize=16)
    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=12) # External legend
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout() # Adjust for external legend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_regression_diagnostics(y_true, y_pred, save_path=None):
    """
    Plots regression diagnostics:
    - Predicted vs Actual values
    - Residuals vs Predicted values

    Args:
        y_true (array-like): True continuous targets.
        y_pred (array-like): Predicted continuous outputs.
        save_path (str, optional): Path to save the plot.
    """
    residuals = y_true - y_pred

    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # Larger figure

    # Predicted vs Actual
    axs[0].scatter(y_true, y_pred, alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    axs[0].set_xlabel('Actual Values', fontsize=13)
    axs[0].set_ylabel('Predicted Values', fontsize=13)
    axs[0].set_title('Predicted vs Actual Values', fontsize=15)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].legend(loc='upper left', fontsize=11)


    # Residuals vs Predicted
    axs[1].scatter(y_pred, residuals, alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
    axs[1].axhline(0, linestyle='--', color='gray', lw=2, label='Zero Residuals')
    axs[1].set_xlabel('Predicted Values', fontsize=13)
    axs[1].set_ylabel('Residuals', fontsize=13)
    axs[1].set_title('Residuals vs Predicted Values', fontsize=15)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].legend(loc='upper right', fontsize=11)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiple_roc_curves_with_comparison(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    method: str = "delong",
    parametric: bool = True,
    n_resamples: int = 1000,
    return_stats: bool = False,
    show_pvalues_on_plot: bool = True,
    save_path: str = None
):
    """
    Plot multiple ROC curves and optionally annotate pairwise p-values.

    Args:
        y_true (np.ndarray): Ground truth binary labels (1D array)
        model_probs (Dict[str, np.ndarray]): Dict of model name -> predicted probabilities (1D array)
        method (str): 'delong' or 'bootstrap' for p-value calculation.
        parametric (bool): If using bootstrap, use parametric t-test (True) or Mann-Whitney U test (False).
        n_resamples (int): Number of bootstrap resamples.
        return_stats (bool): If True, return a DataFrame of p-values and AUCs.
        show_pvalues_on_plot (bool): If True, add pairwise p-values as text annotations on the plot.
        save_path (str, optional): Path to save the plot.
    """
    model_names = list(model_probs.keys())
    aucs = {}
    stats = []

    plt.figure(figsize=(9, 8)) # Larger figure
    colors = sns.color_palette("deep", len(model_names))

    for i, name in enumerate(model_names):
        y_prob = model_probs[name]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        aucs[name] = auc_val
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {auc_val:.3f})', color=colors[i])

    # Plot settings
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, alpha=0.7, label='Random Guess')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Comparison of ROC Curves", fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box')

    # Compute pairwise comparisons
    if len(model_names) >= 2:
        text_y_offset = 0.04
        current_y_pos = 0.95 # Starting y position for p-values
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                
                # Check for mock functions and adjust calls
                if method == "delong":
                    if 'mock_delong_test' in globals() and delong_test.__name__ == 'mock_delong_test':
                        # Call mock function with appropriate arguments
                        pval, auc1, auc2 = delong_test(y_true, model_probs[name1], model_probs[name2])
                    else:
                        pval, auc1, auc2 = delong_test(y_true, model_probs[name1], model_probs[name2])
                elif method == "bootstrap":
                    bs = Bootstrapper(n_resamples=n_resamples)
                    auc_dist1 = bs.bootstrap(metric_fn=roc_auc_score, y_true=y_true, y_pred=model_probs[name1])
                    auc_dist2 = bs.bootstrap(metric_fn=roc_auc_score, y_true=y_true, y_pred=model_probs[name2])
                    
                    test_func = (
                        ModelComparer.parametric if parametric else ModelComparer.nonparametric
                    )
                    pval = test_func(auc_dist1, auc_dist2)
                    auc1, auc2 = np.median(auc_dist1), np.median(auc_dist2)
                else:
                    raise ValueError("method must be 'delong' or 'bootstrap'")

                stats.append({
                    "Model 1": name1,
                    "Model 2": name2,
                    "AUC 1": auc1,
                    "AUC 2": auc2,
                    "p-value": pval
                })

                if show_pvalues_on_plot:
                    if pval < 0.001:
                        pval_str = 'p < 0.001'
                    else:
                        pval_str = f'p = {pval:.3f}'
                    
                    plt.text(
                        0.02, current_y_pos,
                        f"Comparison {name1} vs {name2}: {pval_str}",
                        transform=plt.gca().transAxes, # Use axes coordinates
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=0.5, alpha=0.8) # Add bbox for clarity
                    )
                    current_y_pos -= text_y_offset # Move down for next p-value


    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    if return_stats:
        return pd.DataFrame(stats)

def plot_auc_comparison_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = "p-value",
    cmap: str = "RdYlBu_r", # Diverging colormap
    annot: bool = True,
    fmt: str = ".3f",
    figsize: tuple = None,
    title: str = None,
    save_path: str = None
):
    """
    Plot a symmetric heatmap from a pairwise AUC comparison dataframe.

    Args:
        comparison_df (pd.DataFrame): Output DataFrame with columns: ["Model 1", "Model 2", metric]
        metric (str): Column name to visualize (e.g., "p-value", "AUC 1", "AUC 2").
        cmap (str): Colormap for the heatmap.
        annot (bool): Annotate heatmap cells.
        fmt (str): Format for annotation values.
        figsize (tuple, optional): Optional custom figure size.
        title (str, optional): Optional title (default: auto-generated).
        save_path (str, optional): Path to save the plot.
    """
    if not {"Model 1", "Model 2", metric}.issubset(comparison_df.columns):
        raise ValueError(f"Input DataFrame must contain 'Model 1', 'Model 2', and '{metric}' columns.")

    models = sorted(set(comparison_df["Model 1"]).union(set(comparison_df["Model 2"])))
    matrix = pd.DataFrame(index=models, columns=models, dtype=object) # Use object to store formatted strings

    for _, row in comparison_df.iterrows():
        m1, m2 = row["Model 1"], row["Model 2"]
        value = row[metric]
        
        display_value = ""
        if metric == 'p-value':
            if value < 0.001:
                display_value = '< 0.001'
            else:
                display_value = f'{value:.3f}'
        else:
            display_value = f'{value:{fmt}}' # Apply general format
        
        matrix.loc[m1, m2] = display_value
        matrix.loc[m2, m1] = display_value

    # Fill diagonal with empty string or model name if desired
    for model in models:
        matrix.loc[model, model] = "" # Or model name, e.g., f"AUC: {aucs.get(model, 'N/A'):.3f}" if AUCs are available

    if figsize is None:
        n = len(models)
        figsize = (max(8, n * 1.2), max(7, n)) # Adjusted size for readability

    if title is None:
        title = f"Pairwise {metric} Comparison between Models"

    plt.figure(figsize=figsize)
    # Convert matrix to numeric for cmap if metric is numeric, else cmap might not apply well
    # For p-values, we might want a custom colormap that highlights significance
    numeric_matrix = matrix.apply(pd.to_numeric, errors='coerce') # For cmap
    
    sns.heatmap(numeric_matrix, annot=matrix, fmt="", cmap=cmap, square=True,
                linewidths=0.5, linecolor='black', cbar_kws={"label": metric.replace('_', ' ').title()})
    plt.title(title, fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_auc_bar_chart_with_error_bars(
    auc_dict: Dict[str, np.ndarray],
    pval_df: pd.DataFrame = None, # Changed to DataFrame for more flexible p-value display
    ci: float = 0.95,
    title: str = "AUC Comparison with Confidence Intervals",
    figsize: tuple = (10, 7), # Increased figure size
    annotate: bool = True,
    ylim: tuple = None,
    save_path: str = None
):
    """
    Plots a bar chart of AUCs with confidence intervals and optional p-value annotations.

    Args:
        auc_dict (Dict[str, np.ndarray]): Dictionary of model name -> bootstrap distribution of AUCs.
        pval_df (pd.DataFrame, optional): DataFrame with pairwise p-values (from plot_multiple_roc_curves_with_comparison).
                                        Expected columns: "Model 1", "Model 2", "p-value".
        ci (float): Confidence interval level (e.g., 0.95 for 95% CI).
        title (str): Plot title.
        figsize (tuple): Figure size.
        annotate (bool): If True, annotate bars with AUC and CI.
        ylim (tuple, optional): Custom y-axis limits.
        save_path (str, optional): Path to save the plot.
    """
    model_names = list(auc_dict.keys())
    auc_values = [np.median(auc_dict[m]) for m in model_names]
    
    alpha_ci = (1 - ci) / 2 * 100
    lower_bounds = [np.percentile(auc_dict[m], alpha_ci) for m in model_names]
    upper_bounds = [np.percentile(auc_dict[m], 100 - alpha_ci) for m in model_names]
    
    error_bars = np.array([
        [median - lower, upper - median]
        for median, lower, upper in zip(auc_values, lower_bounds, upper_bounds)
    ]).T

    plt.figure(figsize=figsize)
    bars = plt.bar(
        model_names,
        auc_values,
        yerr=error_bars,
        capsize=8, # Larger caps for error bars
        alpha=0.9,
        color=sns.color_palette("viridis", len(model_names)), # Use viridis for bars
        edgecolor="black",
        lw=1.5 # Thicker bar edges
    )

    # Annotate each bar with AUC and CI
    if annotate:
        for i, bar in enumerate(bars):
            auc = auc_values[i]
            lb = lower_bounds[i]
            ub = upper_bounds[i]
            label = f"{auc:.3f}\n[{lb:.3f}, {ub:.3f}]"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                ub + 0.015, # Position slightly above upper CI
                label,
                ha='center',
                va='bottom',
                fontsize=10,
                color='darkblue'
            )

    # Calculate max Y for positioning brackets
    max_y_for_brackets = max(upper_bounds) + 0.02
    
    if pval_df is not None and not pval_df.empty:
        # Sort p-values to ensure consistent bracket drawing, e.g., smallest p-value first
        pval_df = pval_df.sort_values(by="p-value").reset_index(drop=True)
        
        # Track occupied y-levels for brackets to avoid overlap
        y_level_tracker = {} # Key: model name, Value: highest y-level used by its brackets

        for i, row in pval_df.iterrows():
            m1, m2, pval = row["Model 1"], row["Model 2"], row["p-value"]
            
            idx1 = model_names.index(m1)
            idx2 = model_names.index(m2)
            
            x1 = bars[idx1].get_x() + bars[idx1].get_width() / 2
            x2 = bars[idx2].get_x() + bars[idx2].get_width() / 2
            
            # Determine starting y for this bracket based on involved bars' heights
            current_bracket_y = max(upper_bounds[idx1], upper_bounds[idx2]) + 0.015
            
            # Adjust y-level to avoid collision with other brackets
            if m1 in y_level_tracker:
                current_bracket_y = max(current_bracket_y, y_level_tracker[m1] + 0.015)
            if m2 in y_level_tracker:
                current_bracket_y = max(current_bracket_y, y_level_tracker[m2] + 0.015)
            
            # Update tracker
            y_level_tracker[m1] = current_bracket_y
            y_level_tracker[m2] = current_bracket_y

            # Adjust max_y_for_brackets
            max_y_for_brackets = max(max_y_for_brackets, current_bracket_y + 0.02) # Add space for p-value text

            h = 0.005 # Height of the vertical line of the bracket
            
            if pval < 0.001:
                pval_str = 'p < 0.001'
            else:
                pval_str = f'p = {pval:.3f}'
            
            # Draw the bracket
            plt.plot([x1, x1, x2, x2], 
                     [current_bracket_y, current_bracket_y + h, current_bracket_y + h, current_bracket_y], 
                     lw=1.5, color='black')
            
            # Place the p-value text
            plt.text((x1 + x2) / 2, current_bracket_y + h + 0.003, pval_str,
                     ha='center', va='bottom', fontsize=11, color='darkred')


    # Set Y-axis limits
    if ylim:
        plt.ylim(*ylim)
    else:
        # Ensure there's enough space for bars, CIs, and potential p-value annotations
        min_y = min(lower_bounds) - 0.05
        max_y = max(max_y_for_brackets, max(upper_bounds) + 0.05)
        plt.ylim(min_y, max_y)

    plt.ylabel("AUC", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=15, ha='right') # Rotate x-axis labels for better readability
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_kaplan_meier(durations, events, ax=None, label=None, save_path=None, **kwargs):
    """
    Plot Kaplan-Meier survival curve.

    Args:
        durations (array-like): Time to event or censoring.
        events (array-like): Event indicators (1=event, 0=censored).
        ax (matplotlib.axes.Axes, optional): Existing axis to plot into.
        label (str, optional): Label for the curve.
        save_path (str, optional): Path to save the plot.
        **kwargs: Additional arguments passed to lifelines.KaplanMeierFitter.plot_survival_function.

    Returns:
        ax: The matplotlib axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6)) # Create new figure if no ax provided
    else:
        fig = ax.figure # Get figure from ax

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label=label)
    
    # Customize plot_survival_function arguments for publication quality
    kmf.plot_survival_function(
        ax=ax, 
        ci_show=True,       # Show confidence intervals
        color='steelblue',  # Line color
        ls='-',             # Line style
        lw=2,               # Line width
        censor_styles={'marker': 'o', 'ms': 6, 'mew': 1.5, 'alpha': 0.7}, # Censor marker style
        **kwargs
    )

    ax.set_title("Kaplan-Meier Survival Curve", fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Survival Probability", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower left', fontsize=12) # Better legend placement for survival curves
    ax.set_ylim([-0.02, 1.02]) # Ensure y-axis covers 0 to 1 with slight padding
    ax.set_xlim(left=0) # Ensure x-axis starts at 0 or appropriate minimum

    plt.tight_layout()
    if save_path and ax is fig.gca(): # Only save if we created the figure in this function
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    elif save_path and fig: # If an ax was passed but we still want to save the figure it belongs to
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if ax is fig.gca(): # Only show if we created the figure
        plt.show()
    return ax