"""
Plotting helpers — every figure produced in the thesis lives here.

Each function saves to `outputs/` (via config) and also shows the figure, so
the notebooks stay short. Colours match the original per-model scheme:
XGBoost=Blues/steelblue, TabPFN=Oranges/darkorange, AutoGluon=Greens.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src import evaluation


def plot_confusion_pair(y_true, y_pred, model_name, cmap="Blues", save_path=None):
    """Side-by-side raw-count and row-normalised confusion matrices."""
    cm, cm_norm, _, labels = evaluation.confusion_matrices(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{model_name} - Confusion Matrix (row-normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    if save_path:
        print(f"Saved confusion matrix -> {save_path}")


def plot_feature_importance(importances, model_name, color="steelblue",
                            top_n=None, save_path=None, horizontal=False):
    """Bar chart of feature importances.

    `importances` is a pandas Series indexed by feature name.
    """
    imp = importances.sort_values(ascending=False)
    if top_n:
        imp = imp.head(top_n)

    if horizontal:
        plt.figure(figsize=(10, 6))
        imp.sort_values().plot(kind="barh", color=color, edgecolor="white")
        plt.xlabel("Importance score")
        plt.ylabel("Feature")
    else:
        plt.figure(figsize=(10, 5))
        imp.plot(kind="bar", color=color, edgecolor="white")
        plt.ylabel("Importance score")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha="right")

    plt.title(f"{model_name} - Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    if save_path:
        print(f"Saved feature importance -> {save_path}")


def plot_tabpfn_sweep(results_df: pd.DataFrame, save_path=None):
    """Performance vs n_estimators (left) and inference time (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df["n_estimators"], results_df["f1_macro"],
                 marker="o", color="darkorange", linewidth=2, label="F1 Macro")
    axes[0].plot(results_df["n_estimators"], results_df["f1_weighted"],
                 marker="s", color="steelblue", linewidth=2, label="F1 Weighted")
    axes[0].plot(results_df["n_estimators"], results_df["accuracy"],
                 marker="^", color="seagreen", linewidth=2, label="Accuracy")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(config.N_ESTIMATORS_RANGE)
    axes[0].set_xticklabels([str(n) for n in config.N_ESTIMATORS_RANGE])
    axes[0].set_xlabel("n_estimators (log scale)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("TabPFN-3 - Performance vs n_estimators")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(n) for n in results_df["n_estimators"]],
                results_df["inference_time_s"],
                color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("Inference time (seconds)")
    axes[1].set_title("TabPFN-3 - Inference Time vs n_estimators")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    if save_path:
        print(f"Saved performance plot -> {save_path}")


def plot_tabpfn_marginal_gain(results_df: pd.DataFrame, save_path=None):
    """Incremental F1-macro gain per additional estimator."""
    marginal_gain = results_df["f1_macro"].diff().fillna(0)

    plt.figure(figsize=(8, 4))
    plt.bar([str(n) for n in results_df["n_estimators"]], marginal_gain,
            color="steelblue", edgecolor="white", alpha=0.85)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("n_estimators")
    plt.ylabel("\u0394F1 Macro (vs previous)")
    plt.title("TabPFN-3 - Marginal Gain in F1 Macro per Estimator Step")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    if save_path:
        print(f"Saved marginal gain plot -> {save_path}")


def plot_grid_heatmap(results_df, value_col, title, cmap="Greens",
                      fmt=".4f", save_path=None):
    """Heatmap of an AutoGluon grid metric (rows=folds, cols=stack levels)."""
    pivot = results_df.pivot(
        index="num_bag_folds", columns="num_stack_levels", values=value_col
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                cbar_kws={"label": title})
    plt.title(f"AutoGluon Grid Search - {title}")
    plt.xlabel("num_stack_levels")
    plt.ylabel("num_bag_folds")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    if save_path:
        print(f"Saved heatmap -> {save_path}")
