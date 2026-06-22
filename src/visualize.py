"""
CIES-specific visualization utilities.
Phase 8 of the pipeline — analysis & reporting.

Plots:
    - CIES boxplot (instance-level distribution, Figure 5 style)
    - Accuracy–Credibility scatter (PR-AUC vs CIES, Figure 3 style)
    - Sensitivity analysis (CIES vs ε)
    - Performance comparison bar chart
    - Results heatmap (model × condition)
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Theme ────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
})

# Color palette for models
MODEL_COLORS = {
    "RF": "#2ecc71",
    "XGB": "#e74c3c",
    "CatBoost": "#3498db",
    "LR": "#9b59b6",
}

CONDITION_MARKERS = {
    "Class-weighting": "o",
    "SMOTE": "s",
    "SMOTE-ENN": "D",
}


# ── CIES Boxplot (Figure 5 style) ───────────────────────────────────

def plot_cies_boxplot(
    results: dict,
    save_path: str = "outputs/figures/cies_boxplot.png",
):
    """
    Boxplot of instance-level CIES scores grouped by model and condition.

    Args:
        results: Dict keyed by (model, condition) → CIES evaluate() output.
        save_path: Where to save the figure.
    """
    rows = []
    for (model, cond), res in results.items():
        for score in res["cies_scores"]:
            rows.append({"Model": model, "Condition": cond, "CIES": score})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=df, x="Model", y="CIES", hue="Condition",
        palette="Set2", ax=ax, linewidth=1.5,
    )
    ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.6, label="Alert threshold (0.75)")
    ax.set_title("CIES Distribution by Model × Condition", fontweight="bold", pad=15)
    ax.set_ylabel("CIES Score", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.legend(title="Condition", loc="lower left")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("CIES boxplot saved to %s", save_path)


# ── Accuracy–Credibility Scatter (Figure 3 style) ───────────────────

def plot_accuracy_credibility_scatter(
    summary_df: pd.DataFrame,
    save_path: str = "outputs/figures/accuracy_credibility_scatter.png",
):
    """
    Scatter plot: PR-AUC (x) vs CIES mean (y).
    Each point = one (model, condition) configuration.

    Args:
        summary_df: DataFrame with columns [Model, Condition, PR-AUC, CIES_mean].
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in summary_df.iterrows():
        model = row["Model"]
        cond = row["Condition"]
        ax.scatter(
            row["PR-AUC"], row["CIES_mean"],
            color=MODEL_COLORS.get(model, "#666"),
            marker=CONDITION_MARKERS.get(cond, "o"),
            s=150, edgecolor="white", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            f"{model}\n{cond}",
            (row["PR-AUC"], row["CIES_mean"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, alpha=0.8,
        )

    ax.set_xlabel("PR-AUC (Accuracy)", fontweight="bold")
    ax.set_ylabel("CIES (Credibility)", fontweight="bold")
    ax.set_title("Accuracy–Credibility Trade-off", fontweight="bold", pad=15)
    ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.4, label="CIES threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Accuracy–Credibility scatter saved to %s", save_path)


# ── Sensitivity Analysis ────────────────────────────────────────────

def plot_sensitivity_analysis(
    sensitivity_results: dict,
    save_path: str = "outputs/figures/sensitivity_analysis.png",
):
    """
    Line plot: CIES mean vs ε for each (model, condition).

    Args:
        sensitivity_results: Dict keyed by (model, cond) → {eps: {cies_mean, cies_std}}.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for (model, cond), eps_dict in sensitivity_results.items():
        epsilons = sorted(eps_dict.keys())
        means = [eps_dict[e]["cies_mean"] for e in epsilons]
        stds = [eps_dict[e]["cies_std"] for e in epsilons]

        color = MODEL_COLORS.get(model, "#666")
        marker = CONDITION_MARKERS.get(cond, "o")
        label = f"{model} ({cond})"

        ax.errorbar(
            epsilons, means, yerr=stds,
            marker=marker, color=color, label=label,
            linewidth=1.5, capsize=3, markersize=7,
        )

    ax.set_xlabel("Noise Level (ε)", fontweight="bold")
    ax.set_ylabel("CIES Mean", fontweight="bold")
    ax.set_title("CIES Sensitivity to Perturbation Level", fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Sensitivity analysis plot saved to %s", save_path)


# ── Performance Comparison Bar Chart ─────────────────────────────────

def plot_performance_comparison(
    summary_df: pd.DataFrame,
    save_path: str = "outputs/figures/performance_comparison.png",
):
    """
    Grouped bar chart: PR-AUC / F1 / Recall for each configuration.
    """
    df_melted = summary_df.melt(
        id_vars=["Model", "Condition"],
        value_vars=["PR-AUC", "F1", "Recall"],
        var_name="Metric", value_name="Score",
    )
    df_melted["Config"] = df_melted["Model"] + " (" + df_melted["Condition"] + ")"

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=df_melted, x="Score", y="Config", hue="Metric",
        palette="Set2", ax=ax,
    )
    ax.set_title("Performance Comparison (sorted by PR-AUC)", fontweight="bold", pad=15)
    ax.set_xlabel("Score", fontweight="bold")
    ax.set_ylabel("Configuration", fontweight="bold")
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Performance comparison saved to %s", save_path)


# ── Results Heatmap ──────────────────────────────────────────────────

def plot_results_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "CIES_mean",
    save_path: str = "outputs/figures/cies_heatmap.png",
):
    """
    Heatmap of a metric across Model × Condition.
    """
    pivot = summary_df.pivot_table(
        index="Model", columns="Condition", values=metric, aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"{metric} by Model × Condition", fontweight="bold", pad=15)
    ax.set_ylabel("Model", fontweight="bold")
    ax.set_xlabel("Condition", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("%s heatmap saved to %s", metric, save_path)


# ── Summary table loader ────────────────────────────────────────────

def load_all_results(results_dir: str = "outputs/results") -> pd.DataFrame:
    """Load all JSON result files into a summary DataFrame."""
    rows = []
    if not os.path.exists(results_dir):
        logger.warning("Results directory %s does not exist.", results_dir)
        return pd.DataFrame()

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                rows.append(json.load(f))

    return pd.DataFrame(rows)
