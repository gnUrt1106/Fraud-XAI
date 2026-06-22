"""
CIES Dashboard — Proof-of-Concept (RQ3)

Streamlit app with 4 panels:
    1. Performance — PR-AUC / F1 / Recall bar chart
    2. Explanation Fidelity — CIES mean ± std with alert threshold
    3. SHAP Comparison — Beeswarm / waterfall across conditions
    4. Alert Log — Instances with CIES < threshold

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is importable
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ── Config ───────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(project_root, "outputs", "results")
FIGURES_DIR = os.path.join(project_root, "outputs", "figures")
CIES_ALERT_THRESHOLD = 0.75

st.set_page_config(
    page_title="CIES Dashboard — Credit Card Fraud",
    page_icon="🔍",
    layout="wide",
)


# ── Data Loading ─────────────────────────────────────────────────────

@st.cache_data
def load_summary():
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    st.warning("No summary.csv found. Run experiments first.")
    return pd.DataFrame()


@st.cache_data
def load_cies_details():
    """Load per-config CIES JSON files."""
    details = {}
    if not os.path.exists(RESULTS_DIR):
        return details
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_cies.json"):
            key = fname.replace("_cies.json", "")
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                details[key] = json.load(f)
    return details


# ── Main App ─────────────────────────────────────────────────────────

def main():
    st.title("🔍 CIES Dashboard — Explainable Fraud Detection")
    st.caption(
        "Proof-of-concept monitoring dashboard for Credibility Index via "
        "Explanation Stability (Văduva et al., 2026)"
    )

    df = load_summary()
    cies_details = load_cies_details()

    if df.empty:
        st.error(
            "No experiment results found. "
            "Please run `python run_experiments.py` first."
        )
        return

    # ── Panel 1: Performance ─────────────────────────────────────────
    st.header("📊 Panel 1 — Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Metric", ["PR-AUC", "F1", "Recall", "ROC-AUC"])
    with col2:
        sort_by = st.selectbox("Sort by", ["PR-AUC", "CIES_mean", "F1"])

    if sort_by in df.columns:
        df_sorted = df.sort_values(sort_by, ascending=False)
    else:
        df_sorted = df

    st.bar_chart(
        df_sorted.set_index(
            df_sorted["Model"] + " (" + df_sorted["Condition"] + ")"
        )[metric] if metric in df.columns else pd.Series(),
    )
    st.dataframe(df_sorted, use_container_width=True)

    # ── Panel 2: Explanation Fidelity ────────────────────────────────
    st.header("🛡️ Panel 2 — Explanation Fidelity (CIES)")

    if "CIES_mean" in df.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            best = df.loc[df["CIES_mean"].idxmax()]
            st.metric("Best CIES", f"{best['CIES_mean']:.3f}",
                       delta=f"{best['Model']} ({best['Condition']})")
        with col2:
            worst = df.loc[df["CIES_mean"].idxmin()]
            st.metric("Worst CIES", f"{worst['CIES_mean']:.3f}",
                       delta=f"{worst['Model']} ({worst['Condition']})",
                       delta_color="inverse")
        with col3:
            below = (df["CIES_mean"] < CIES_ALERT_THRESHOLD).sum()
            st.metric("⚠️ Below Threshold", f"{below} / {len(df)}",
                       delta=f"Threshold = {CIES_ALERT_THRESHOLD}")

        # CIES heatmap image
        heatmap_path = os.path.join(FIGURES_DIR, "cies_heatmap.png")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="CIES Heatmap (Model × Condition)")
    else:
        st.info("CIES results not available. Re-run experiments without --skip-cies.")

    # ── Panel 3: SHAP Comparison ─────────────────────────────────────
    st.header("🔬 Panel 3 — SHAP Comparison")

    boxplot_path = os.path.join(FIGURES_DIR, "cies_boxplot.png")
    scatter_path = os.path.join(FIGURES_DIR, "accuracy_credibility_scatter.png")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(boxplot_path):
            st.image(boxplot_path, caption="CIES Distribution (Boxplot)")
        else:
            st.info("CIES boxplot not generated yet.")
    with col2:
        if os.path.exists(scatter_path):
            st.image(scatter_path, caption="Accuracy–Credibility Trade-off")
        else:
            st.info("Scatter plot not generated yet.")

    # ── Panel 4: Alert Log ───────────────────────────────────────────
    st.header("🚨 Panel 4 — Alert Log")

    threshold = st.slider(
        "CIES Alert Threshold",
        min_value=0.0, max_value=1.0,
        value=CIES_ALERT_THRESHOLD, step=0.05,
    )

    if cies_details:
        alert_rows = []
        for config_key, cies_data in cies_details.items():
            scores = np.array(cies_data.get("cies_scores", []))
            low_mask = scores < threshold
            n_alerts = low_mask.sum()
            if n_alerts > 0:
                alert_rows.append({
                    "Configuration": config_key,
                    "Instances Below Threshold": int(n_alerts),
                    "Pct Below": f"{100 * n_alerts / len(scores):.1f}%",
                    "Min CIES": f"{scores.min():.3f}",
                    "Mean CIES": f"{scores.mean():.3f}",
                })

        if alert_rows:
            st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
        else:
            st.success(f"✅ No instances below CIES threshold {threshold}")
    else:
        st.info("No CIES detail files found.")


if __name__ == "__main__":
    main()
