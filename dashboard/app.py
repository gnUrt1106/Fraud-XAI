"""
CIES Dashboard — Proof-of-Concept (RQ3)

Interactive Streamlit app with 4 panels:
    1. Performance — PR-AUC / F1 / Recall bar chart (Plotly)
    2. Explanation Fidelity — CIES mean ± std with alert threshold (Plotly Heatmap)
    3. SHAP Comparison — CIES Distribution (Plotly Boxplot) & Accuracy-Credibility (Plotly Scatter)
    4. Alert Log — Instances with CIES < threshold & Instance Explorer

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
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Ensure project root is importable
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Config ───────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(project_root, "outputs", "results")
CIES_ALERT_THRESHOLD = 0.96

st.set_page_config(
    page_title="CIES Dashboard — Credit Card Fraud",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color palette mapping for Models and Conditions
MODEL_COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "CatBoost": "#3498db",
    "Logistic Regression": "#9b59b6",
}
CONDITION_MARKERS = {
    "Class-weighting": "circle",
    "SMOTE": "square",
    "SMOTE-ENN": "diamond",
}

# ── Data Loading ─────────────────────────────────────────────────────

@st.cache_data
def load_summary():
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
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
    # Inject Custom Premium CSS
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Reduce top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Premium Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            border: 1px solid #e9ecef;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
            border-color: #cbd5e1;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2c3e50 !important;
            color: white !important;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Premium Metrics */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        }
        
        /* Headers styling */
        h1, h2, h3 {
            color: #1e293b;
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🔍 Fraud-XAI: Explanation Stability Dashboard</h1>", unsafe_allow_html=True)

    raw_df = load_summary()
    raw_cies_details = load_cies_details()

    if raw_df.empty:
        st.error(
            "No experiment results found. "
            "Please run `python run_experiments.py` first."
        )
        return

    # Map model abbreviations to full names
    MODEL_NAME_MAP = {
        "RF": "Random Forest",
        "XGB": "XGBoost",
        "LR": "Logistic Regression",
        "CatBoost": "CatBoost"
    }
    
    df = raw_df.copy()
    df["Model"] = df["Model"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    df["Config"] = df["Model"] + " (" + df["Condition"] + ")"

    # Map details keys
    cies_details = {}
    for k, v in raw_cies_details.items():
        parts = k.split("_", 1)
        if len(parts) == 2:
            m, c = parts
            full_m = MODEL_NAME_MAP.get(m, m)
            new_key = f"{full_m}_{c}"
            cies_details[new_key] = v
        else:
            cies_details[k] = v

    # Create Tabs for layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance", 
        "🛡️ Explanation Fidelity", 
        "🔬 Analysis (Boxplot & Scatter)", 
        "🚨 Alert Log & Explorer"
    ])

    # ── Panel 1: Performance ─────────────────────────────────────────
    with tab1:
        st.markdown("<h3 style='margin-top: 0;'>📊 Performance Metrics Comparison</h3>", unsafe_allow_html=True)
        st.markdown("Đánh giá độ chính xác của các mô hình trên tập kiểm thử (Test Set).")
        st.write("") # Spacer
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            metric = st.selectbox("Select Metric", ["PR-AUC", "F1", "Recall", "ROC-AUC"], index=0)
        with col2:
            sort_by = st.selectbox("Sort Output By", ["PR-AUC", "CIES_mean", "F1"], index=0)

        df_sorted = df.sort_values(sort_by, ascending=False) if sort_by in df.columns else df

        # Plotly Bar Chart
        if metric in df.columns:
            with st.container(border=True):
                fig = px.bar(
                    df_sorted, 
                    x="Config", 
                    y=metric,
                    color="Model",
                    color_discrete_map=MODEL_COLORS,
                    title=f"{metric} Comparison by Configuration",
                    labels={"Config": "Configuration", metric: metric},
                    text_auto='.3f'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Raw Data Table"):
            st.dataframe(df_sorted, use_container_width=True)

    # ── Panel 2: Explanation Fidelity ────────────────────────────────
    with tab2:
        st.markdown("<h3 style='margin-top: 0;'>🛡️ Explanation Fidelity (CIES)</h3>", unsafe_allow_html=True)
        st.markdown("Đánh giá mức độ ổn định của tính giải thích (SHAP Stability) thông qua chỉ số CIES.")
        st.write("") # Spacer

        if "CIES_mean" in df.columns:
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            with col1:
                best = df.loc[df["CIES_mean"].idxmax()]
                st.metric("Best CIES", f"{best['CIES_mean']:.3f}", delta=f"{best['Model']} ({best['Condition']})")
            with col2:
                worst = df.loc[df["CIES_mean"].idxmin()]
                st.metric("Worst CIES", f"{worst['CIES_mean']:.3f}", delta=f"{worst['Model']} ({worst['Condition']})", delta_color="inverse")
            with col3:
                below = (df["CIES_mean"] < CIES_ALERT_THRESHOLD).sum()
                st.metric("⚠️ Below Threshold", f"{below} / {len(df)}", delta=f"Threshold = {CIES_ALERT_THRESHOLD}", delta_color="inverse")

            # Plotly Heatmap
            st.markdown("### CIES Mean Heatmap")
            with st.container(border=True):
                pivot_df = df.pivot(index="Model", columns="Condition", values="CIES_mean")
                
                fig = px.imshow(
                    pivot_df, 
                    text_auto=".3f", 
                    color_continuous_scale="YlGnBu",
                    title="CIES Mean by Model × Condition",
                    labels=dict(x="Condition", y="Model", color="CIES Mean")
                )
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("💡 Threshold Selection Rationale & CIES Analysis")
            col_rationale, col_std = st.columns(2)
            with col_rationale:
                st.info(
                    "**🎯 Data-Driven Threshold (0.96)**\n\n"
                    "Dựa vào phân bố tự nhiên, có một điểm gãy rõ rệt phân tách 2 nhóm:\n"
                    "* **Nhóm ổn định:** Đạt CIES từ `0.976 – 0.993` (RF, LR, CatBoost, XGB Class-weighting).\n"
                    "* **Nhóm rủi ro:** Đạt CIES dưới `0.960` (XGB SMOTE-ENN, XGB SMOTE).\n\n"
                    "Ngưỡng `0.96` giúp hệ thống bắt được các cấu hình bị mất ổn định giải thích khi bị ảnh hưởng bởi nhiễu sinh ra do SMOTE."
                )
            with col_std:
                st.warning(
                    "**⚠️ Vai trò quan trọng của Độ lệch chuẩn (CIES_std)**\n\n"
                    "CIES_std rất hữu ích để giám sát mức độ nhất quán của giải thích. Một mô hình có CIES_mean ở mức khá "
                    "nhưng CIES_std lớn (outliers) cảnh báo sự thiếu ổn định nghiêm trọng ở các vùng biên quyết định (decision boundaries)."
                )
        else:
            st.info("CIES results not available. Re-run experiments without --skip-cies.")

    # ── Panel 3: Boxplot & Scatter ───────────────────────────────────
    with tab3:
        st.markdown("<h3 style='margin-top: 0;'>🔬 Detailed Analysis: Distribution & Trade-offs</h3>", unsafe_allow_html=True)
        st.write("")

        if cies_details and "CIES_mean" in df.columns:
            # 1. Plotly Boxplot for CIES Distribution
            rows = []
            for conf_key, res in cies_details.items():
                m, c = conf_key.split("_", 1)
                for score in res["cies_scores"]:
                    rows.append({"Model": m, "Condition": c, "CIES": score, "Config": f"{m} ({c})"})
            df_box = pd.DataFrame(rows)

            st.markdown("#### CIES Distribution by Model × Condition")
            with st.container(border=True):
                fig_box = px.box(
                    df_box, 
                    x="Model", 
                    y="CIES", 
                    color="Condition", 
                    color_discrete_sequence=["#E91E63", "#00BCD4", "#FF9800"], # Vibrant Pink, Cyan, Orange
                    points="outliers",
                    hover_data=["Config"]
                )
                fig_box.add_hline(y=CIES_ALERT_THRESHOLD, line_dash="dash", line_color="red", annotation_text=f"Alert Threshold ({CIES_ALERT_THRESHOLD})")
                fig_box.update_layout(height=500, boxmode='group', yaxis_title="CIES Score", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_box, use_container_width=True)

            # 2. Plotly Scatter Plot: Accuracy vs Credibility
            st.subheader("Accuracy–Credibility Trade-off")
            with st.container(border=True):
                fig_scatter = px.scatter(
                    df, 
                    x="PR-AUC", 
                    y="CIES_mean", 
                    color="Model",
                    symbol="Condition",
                    color_discrete_map=MODEL_COLORS,
                    symbol_map=CONDITION_MARKERS,
                    size_max=15,
                    hover_name="Config",
                    hover_data={"PR-AUC": ':.3f', "CIES_mean": ':.3f'}
                )
                # Increase marker size and use white outline for better contrast
                fig_scatter.update_traces(marker=dict(size=14, line=dict(width=1.5, color='#ffffff')))
                fig_scatter.add_hline(y=CIES_ALERT_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Threshold")
                fig_scatter.update_layout(height=500, xaxis_title="PR-AUC (Accuracy)", yaxis_title="CIES_mean (Credibility)", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.info("No detailed CIES results found. Run experiments first.")

    # ── Panel 4: Alert Log & Explorer ────────────────────────────────
    with tab4:
        st.markdown("<h3 style='margin-top: 0;'>🚨 Alert Log & Interactive Instance Explorer</h3>", unsafe_allow_html=True)
        st.write("")
        
        col_thresh, _ = st.columns([1, 2])
        with col_thresh:
            threshold = st.slider(
                "CIES Alert Threshold",
                min_value=0.0, max_value=1.0,
                value=CIES_ALERT_THRESHOLD, step=0.01,
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
                st.warning(f"**Alert!** Found instances falling below the credibility threshold of {threshold}.")
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
            else:
                st.success(f"✅ No instances below CIES threshold {threshold}")
                
            st.markdown("---")
            st.subheader("🔍 Interactive SHAP & CIES Instance Explorer")
            
            config_list = list(cies_details.keys())
            selected_config = st.selectbox(
                "Select Configuration to Explore Instances",
                config_list,
                key="explore_config"
            )
            
            config_data = cies_details[selected_config]
            if "shap_values" in config_data and "feature_values" in config_data:
                cies_scores = config_data["cies_scores"]
                shap_values = config_data["shap_values"]
                feature_values = config_data["feature_values"]
                feature_names = config_data["feature_names"]

                # Global SHAP Beeswarm Plot
                with st.expander("🐝 Global SHAP Summary (Beeswarm Plot)", expanded=False):
                    st.write("Biểu đồ phân phối giá trị SHAP của toàn bộ 100 mẫu thử nghiệm:")
                    shap_arr = np.array(shap_values)
                    feat_arr = np.array(feature_values)

                    with st.container(border=True):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_arr, feat_arr, feature_names=feature_names, show=False)
                        plt.title(f"SHAP Summary - {selected_config}", fontsize=14, pad=15)
                        plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)

                # Individual Instance Options
                st.write("### Individual Instance Analysis")
                instance_options = [
                    f"Instance {i} (CIES Score: {cies_scores[i]:.3f})"
                    for i in range(len(cies_scores))
                ]
                
                selected_idx = st.selectbox(
                    "Select Instance to Visualize",
                    range(len(cies_scores)),
                    format_func=lambda x: instance_options[x],
                    key="explore_instance"
                )
                
                inst_shap = shap_values[selected_idx]
                inst_feat = feature_values[selected_idx]
                
                inst_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": inst_shap,
                    "Feature Value (Scaled)": inst_feat,
                    "Absolute Impact": np.abs(inst_shap)
                }).sort_values(by="Absolute Impact", ascending=False).head(10)
                
                st.write(f"**Top 10 Contributing Features for Instance {selected_idx}**")
                
                # Plotly Bar Chart for Top 10 Features
                with st.container(border=True):
                    fig_inst = px.bar(
                        inst_df.sort_values(by="Absolute Impact", ascending=True), 
                        x="SHAP Value", 
                        y="Feature", 
                        orientation='h',
                        color="SHAP Value",
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        color_continuous_midpoint=0,
                        title="Feature Contribution Breakdown"
                    )
                    fig_inst.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_inst, use_container_width=True)
                
                st.caption("💡 Giá trị dương (Đỏ) đẩy dự đoán về FRAUD, giá trị âm (Xanh dương) đẩy dự đoán về LEGITIMATE.")
                
                with st.expander("View Data Table"):
                    st.dataframe(
                        inst_df[["Feature", "Feature Value (Scaled)", "SHAP Value"]].style.format({
                            "Feature Value (Scaled)": "{:.4f}",
                            "SHAP Value": "{:.4f}"
                        }),
                        use_container_width=True
                    )
            else:
                st.info("The selected configuration does not contain detailed SHAP data.")
        else:
            st.info("No CIES detail files found.")

if __name__ == "__main__":
    main()
