#!/usr/bin/env python3
"""
Tái tạo toàn bộ visualizations từ outputs/results/ đã có.

Chạy sau run_experiments.py hoặc bất cứ lúc nào muốn vẽ lại:
    /Users/thetrung/venvs/general/bin/python generate_visuals.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize import (
    plot_cies_boxplot,
    plot_accuracy_credibility_scatter,
    plot_sensitivity_analysis,
    plot_performance_comparison,
    plot_results_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"


def load_results():
    """Load tất cả JSON results thành summary DataFrame + CIES detail dicts."""
    perf_rows = []
    cies_details = {}

    if not os.path.exists(RESULTS_DIR):
        logger.error("Không tìm thấy %s — chạy run_experiments.py trước!", RESULTS_DIR)
        sys.exit(1)

    for fname in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, fname)

        if fname == "summary.csv" or fname.endswith(".gitkeep"):
            continue

        if fname.endswith("_cies.json"):
            key = fname.replace("_cies.json", "")
            with open(path) as f:
                cies_details[key] = json.load(f)

        elif fname.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            # Parse model + condition từ filename (e.g. RF_C0.json)
            stem = fname.replace(".json", "")
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                data["Model"] = parts[0]
                data["Condition"] = parts[1]
            perf_rows.append(data)

    if not perf_rows:
        logger.error("Không có kết quả nào trong %s!", RESULTS_DIR)
        sys.exit(1)

    df = pd.DataFrame(perf_rows)
    logger.info("Loaded %d experiment results", len(df))

    # Merge CIES means vào df
    for key, cies_data in cies_details.items():
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            model, cond = parts
            mask = (df["Model"] == model) & (df["Condition"] == cond)
            df.loc[mask, "CIES_mean"] = cies_data.get("cies_mean")
            df.loc[mask, "CIES_std"] = cies_data.get("cies_std")
            df.loc[mask, "Baseline_mean"] = cies_data.get("baseline_mean")

    # Parse cies_details thành nested dict keyed by (model, condition)
    cies_by_config = {}
    for key, data in cies_details.items():
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            cies_by_config[(parts[0], parts[1])] = data

    return df, cies_by_config


def main():
    logger.info("=" * 60)
    logger.info("Generating all visualizations from outputs/results/")
    logger.info("=" * 60)

    df, cies_by_config = load_results()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Performance comparison ─────────────────────────────────────
    logger.info("Plotting performance comparison...")
    plot_performance_comparison(
        df, save_path=f"{FIGURES_DIR}/performance_comparison.png"
    )

    # ── 2. PR-AUC heatmap ─────────────────────────────────────────────
    logger.info("Plotting PR-AUC heatmap...")
    if "PR-AUC" in df.columns:
        plot_results_heatmap(
            df, metric="PR-AUC",
            save_path=f"{FIGURES_DIR}/prauc_heatmap.png"
        )

    # ── 3. CIES plots (nếu có CIES results) ──────────────────────────
    if cies_by_config:
        logger.info("Plotting CIES boxplot...")
        plot_cies_boxplot(
            cies_by_config, save_path=f"{FIGURES_DIR}/cies_boxplot.png"
        )

        if "CIES_mean" in df.columns and df["CIES_mean"].notna().any():
            logger.info("Plotting CIES heatmap...")
            plot_results_heatmap(
                df, metric="CIES_mean",
                save_path=f"{FIGURES_DIR}/cies_heatmap.png"
            )

            logger.info("Plotting accuracy–credibility scatter...")
            plot_accuracy_credibility_scatter(
                df, save_path=f"{FIGURES_DIR}/accuracy_credibility_scatter.png"
            )
    else:
        logger.warning("Không có CIES results — chạy lại run_experiments.py không có --skip-cies")

    # ── 4. Save updated summary CSV ───────────────────────────────────
    csv_path = f"{RESULTS_DIR}/summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Summary CSV updated: %s", csv_path)

    logger.info("=" * 60)
    logger.info("✅ Tất cả visualizations đã được tạo trong: %s", FIGURES_DIR)
    logger.info("=" * 60)
    logger.info("Mở dashboard: /Users/thetrung/venvs/general/bin/python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
