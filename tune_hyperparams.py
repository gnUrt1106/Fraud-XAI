#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning CLI.

Supports: XGBClassifier, RandomForestClassifier, CatBoostClassifier, LogisticRegression.

Usage:
    python tune_hyperparams.py --model XGBClassifier --trials 50
    python tune_hyperparams.py --model CatBoostClassifier --trials 30
    python tune_hyperparams.py --model LogisticRegression --trials 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import load_and_split, scale_features
from src.tuning.optuna_tuner import run_optimization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "RandomForestClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "LogisticRegression",
]


def main(n_trials: int, model_name: str):
    if model_name not in SUPPORTED_MODELS:
        logger.error(
            "Unsupported model '%s'. Choose from: %s",
            model_name, SUPPORTED_MODELS,
        )
        return

    logger.info("Loading data for hyperparameter tuning (%s)...", model_name)

    X_train, X_test, y_train, y_test = load_and_split()
    X_train_scaled, _ = scale_features(X_train, X_test)

    logger.info("Starting Optuna tuning on %s...", model_name)
    best_params = run_optimization(
        X_train_scaled, y_train,
        model_name=model_name,
        n_trials=n_trials,
    )

    logger.info("\n" + "=" * 50)
    logger.info("🎉 OPTIMIZATION COMPLETE 🎉")
    logger.info("=" * 50)
    logger.info("Best parameters for %s:", model_name)
    for key, value in best_params.items():
        if isinstance(value, float):
            logger.info("    %s: %.6f", key, value)
        else:
            logger.info("    %s: %s", key, value)
    logger.info("=" * 50)
    logger.info(
        "TIP: You can pass these as custom_params to src.models.get_model() "
        "or update configs/default.yaml."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Tuning for CIES experiment models",
    )
    parser.add_argument(
        "--model", type=str, default="XGBClassifier",
        choices=SUPPORTED_MODELS,
        help="Model to tune",
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of Optuna trials",
    )
    args = parser.parse_args()
    main(args.trials, args.model)
