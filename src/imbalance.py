"""
Imbalance handling for CIES experiments.
Phase 3 of the pipeline.

Three experimental conditions:
    C0 — Class-weighting baseline (no synthetic data; weight applied in model)
    C1 — SMOTE oversampling (1:1 ratio)
    C2 — SMOTE-ENN hybrid (oversample + clean noise)

⚠️  Only applied to *training* data.  Test set is NEVER resampled.
"""

import logging

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# ── Condition labels ─────────────────────────────────────────────────

CONDITIONS = {
    "C0": "Class-weight (no resampling)",
    "C1": "SMOTE",
    "C2": "SMOTE-ENN",
}


def apply_condition(
    X_train,
    y_train,
    condition: str = "C0",
    random_state: int = 42,
) -> tuple:
    """
    Apply an imbalance-handling condition to training data.

    Args:
        X_train: Training features (DataFrame or ndarray).
        y_train: Training labels.
        condition: One of 'C0', 'C1', 'C2'.
        random_state: Seed for reproducibility.

    Returns:
        (X_resampled, y_resampled) — unchanged for C0.
    """
    condition = condition.upper()

    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Choose from {list(CONDITIONS.keys())}"
        )

    _log_ratio("Before", y_train)

    if condition == "C0":
        logger.info(
            "C0 — Class-weighting baseline: returning data unchanged. "
            "Class weights will be set inside the model."
        )
        return X_train, y_train

    elif condition == "C1":
        logger.info("C1 — Applying SMOTE (random_state=%d)", random_state)
        sampler = SMOTE(random_state=random_state)

    elif condition == "C2":
        logger.info("C2 — Applying SMOTE-ENN (random_state=%d)", random_state)
        sampler = SMOTEENN(random_state=random_state)

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    _log_ratio("After", y_res)

    return X_res, y_res


def _log_ratio(label: str, y) -> None:
    """Log class distribution."""
    if hasattr(y, "value_counts"):
        counts = y.value_counts()
    else:
        unique, cnts = np.unique(y, return_counts=True)
        counts = dict(zip(unique, cnts))

    logger.info("%s resampling — class distribution: %s", label, dict(counts))
