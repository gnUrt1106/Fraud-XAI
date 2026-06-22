"""
Optuna hyperparameter tuning for all CIES experiment models.

Supports: RF, XGBoost, CatBoost, Logistic Regression.
Objective: maximize PR-AUC via 5-fold Stratified CV.
"""

import logging

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Suppress Optuna info spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y, model_name):
    """Single Optuna trial — 5-fold CV, returns mean PR-AUC."""
    pr_auc_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in cv.split(X, y):
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        if model_name == "XGBClassifier":
            param = {
                "tree_method": "hist",
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            neg = np.sum(np.asarray(y_train) == 0)
            pos = np.sum(np.asarray(y_train) == 1)
            if pos > 0:
                param["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", 1.0, neg / pos,
                )
            model = XGBClassifier(**param, random_state=42, eval_metric="aucpr")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "RandomForestClassifier":
            param = {
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", "balanced_subsample", None],
                ),
            }
            model = RandomForestClassifier(**param, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "CatBoostClassifier":
            param = {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "auto_class_weights": trial.suggest_categorical(
                    "auto_class_weights", ["Balanced", "SqrtBalanced", None],
                ),
                "verbose": 0,
            }
            model = CatBoostClassifier(**param, random_seed=42)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "LogisticRegression":
            param = {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "saga",  # supports both l1 and l2
                "max_iter": 2000,
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None],
                ),
            }
            model = LogisticRegression(**param, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

        else:
            raise ValueError(f"Tuning not supported for {model_name}")

        score = average_precision_score(y_val, y_prob)
        pr_auc_scores.append(score)

    return np.mean(pr_auc_scores)


def run_optimization(X, y, model_name="XGBClassifier", n_trials=20):
    """
    Run Optuna study to find optimal hyperparameters.

    Returns:
        dict of best parameters.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_PR_AUC",
    )

    logger.info(
        "Starting Optuna tuning for %s (%d trials)...", model_name, n_trials
    )
    study.optimize(
        lambda trial: objective(trial, X, y, model_name),
        n_trials=n_trials,
        n_jobs=1,
    )

    logger.info("Finished %d trials", len(study.trials))
    best = study.best_trial
    logger.info("Best PR-AUC: %.4f", best.value)
    logger.info("Best params:")
    for key, value in best.params.items():
        logger.info("  %s: %s", key, value)

    return best.params
