"""
Credibility Index via Explanation Stability (CIES).
Phase 7 of the pipeline — core contribution metric.

Implementation follows Algorithm 1 from:
    Văduva, A.-G., Oprea, S.-V., & Bâra, A. (2026).
    "Measuring the fragility of trust: Devising CIES for
     business decision support systems."

Parameters (paper defaults):
    N = 100   test instances for model-level aggregation
    K = 20    perturbed neighbors per instance
    ε = 0.03  noise level (3% multiplicative Gaussian)
"""

import logging

import numpy as np
import shap
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


class CIESEvaluator:
    """
    Credibility Index via Explanation Stability.
    Theo Algorithm 1, Văduva et al. (2026).
    """

    def __init__(
        self,
        model,
        model_type: str = "tree",       # 'tree' | 'linear'
        X_background=None,               # Required for linear explainer
        noise_level: float = 0.03,       # epsilon
        n_neighbors: int = 20,           # K
        n_instances: int = 100,          # N
        random_state: int = 42,
    ):
        self.noise_level = noise_level
        self.n_neighbors = n_neighbors
        self.n_instances = n_instances
        self.rng = np.random.RandomState(random_state)

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            if X_background is None:
                raise ValueError("X_background is required for LinearExplainer")
            self.explainer = shap.LinearExplainer(model, X_background)
        else:
            raise ValueError("model_type must be 'tree' or 'linear'")

        logger.info(
            "CIESEvaluator initialized: ε=%.3f, K=%d, N=%d, type=%s",
            noise_level, n_neighbors, n_instances, model_type,
        )

    # ── SHAP values ──────────────────────────────────────────────────

    def _get_shap(self, X) -> np.ndarray:
        """Return SHAP values for class 1 (binary classification)."""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        return np.atleast_2d(sv)

    # ── Harmonic rank weights ────────────────────────────────────────

    @staticmethod
    def _harmonic_weights(phi: np.ndarray) -> np.ndarray:
        """Compute harmonic rank weights from a SHAP vector."""
        # rank 1 = most important (sorted by |phi| descending)
        ranks = np.argsort(np.argsort(-np.abs(phi))) + 1
        inv_ranks = 1.0 / ranks
        return inv_ranks / inv_ranks.sum()

    # ── Perturbation ─────────────────────────────────────────────────

    def _perturb(self, x: np.ndarray) -> np.ndarray:
        """Gaussian noise proportional to |x_j|; zero-valued features get small noise."""
        sigmas = np.where(x != 0, self.noise_level * np.abs(x), self.noise_level)
        return x + self.rng.normal(0, sigmas)

    # ── Instance-level CIES ──────────────────────────────────────────

    def _instance_cies(
        self, x: np.ndarray, phi_x: np.ndarray
    ) -> "tuple[float, float]":
        """
        Compute CIES(x) and Baseline(x) (uniform weights) for one instance.

        Returns:
            (cies_score, baseline_score)
        """
        M = len(phi_x)
        weights_R = self._harmonic_weights(phi_x)          # rank-weighted
        weights_U = np.full(M, 1.0 / M)                    # uniform baseline

        mag_R = np.sum(weights_R * np.abs(phi_x))
        mag_U = np.sum(np.abs(phi_x))                       # = sum |phi|

        if mag_R == 0:
            return 1.0, 1.0

        d_R_list, d_U_list = [], []

        for _ in range(self.n_neighbors):
            x_k = self._perturb(x)
            phi_k = self._get_shap(x_k.reshape(1, -1)).flatten()

            diff = np.abs(phi_x - phi_k)
            d_R_list.append(np.dot(weights_R, diff))
            d_U_list.append(np.dot(weights_U, diff))

        d_R_mean = np.mean(d_R_list)
        d_U_mean = np.mean(d_U_list)

        cies = max(0.0, 1 - d_R_mean / mag_R)
        baseline = max(0.0, 1 - d_U_mean * M / max(mag_U, 1e-10))
        return cies, baseline

    # ── Model-level evaluation ───────────────────────────────────────

    def evaluate(self, X_test, feature_names=None) -> dict:
        """
        Compute CIES for N random instances from the test set.

        Returns dict with:
            cies_mean, cies_std, cies_min/max, quartiles,
            baseline_mean, wilcoxon_stat/p, significance flag,
            raw cies_scores and baseline_scores,
            and feature values, SHAP values, and feature names.
        """
        n = min(self.n_instances, len(X_test))
        idx = self.rng.choice(len(X_test), n, replace=False)

        # Ensure numpy array and retrieve feature names if available
        if hasattr(X_test, "values"):
            X_sample = X_test.values[idx]
            if feature_names is None:
                feature_names = list(X_test.columns)
        elif hasattr(X_test, "iloc"):
            X_sample = X_test.iloc[idx].values
            if feature_names is None:
                feature_names = list(X_test.columns)
        else:
            X_sample = np.asarray(X_test)[idx]

        if feature_names is None:
            feature_names = [f"V{i}" for i in range(X_sample.shape[1])]

        # Batch SHAP (faster than per-instance)
        phi_all = self._get_shap(X_sample)

        cies_scores, base_scores = [], []
        for i in range(n):
            c, b = self._instance_cies(X_sample[i], phi_all[i])
            cies_scores.append(c)
            base_scores.append(b)
            if (i + 1) % 10 == 0:
                logger.info("  CIES progress: %d/%d instances", i + 1, n)

        cs = np.array(cies_scores)
        bs = np.array(base_scores)

        # Wilcoxon signed-rank test: CIES vs Baseline (paired, two-sided)
        try:
            stat, pval = wilcoxon(cs, bs)
        except ValueError:
            # All differences are zero
            stat, pval = 0.0, 1.0

        result = {
            "cies_mean": float(cs.mean()),
            "cies_std": float(cs.std()),
            "cies_min": float(cs.min()),
            "cies_q25": float(np.percentile(cs, 25)),
            "cies_median": float(np.median(cs)),
            "cies_q75": float(np.percentile(cs, 75)),
            "cies_max": float(cs.max()),
            "baseline_mean": float(bs.mean()),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(pval),
            "significant": bool(pval < 0.01),
            "cies_scores": cs.tolist(),
            "baseline_scores": bs.tolist(),
            "n_instances": n,
            "feature_names": feature_names,
            "feature_values": X_sample.tolist(),
            "shap_values": phi_all.tolist(),
        }

        logger.info(
            "CIES: %.3f ± %.3f | Baseline: %.3f | Wilcoxon p=%.2e %s",
            result["cies_mean"], result["cies_std"],
            result["baseline_mean"], result["wilcoxon_p"],
            "✓ significant" if result["significant"] else "✗ not significant",
        )
        return result

    # ── Sensitivity analysis ─────────────────────────────────────────

    def sensitivity_analysis(
        self,
        X_test: np.ndarray,
        epsilons: tuple = (0.01, 0.03, 0.05, 0.10),
    ) -> dict:
        """Run CIES across multiple noise levels to validate robustness."""
        results = {}
        original_eps = self.noise_level

        for eps in epsilons:
            logger.info("Sensitivity analysis: ε=%.2f", eps)
            self.noise_level = eps
            res = self.evaluate(X_test)
            results[eps] = {
                "cies_mean": res["cies_mean"],
                "cies_std": res["cies_std"],
            }

        self.noise_level = original_eps
        return results
