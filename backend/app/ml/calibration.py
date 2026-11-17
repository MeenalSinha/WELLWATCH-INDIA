"""
Probability Calibration and Metric Evaluation Framework.
Ensures that predicted risk probabilities correspond to empirical clinical frequencies
using Platt scaling (sigmoid calibration) or isotonic regression.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, f1_score


class ProbabilityCalibrator:
    """
    Fits and evaluates post-hoc probability calibration for clinical risk scores.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, uncalibrated_scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fits calibration model on held-out validation predictions.
        """
        scores = np.asarray(uncalibrated_scores).reshape(-1, 1)
        labels = np.asarray(y_true).astype(int)

        self.calibrator.fit(scores, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, uncalibrated_scores: np.ndarray) -> np.ndarray:
        """
        Outputs calibrated posterior probabilities in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator has not been fitted.")

        scores = np.asarray(uncalibrated_scores).reshape(-1, 1)
        # Returns probability for positive class
        return self.calibrator.predict_proba(scores)[:, 1]

    @staticmethod
    def compute_calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Calculates Brier score loss, Expected Calibration Error (ECE), AUC-ROC, and PR-AUC.
        """
        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(y_prob).clip(1e-6, 1.0 - 1e-6)

        brier = float(brier_score_loss(y_t, y_p))

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_t)

        for i in range(n_bins):
            bin_mask = (y_p >= bins[i]) & (y_p < bins[i + 1])
            if i == n_bins - 1:
                bin_mask = (y_p >= bins[i]) & (y_p <= bins[i + 1])

            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_acc = np.mean(y_t[bin_mask])
                bin_conf = np.mean(y_p[bin_mask])
                ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

        metrics = {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(float(ece), 4),
        }

        # Classification metrics if both classes present
        if len(np.unique(y_t)) > 1:
            metrics["auc_roc"] = round(float(roc_auc_score(y_t, y_p)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_t, y_p)), 4)
            pred_labels = (y_p >= 0.5).astype(int)
            metrics["f1_score"] = round(float(f1_score(y_t, pred_labels, zero_division=0)), 4)

        return metrics
