"""
Leakage-Resistant Preprocessing Pipeline.
Guarantees that feature encoders, imputation statistics, and scalers are fitted
strictly on the training split, completely isolated from test and inference inputs.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class LeakageResistantPreprocessor:
    """
    Fits imputers, scalers, and bounds exclusively on training data.
    Applies frozen parameters during testing and production inference.
    """

    def __init__(self, feature_columns: List[str]):
        self.feature_columns = list(feature_columns)
        self.medians_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.mins_: Dict[str, float] = {}
        self.maxs_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "LeakageResistantPreprocessor":
        """
        Learns feature parameters strictly from training dataset.
        """
        for col in self.feature_columns:
            if col not in X.columns:
                raise ValueError(f"Required training column '{col}' missing from training dataframe.")

            series = pd.to_numeric(X[col], errors="coerce")
            valid = series.dropna()
            if len(valid) == 0:
                raise ValueError(f"Feature column '{col}' has zero valid numeric entries in training split.")

            self.medians_[col] = float(valid.median())
            self.means_[col] = float(valid.mean())
            std = float(valid.std())
            self.stds_[col] = std if std > 1e-6 else 1.0
            self.mins_[col] = float(valid.min())
            self.maxs_[col] = float(valid.max())

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, fill_missing: bool = True) -> np.ndarray:
        """
        Transforms input using frozen parameters. Does not compute any statistics from X.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted on training data yet.")

        transformed_cols = []
        for col in self.feature_columns:
            if col not in X.columns:
                if not fill_missing:
                    raise ValueError(f"Missing required feature column '{col}' during inference.")
                # Impute with frozen training median
                series = pd.Series([self.medians_[col]] * len(X))
            else:
                series = pd.to_numeric(X[col], errors="coerce")
                if fill_missing:
                    series = series.fillna(self.medians_[col])

            # Standardize using training mean and standard deviation
            scaled = (series.values - self.means_[col]) / self.stds_[col]
            transformed_cols.append(scaled.reshape(-1, 1))

        return np.hstack(transformed_cols)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def check_out_of_distribution(self, X: pd.DataFrame, threshold_std: float = 4.0) -> List[Dict[str, Any]]:
        """
        Identifies whether any patient observation falls beyond acceptable clinical distribution bounds.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to evaluate out-of-distribution instances.")

        ood_flags = []
        for col in self.feature_columns:
            if col in X.columns:
                val = pd.to_numeric(X[col], errors="coerce").values[0]
                if pd.notna(val):
                    z_score = abs(val - self.means_[col]) / self.stds_[col]
                    if z_score > threshold_std:
                        ood_flags.append({
                            "feature": col,
                            "observed_value": float(val),
                            "training_mean": self.means_[col],
                            "training_std": self.stds_[col],
                            "z_score": float(z_score),
                            "reason": f"Value deviates by {z_score:.1f} standard deviations from validated baseline."
                        })
        return ood_flags
