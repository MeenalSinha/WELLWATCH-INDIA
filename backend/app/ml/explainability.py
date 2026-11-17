"""
Machine Learning Feature Attribution & Explainability Framework.
Decomposes patient-level predictions into transparent, weighted feature contributions.
"""
from typing import Dict, List, Any
import numpy as np


class FeatureAttributionExplainer:
    """
    Computes local feature contributions for explainable clinical machine learning.
    """

    FEATURE_CLINICAL_NAMES = {
        "age": "Chronological Age",
        "bmi": "Body Mass Index",
        "systolic_bp": "Systolic Blood Pressure",
        "diastolic_bp": "Diastolic Blood Pressure",
        "fasting_glucose": "Fasting Blood Glucose",
        "pulse_rate": "Resting Pulse Rate",
        "family_diabetes": "Family History of Diabetes",
        "family_hypertension": "Family History of Hypertension",
        "family_heart_disease": "Family History of Cardiovascular Disease",
        "smoking": "Current Tobacco Use",
        "alcohol": "Alcohol Consumption"
    }

    @classmethod
    def explain_prediction(
        cls,
        feature_names: List[str],
        feature_values: np.ndarray,
        weights: np.ndarray,
        base_value: float = 0.0,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Decomposes a prediction into additive feature contributions.
        contribution_i = weight_i * feature_value_i
        """
        feats = np.asarray(feature_values).flatten()
        w = np.asarray(weights).flatten()

        contributions = feats * w
        total_score = base_value + float(np.sum(contributions))

        items = []
        for i, name in enumerate(feature_names):
            c_val = float(contributions[i])
            items.append({
                "feature": name,
                "display_name": cls.FEATURE_CLINICAL_NAMES.get(name, name.replace("_", " ").title()),
                "observed_value": round(float(feats[i]), 2),
                "contribution": round(c_val, 4),
                "impact": "RISK_INCREASING" if c_val > 0 else "PROTECTIVE_OR_NEUTRAL"
            })

        # Sort by absolute magnitude of contribution
        items_sorted = sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "base_value": round(base_value, 4),
            "total_score": round(total_score, 4),
            "top_risk_increasing_factors": [item for item in items_sorted if item["contribution"] > 0][:top_k],
            "top_protective_factors": [item for item in items_sorted if item["contribution"] < 0][:top_k],
            "all_attributions": items_sorted
        }
