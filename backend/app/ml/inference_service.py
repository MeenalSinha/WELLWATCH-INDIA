"""
Controlled Clinical Machine Learning Inference Service.
Orchestrates model authorization verification, out-of-distribution safeguards,
algorithmic abstention, calibrated probability estimation, and explainability attributions.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.ml.model_registry import ModelRegistry, ModelLifecycleStatus
from backend.app.ml.preprocessing import LeakageResistantPreprocessor
from backend.app.ml.calibration import ProbabilityCalibrator
from backend.app.ml.explainability import FeatureAttributionExplainer


class ControlledInferenceService:
    """
    Executes governed clinical ML inference with strict safety and abstention guardrails.
    """

    @classmethod
    def execute_inference(
        cls,
        model_id: str,
        patient_features: Dict[str, Any],
        preprocessor: LeakageResistantPreprocessor,
        estimator: Any,
        calibrator: Optional[ProbabilityCalibrator] = None,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes a safe inference pipeline:
        1. Registry Authorization Check
        2. Input Schema & Missingness Check (Abstention on critical missing vitals)
        3. Out-Of-Distribution (OOD) Check
        4. Feature Transformation
        5. Calibrated Prediction
        6. Explainability Decomposition
        """
        # 1. Governance check
        metadata = ModelRegistry.verify_clinical_inference_authorization(model_id)

        # 2. Input validation & missingness check
        input_df = pd.DataFrame([patient_features])
        missing_critical = []
        for col in metadata.feature_columns:
            if col not in input_df.columns or pd.isna(input_df[col].iloc[0]):
                missing_critical.append(col)

        # If more than 30% of features missing, abstain
        missing_ratio = len(missing_critical) / max(len(metadata.feature_columns), 1)
        if missing_ratio > 0.30:
            return {
                "model_id": model_id,
                "status": "ABSTAINED_INSUFFICIENT_DATA",
                "calibrated_risk_probability": None,
                "risk_tier": "INSUFFICIENT_DATA",
                "missing_features": missing_critical,
                "recommendation": "Abstained from predictive modeling due to missing clinical features. Collect baseline vitals first."
            }

        # 3. Out-of-distribution check
        ood_issues = preprocessor.check_out_of_distribution(input_df)
        if len(ood_issues) > 0:
            return {
                "model_id": model_id,
                "status": "ABSTAINED_OUT_OF_DISTRIBUTION",
                "calibrated_risk_probability": None,
                "risk_tier": "UNCERTAIN_OUT_OF_BOUNDS",
                "ood_findings": ood_issues,
                "recommendation": "Observations fall outside validated physiological bounds. Mandatory clinical physician review required."
            }

        # 4. Transform features
        X_scaled = preprocessor.transform(input_df, fill_missing=True)

        # 5. Raw score & Calibrated probability
        if hasattr(estimator, "decision_function"):
            raw_score = estimator.decision_function(X_scaled)
        elif hasattr(estimator, "predict_proba"):
            raw_score = estimator.predict_proba(X_scaled)[:, 1]
        else:
            raw_score = estimator.predict(X_scaled)

        if calibrator and calibrator.is_fitted:
            calibrated_prob = float(calibrator.predict_proba(raw_score)[0])
        else:
            calibrated_prob = float(raw_score[0]) if isinstance(raw_score, np.ndarray) else float(raw_score)

        # Bound probability strictly in [0.0, 1.0]
        calibrated_prob = max(0.0, min(1.0, calibrated_prob))

        # Risk tier assignment based on calibrated probability
        if calibrated_prob >= 0.70:
            tier = "ELEVATED_RISK"
        elif calibrated_prob >= 0.35:
            tier = "MODERATE_RISK"
        else:
            tier = "LOW_RISK"

        # 6. Explainability attributions
        w = weights if weights is not None else getattr(estimator, "coef_", np.ones(len(metadata.feature_columns)))
        explanation = FeatureAttributionExplainer.explain_prediction(
            feature_names=metadata.feature_columns,
            feature_values=X_scaled[0],
            weights=w,
            top_k=3
        )

        return {
            "model_id": model_id,
            "model_version": metadata.version,
            "status": "COMPLETED",
            "calibrated_risk_probability": round(calibrated_prob, 4),
            "risk_tier": tier,
            "explainability": explanation,
            "target_endpoint": metadata.target_endpoint,
            "clinical_use": metadata.clinical_intended_use
        }
