"""
Unit and Integration Test Suite for Phase 8 Machine Learning Subsystem.
Verifies dataset governance, leakage-resistant preprocessing, probability calibration,
feature explainability, model registry lifecycle, and controlled inference abstention.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backend.app.ml.dataset_governance import DatasetGovernance, DatasetMetadata
from backend.app.ml.preprocessing import LeakageResistantPreprocessor
from backend.app.ml.calibration import ProbabilityCalibrator
from backend.app.ml.explainability import FeatureAttributionExplainer
from backend.app.ml.model_registry import (
    ModelRegistry,
    RegisteredModelMetadata,
    ModelLifecycleStatus,
)
from backend.app.ml.inference_service import ControlledInferenceService


def test_dataset_split_patient_level_isolation():
    """
    Verifies that dataset splitting operates strictly at the unique patient level,
    guaranteeing zero patient overlap between train and test cohorts.
    """
    data = {
        "patient_id": [f"P{i:03d}" for i in range(100)] * 3,  # 100 unique patients, 3 encounters each
        "systolic_bp": np.random.uniform(110, 160, 300),
        "fasting_glucose": np.random.uniform(80, 180, 300),
        "target": np.random.choice([0, 1], size=300)
    }
    df = pd.DataFrame(data)

    train_df, test_df = DatasetGovernance.split_cohort_by_patient(
        df=df,
        patient_id_col="patient_id",
        test_ratio=0.20,
        random_seed=42
    )

    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    # Assert zero overlap between train and test patients
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(train_patients) == 80
    assert len(test_patients) == 20


def test_synthetic_dataset_governance_audit():
    """
    Verifies that synthetic datasets are explicitly prohibited from clinical deployment clearance.
    """
    synth_meta = DatasetMetadata(
        dataset_id="SYNTH-001",
        name="Legacy Synthetic Cohort",
        version="1.0",
        is_synthetic=True,
        data_origin="Artificially generated CSV",
        governance_approval="NONE",
        total_records=2000,
        total_unique_patients=2000,
        features=["age", "bmi", "systolic_bp"],
        target_column="risk_flag"
    )

    audit = DatasetGovernance.audit_synthetic_data_usage(synth_meta)
    assert audit["authorized_for_clinical_inference"] is False
    assert "DENIED" in audit["regulatory_clearance"]
    assert "RESEARCH_AND_UNIT_TESTING_ONLY" in audit["recommended_environment"]


def test_leakage_resistant_preprocessing_frozen_parameters():
    """
    Verifies that scalers and imputers are strictly learned from train data
    and applied without modification to test data.
    """
    train_data = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "systolic_bp": [110, 120, 130, 140, 150]
    })
    test_data = pd.DataFrame({
        "age": [25, 45],
        "systolic_bp": [115, 135]
    })

    preprocessor = LeakageResistantPreprocessor(feature_columns=["age", "systolic_bp"])
    preprocessor.fit(train_data)

    # Check learned stats match train exactly
    assert preprocessor.means_["age"] == 40.0
    assert preprocessor.medians_["systolic_bp"] == 130.0

    # Transform test data
    X_test_scaled = preprocessor.transform(test_data)
    assert X_test_scaled.shape == (2, 2)
    # Check that age 40 in test would scale to exactly 0.0
    scaled_40 = (40.0 - preprocessor.means_["age"]) / preprocessor.stds_["age"]
    assert scaled_40 == 0.0


def test_probability_calibration_and_metrics():
    """
    Verifies that Platt scaling fits on validation predictions and produces
    valid probabilities with Brier and ECE calibration metrics.
    """
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    raw_scores = np.concatenate([
        np.random.normal(loc=-1.0, scale=0.8, size=50),
        np.random.normal(loc=1.2, scale=0.8, size=50)
    ])

    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(raw_scores, y_true)

    probs = calibrator.predict_proba(raw_scores)
    assert len(probs) == 100
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    metrics = ProbabilityCalibrator.compute_calibration_metrics(y_true, probs)
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert "auc_roc" in metrics
    assert metrics["brier_score"] < 0.25  # Well-calibrated model has lower Brier score


def test_feature_explainability_attribution():
    """
    Verifies that local feature attribution accurately breaks down predictions
    and ranks top risk-increasing and protective factors.
    """
    feature_names = ["age", "bmi", "systolic_bp", "fasting_glucose"]
    feature_values = np.array([1.5, 2.0, -0.5, 0.0])  # Scaled inputs
    weights = np.array([0.4, 0.6, 0.5, 0.3])

    explanation = FeatureAttributionExplainer.explain_prediction(
        feature_names=feature_names,
        feature_values=feature_values,
        weights=weights,
        base_value=0.1,
        top_k=2
    )

    assert "top_risk_increasing_factors" in explanation
    assert "top_protective_factors" in explanation
    top_risk = explanation["top_risk_increasing_factors"]
    assert len(top_risk) > 0
    # BMI has highest positive contribution (2.0 * 0.6 = 1.2)
    assert top_risk[0]["feature"] == "bmi"
    assert top_risk[0]["impact"] == "RISK_INCREASING"

    # Systolic BP has negative value (-0.5 * 0.5 = -0.25)
    top_prot = explanation["top_protective_factors"]
    assert len(top_prot) > 0
    assert top_prot[0]["feature"] == "systolic_bp"


def test_model_registry_lifecycle_and_quarantine():
    """
    Verifies model lifecycle state enforcement and legacy synthetic quarantine.
    """
    # 1. Verify legacy model is permanently quarantined
    legacy = ModelRegistry.get_model("LEGACY-XGBOOST-SYNTHETIC-V1")
    assert legacy is not None
    assert legacy.lifecycle_status == ModelLifecycleStatus.RETIRED
    assert legacy.is_trained_on_synthetic_data is True

    # Attempting to run clinical inference with legacy quarantined model must be blocked
    with pytest.raises(PermissionError) as exc_info:
        ModelRegistry.verify_clinical_inference_authorization("LEGACY-XGBOOST-SYNTHETIC-V1")
    assert "quarantined" in str(exc_info.value) or "RETIRED" in str(exc_info.value)

    # 2. Register new experimental model
    new_model = RegisteredModelMetadata(
        model_id="CARDIOMETABOLIC-LOGISTIC-V1",
        model_name="Cardiometabolic Risk Predictor",
        version="1.0",
        lifecycle_status=ModelLifecycleStatus.EXPERIMENTAL,
        target_endpoint="CARDIOMETABOLIC_ONSET_3YR",
        training_dataset_id="VALIDATED-EHR-COHORT-2026",
        is_trained_on_synthetic_data=False,
        governance_approvals=["DOH-AI-RESEARCH-09"],
        sha256_checksum="abc123sha256checksum",
        feature_columns=["age", "bmi", "systolic_bp", "fasting_glucose"],
        clinical_intended_use="Cardiometabolic risk assessment for periodic IFHAS screening.",
        known_limitations=["Valid for adults aged 18-70."]
    )
    ModelRegistry.register_model(new_model)

    # Experimental cannot be used for clinical inference
    with pytest.raises(PermissionError):
        ModelRegistry.verify_clinical_inference_authorization("CARDIOMETABOLIC-LOGISTIC-V1")

    # Transition to APPROVED_FOR_DEFINED_USE
    ModelRegistry.update_lifecycle_status(
        model_id="CARDIOMETABOLIC-LOGISTIC-V1",
        new_status=ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE,
        approver="Clinical AI Governance Committee",
        justification="Completed Phase 8 calibration and clinical safety review."
    )

    approved = ModelRegistry.verify_clinical_inference_authorization("CARDIOMETABOLIC-LOGISTIC-V1")
    assert approved.lifecycle_status == ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE


def test_controlled_inference_service_execution_and_abstention():
    """
    Verifies that ControlledInferenceService executes authorized models,
    decomposes explanations, and abstains when missing features or out-of-bounds inputs occur.
    """
    feature_cols = ["age", "bmi", "systolic_bp", "fasting_glucose"]
    train_df = pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "bmi": [22.0, 26.0, 28.0, 31.0, 34.0],
        "systolic_bp": [115, 122, 130, 138, 148],
        "fasting_glucose": [85, 92, 105, 118, 135]
    })
    preprocessor = LeakageResistantPreprocessor(feature_columns=feature_cols)
    preprocessor.fit(train_df)

    # Train a simple linear estimator
    X_train = preprocessor.transform(train_df)
    y_train = np.array([0, 0, 0, 1, 1])
    estimator = LogisticRegression()
    estimator.fit(X_train, y_train)

    calibrator = ProbabilityCalibrator()
    calibrator.fit(estimator.predict_proba(X_train)[:, 1], y_train)

    # Scenario A: Valid patient features -> successful prediction with explainability
    patient_valid = {
        "age": 52,
        "bmi": 32.5,
        "systolic_bp": 142.0,
        "fasting_glucose": 120.0
    }
    result = ControlledInferenceService.execute_inference(
        model_id="CARDIOMETABOLIC-LOGISTIC-V1",
        patient_features=patient_valid,
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=calibrator
    )

    assert result["status"] == "COMPLETED"
    assert result["calibrated_risk_probability"] is not None
    assert 0.0 <= result["calibrated_risk_probability"] <= 1.0
    assert result["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "ELEVATED_RISK"]
    assert "explainability" in result
    assert len(result["explainability"]["top_risk_increasing_factors"]) > 0

    # Scenario B: Missing critical features (>30% missing) -> Abstains with INSUFFICIENT_DATA
    patient_missing = {
        "age": 45,
        # bmi missing, systolic_bp missing, fasting_glucose missing (75% missing)
    }
    result_missing = ControlledInferenceService.execute_inference(
        model_id="CARDIOMETABOLIC-LOGISTIC-V1",
        patient_features=patient_missing,
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=calibrator
    )
    assert result_missing["status"] == "ABSTAINED_INSUFFICIENT_DATA"
    assert result_missing["risk_tier"] == "INSUFFICIENT_DATA"
    assert len(result_missing["missing_features"]) >= 3

    # Scenario C: Extreme physiological out-of-distribution value (systolic BP = 290 mmHg) -> Abstains with OOD
    patient_ood = {
        "age": 45,
        "bmi": 24.0,
        "systolic_bp": 290.0,  # Extreme outlier > 4 std deviations
        "fasting_glucose": 95.0
    }
    result_ood = ControlledInferenceService.execute_inference(
        model_id="CARDIOMETABOLIC-LOGISTIC-V1",
        patient_features=patient_ood,
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=calibrator
    )
    assert result_ood["status"] == "ABSTAINED_OUT_OF_DISTRIBUTION"
    assert result_ood["risk_tier"] == "UNCERTAIN_OUT_OF_BOUNDS"
    assert len(result_ood["ood_findings"]) > 0
