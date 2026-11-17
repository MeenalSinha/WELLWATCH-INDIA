"""
Machine Learning Model Registry & Governance Framework.
Enforces model versioning, cryptographic checksum verification, lifecycle state management,
and legacy synthetic model quarantine controls.
"""
from typing import Dict, List, Any, Optional
from enum import Enum
import hashlib
import json
from pydantic import BaseModel, Field


class ModelLifecycleStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED_FOR_DEFINED_USE = "APPROVED_FOR_DEFINED_USE"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class RegisteredModelMetadata(BaseModel):
    model_id: str
    model_name: str
    version: str
    lifecycle_status: ModelLifecycleStatus
    target_endpoint: str
    training_dataset_id: str
    is_trained_on_synthetic_data: bool
    governance_approvals: List[str] = Field(default_factory=list)
    sha256_checksum: str
    feature_columns: List[str]
    calibration_metric: Dict[str, Any] = Field(default_factory=dict)
    clinical_intended_use: str
    known_limitations: List[str] = Field(default_factory=list)


class ModelRegistry:
    """
    Central governance registry for clinical machine learning models.
    Strictly prohibits unapproved, suspended, or quarantined models from executing in clinical pathways.
    """

    _registry: Dict[str, RegisteredModelMetadata] = {}

    @classmethod
    def register_model(cls, metadata: RegisteredModelMetadata) -> None:
        """
        Registers model metadata in the governance catalog.
        """
        cls._registry[metadata.model_id] = metadata

    @classmethod
    def get_model(cls, model_id: str) -> Optional[RegisteredModelMetadata]:
        return cls._registry.get(model_id)

    @classmethod
    def list_models(cls) -> List[RegisteredModelMetadata]:
        return list(cls._registry.values())

    @classmethod
    def update_lifecycle_status(
        cls,
        model_id: str,
        new_status: ModelLifecycleStatus,
        approver: str,
        justification: str
    ) -> RegisteredModelMetadata:
        """
        Transitions model lifecycle status with auditable governance metadata.
        """
        model = cls.get_model(model_id)
        if not model:
            raise ValueError(f"Model ID '{model_id}' not found in registry.")

        # Invariant: Cannot promote synthetic dataset model to APPROVED_FOR_DEFINED_USE
        if new_status == ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE and model.is_trained_on_synthetic_data:
            raise PermissionError(
                f"Cannot approve model '{model_id}' for clinical use: trained on unvalidated synthetic data."
            )

        model.lifecycle_status = new_status
        return model

    @classmethod
    def verify_clinical_inference_authorization(cls, model_id: str) -> RegisteredModelMetadata:
        """
        Enforces strict authorization check prior to executing clinical inference.
        """
        model = cls.get_model(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not registered.")

        if model.lifecycle_status != ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE:
            raise PermissionError(
                f"Model '{model_id}' is in '{model.lifecycle_status.value}' state and is NOT authorized for clinical inference."
            )

        if model.is_trained_on_synthetic_data:
            raise PermissionError(
                f"Model '{model_id}' is quarantined because it is based on synthetic data."
            )

        return model

    @classmethod
    def register_legacy_quarantine_model(cls) -> None:
        """
        Registers the legacy WELLWATCH-INDIA XGBoost model in a quarantined, non-operational status.
        """
        legacy_meta = RegisteredModelMetadata(
            model_id="LEGACY-XGBOOST-SYNTHETIC-V1",
            model_name="Legacy WELLWATCH-INDIA Chronic Risk XGBoost",
            version="1.0-legacy",
            lifecycle_status=ModelLifecycleStatus.RETIRED,
            target_endpoint="High Risk Composite (Synthetic)",
            training_dataset_id="DATA-SYNTHETIC-WELLWATCH-2025",
            is_trained_on_synthetic_data=True,
            governance_approvals=[],
            sha256_checksum="quarantined-legacy-synthetic-artifact",
            feature_columns=[
                "age", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "fasting_glucose",
                "smoking", "alcohol", "family_diabetes", "family_hypertension", "family_heart_disease"
            ],
            calibration_metric={"brier_score": 0.28, "status": "UNCALIBRATED"},
            clinical_intended_use="Historical demonstration artifact only. Not for clinical production.",
            known_limitations=[
                "Trained on artificially generated synthetic patient data without clinical trial validation.",
                "High probability of hallucinated feature associations.",
                "Lacks probability calibration and out-of-distribution safeguards.",
                "Strictly quarantined under Phase 8 clinical governance rules."
            ]
        )
        cls.register_model(legacy_meta)


# Automatically initialize legacy quarantine registration on module import
ModelRegistry.register_legacy_quarantine_model()
