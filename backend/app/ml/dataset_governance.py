"""
Dataset Governance and Cohort Splitting Layer.
Enforces dataset provenance, feature lineage, synthetic vs real data governance,
and patient-level split isolation to eliminate train-test data leakage.
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    version: str
    is_synthetic: bool
    data_origin: str
    governance_approval: str
    total_records: int
    total_unique_patients: int
    features: List[str]
    target_column: str


class DatasetGovernance:
    """
    Manages clinical ML dataset metadata, compliance constraints, and split isolation.
    """

    @staticmethod
    def split_cohort_by_patient(
        df: pd.DataFrame,
        patient_id_col: str,
        test_ratio: float = 0.20,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a dataset into train and test sets strictly at the unique patient level.
        Prevents data leakage where observations from the same patient appear in both train and test.
        """
        if patient_id_col not in df.columns:
            raise ValueError(f"Patient ID column '{patient_id_col}' not found in dataframe.")

        unique_patients = df[patient_id_col].drop_duplicates().values
        rng = np.random.RandomState(random_seed)
        shuffled_patients = rng.permutation(unique_patients)

        n_test = int(len(shuffled_patients) * test_ratio)
        test_patients = set(shuffled_patients[:n_test])
        train_patients = set(shuffled_patients[n_test:])

        train_df = df[df[patient_id_col].isin(train_patients)].copy().reset_index(drop=True)
        test_df = df[df[patient_id_col].isin(test_patients)].copy().reset_index(drop=True)

        # Invariant check: zero overlap
        overlap = set(train_df[patient_id_col]).intersection(set(test_df[patient_id_col]))
        if len(overlap) > 0:
            raise RuntimeError(f"Data leakage detected! {len(overlap)} patients overlap between train and test splits.")

        return train_df, test_df

    @staticmethod
    def audit_synthetic_data_usage(metadata: DatasetMetadata) -> Dict[str, Any]:
        """
        Audits whether a dataset is authorized for clinical production vs research only.
        Synthetic datasets are strictly prohibited from driving production clinical recommendations.
        """
        if metadata.is_synthetic:
            return {
                "authorized_for_clinical_inference": False,
                "regulatory_clearance": "DENIED - SYNTHETIC_DATA_ONLY",
                "recommended_environment": "RESEARCH_AND_UNIT_TESTING_ONLY",
                "quarantine_reason": "Synthetic datasets lack physiological covariance validation and cannot be used for clinical decision support."
            }

        return {
            "authorized_for_clinical_inference": True,
            "regulatory_clearance": metadata.governance_approval,
            "recommended_environment": "PRODUCTION_CLINICAL_DECISION_SUPPORT"
        }
