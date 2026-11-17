from backend.app.models.base import BaseDBModel
from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.consent import ConsentRecord, ConsentStatus, ConsentPurpose
from backend.app.models.screening import ScreeningEncounter, EpistemicStatus, EncounterType
from backend.app.models.audit import AuditLog, AuditOutcome
from backend.app.models.person import (
    Person,
    PersonIdentifier,
    PatientProfile,
    Household,
    HouseholdMember,
    Gender,
    RecordStatus,
    PopulationClassification,
    CompletenessStatus,
    IdentifierNamespace,
    IdentifierVerificationStatus,
)
from backend.app.models.provenance import (
    DataSource,
    DataProvenance,
    SourceType,
    TrustLevel,
    ConnectionStatus,
    ProvenanceVerificationStatus,
)
from backend.app.models.clinical import (
    Encounter,
    ClinicalObservation,
    ClinicalCondition,
    FamilyHistory,
    ScreeningRecord,
    ClinicalDocument,
    EncounterStatus,
    ObservationType,
    ConditionClinicalStatus,
    ConditionVerificationStatus,
    FamilyRelationship,
    ScreeningStatus,
)

from backend.app.models.assessment import (
    PreventiveAssessment,
    AssessmentStatus,
    ClinicalRiskTier,
    HumanReviewStatus,
)
from backend.app.models.genomics import (
    GenomicFinding,
    VariantClassification,
    GenomicReviewStatus,
)

__all__ = [
    "BaseDBModel",
    "Organization",
    "User",
    "UserRole",
    "Guardianship",
    "GuardianshipStatus",
    "RelationshipType",
    "ConsentRecord",
    "ConsentStatus",
    "ConsentPurpose",
    "ScreeningEncounter",
    "EpistemicStatus",
    "EncounterType",
    "AuditLog",
    "AuditOutcome",
    # Assessment
    "PreventiveAssessment",
    "AssessmentStatus",
    "ClinicalRiskTier",
    "HumanReviewStatus",
    # Genomics
    "GenomicFinding",
    "VariantClassification",
    "GenomicReviewStatus",
    # Person & Demographics
    "Person",
    "PersonIdentifier",
    "PatientProfile",
    "Household",
    "HouseholdMember",
    "Gender",
    "RecordStatus",
    "PopulationClassification",
    "CompletenessStatus",
    "IdentifierNamespace",
    "IdentifierVerificationStatus",
    # Provenance
    "DataSource",
    "DataProvenance",
    "SourceType",
    "TrustLevel",
    "ConnectionStatus",
    "ProvenanceVerificationStatus",
    # Clinical
    "Encounter",
    "ClinicalObservation",
    "ClinicalCondition",
    "FamilyHistory",
    "ScreeningRecord",
    "ClinicalDocument",
    "EncounterStatus",
    "ObservationType",
    "ConditionClinicalStatus",
    "ConditionVerificationStatus",
    "FamilyRelationship",
    "ScreeningStatus",
]
