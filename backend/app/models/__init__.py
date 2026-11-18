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
# Phase 9: Recommendations & Care Plans
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationReview,
    PreventiveCarePlan,
    CarePlanItem,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)
# Phase 10: Cultural Outreach
from backend.app.models.outreach import (
    CommunicationPreference,
    OutreachTask,
    InAppNotification,
    CommunicationChannel,
    OutreachAuthorizationStatus,
    OutreachDeliveryStatus,
)
# Phase 11: Care Navigation
from backend.app.models.navigation import (
    Appointment,
    Referral,
    FollowUpTask,
    RecallTask,
    CareGapClosureEvidence,
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
    RecallStatus,
)
# Phase 12: Population Health Intelligence
from backend.app.models.population import (
    PopulationMetricSnapshot,
    AnalyticsAccessLog,
    MetricCategory,
)
# Phase 13: Offline-First CHW
from backend.app.models.offline import (
    DeviceRegistration,
    OfflineAssignment,
    SyncOperationAudit,
    SyncConflict,
    DeviceStatus,
    OfflineAssignmentStatus,
    SyncOperationStatus,
    ConflictResolutionStatus,
)
# Phase 14: Healthcare Interoperability
from backend.app.models.integration import (
    IntegrationProvider,
    IntegrationJobLog,
    ProvenanceRecord,
    ProviderType,
    IntegrationStatus,
    JobExecutionStatus,
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
    # Phase 9
    "PreventiveRecommendation",
    "RecommendationReview",
    "PreventiveCarePlan",
    "CarePlanItem",
    "RecommendationCategory",
    "RecommendationPriority",
    "RecommendationLifecycleStatus",
    # Phase 10
    "CommunicationPreference",
    "OutreachTask",
    "InAppNotification",
    "CommunicationChannel",
    "OutreachAuthorizationStatus",
    "OutreachDeliveryStatus",
    # Phase 11
    "Appointment",
    "Referral",
    "FollowUpTask",
    "RecallTask",
    "CareGapClosureEvidence",
    "AppointmentStatus",
    "ReferralStatus",
    "FollowUpStatus",
    "RecallStatus",
    # Phase 12
    "PopulationMetricSnapshot",
    "AnalyticsAccessLog",
    "MetricCategory",
    # Phase 13
    "DeviceRegistration",
    "OfflineAssignment",
    "SyncOperationAudit",
    "SyncConflict",
    "DeviceStatus",
    "OfflineAssignmentStatus",
    "SyncOperationStatus",
    "ConflictResolutionStatus",
    # Phase 14
    "IntegrationProvider",
    "IntegrationJobLog",
    "ProvenanceRecord",
    "ProviderType",
    "IntegrationStatus",
    "JobExecutionStatus",
]
