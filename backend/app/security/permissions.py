import enum


class Permission(str, enum.Enum):
    # Individual & Dependent
    PROFILE_READ_OWN = "profile:read_own"
    PROFILE_UPDATE_OWN = "profile:update_own"
    DEPENDENT_READ = "dependent:read"

    # Clinical Screening
    SCREENING_CREATE = "screening:create"
    SCREENING_READ_SCOPED = "screening:read_scoped"

    # Clinical Review & Approval
    CLINICAL_REVIEW = "clinical:review"
    PATHWAY_APPROVE = "pathway:approve"

    # Care Coordination
    OUTREACH_MANAGE = "outreach:manage"
    GAP_RESOLVE = "gap:resolve"

    # Population Health Intelligence
    ANALYTICS_POPULATION = "analytics:population"

    # Administration
    USER_MANAGE = "user:manage"
    ORGANIZATION_MANAGE = "organization:manage"

    # Data Governance & Privacy
    AUDIT_READ = "audit:read"
    CONSENT_AUDIT = "consent:audit"
