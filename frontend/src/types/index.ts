export type UserRole =
  | 'INDIVIDUAL'
  | 'PARENT_GUARDIAN'
  | 'COMMUNITY_HEALTH_WORKER'
  | 'HEALTHCARE_PROVIDER'
  | 'CARE_COORDINATOR'
  | 'POPULATION_HEALTH_ADMIN'
  | 'SYSTEM_ADMIN'
  | 'DATA_GOVERNANCE_OFFICER';

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  phone_number?: string;
  organization_id?: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token: string;
  user_id: string;
  role: UserRole;
  full_name: string;
  organization_id?: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
  jurisdiction: string;
  database: string;
  timestamp: string;
}

export type EpistemicStatus =
  | 'VERIFIED'
  | 'UNVERIFIED'
  | 'MISSING'
  | 'UNKNOWN'
  | 'NOT_APPLICABLE'
  | 'OUTDATED'
  | 'CONFLICTING'
  | 'UNAUTHORIZED';

export interface ScreeningEncounter {
  id: string;
  patient_id: string;
  provider_id?: string;
  facility_id: string;
  encounter_type: string;
  encounter_date: string;
  systolic_bp?: number;
  diastolic_bp?: number;
  pulse_rate?: number;
  fasting_glucose?: number;
  height_cm?: number;
  weight_kg?: number;
  bmi?: number;
  epistemic_status: EpistemicStatus;
  notes?: string;
  created_at: string;
}

export interface ConsentRecord {
  id: string;
  user_id: string;
  authorized_representative_id?: string;
  purpose: string;
  data_categories: string[];
  status: 'ACTIVE' | 'REVOKED' | 'EXPIRED';
  consent_version: string;
  collection_source: string;
  valid_from: string;
  valid_until?: string;
  revoked_at?: string;
  created_at: string;
}

export interface AuditLog {
  id: string;
  actor_user_id?: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  organization_id?: string;
  outcome: 'SUCCESS' | 'FAILURE' | 'DENIED';
  ip_address?: string;
  correlation_id: string;
  details?: string;
  created_at: string;
}

// ==========================================
// Phase 4 & 5: Patient Intelligence Types
// ==========================================
export type PopulationClassification = 'PEDIATRIC' | 'ADULT' | 'GERIATRIC';
export type Gender = 'MALE' | 'FEMALE' | 'OTHER' | 'UNKNOWN';

export interface PersonSummary {
  id: string;
  first_name: string;
  last_name: string;
  full_name: string;
  date_of_birth: string;
  gender: Gender;
  age_years: number;
  age_months: number;
  population_classification: PopulationClassification;
  completeness_score: number;
  completeness_status: string;
  emirate: string;
  district?: string;
  preferred_language: string;
  primary_identifier_masked?: string;
  is_high_risk: boolean;
  created_at: string;
}

export interface PersonListResponse {
  items: PersonSummary[];
  total: number;
  skip: number;
  limit: number;
}

export interface PersonCreate {
  first_name: string;
  last_name: string;
  date_of_birth: string;
  gender: Gender;
  preferred_language?: string;
  nationality?: string;
  emirate?: string;
  district?: string;
  organization_id?: string;
  identifier_namespace?: string;
  identifier_value?: string;
  emirates_id?: string;
  mrn?: string;
  guardian_user_id?: string;
  relationship_to_guardian?: string;
}

export interface ObservationCreate {
  observation_type: string;
  code: string;
  display_name: string;
  numeric_value?: number;
  unit?: string;
  string_value?: string;
  collection_time?: string;
}

export interface TimelineEvent {
  event_id: string;
  event_type: 'ENCOUNTER' | 'OBSERVATION' | 'CONDITION' | 'SCREENING';
  title: string;
  description: string;
  timestamp: string;
  source_name: string;
  epistemic_status: EpistemicStatus;
  value_display?: string;
}

export interface TimelineResponse {
  patient_id: string;
  total_events: number;
  events: TimelineEvent[];
}

export interface PediatricProfile {
  summary: PersonSummary;
  age_months: number;
  milestones_due_count: number;
  well_child_visit_count: number;
  last_well_child_visit?: string;
  guardian_info?: {
    guardian_user_id: string;
    relationship: string;
    status: string;
  };
  growth_observations: {
    height_cm?: number;
    weight_kg?: number;
    pulse_rate?: number;
    last_recorded?: string;
  };
}

export interface AdultProfile {
  summary: PersonSummary;
  age_years: number;
  bmi?: number;
  blood_pressure?: {
    systolic?: number;
    diastolic?: number;
    recorded_at?: string;
  };
  fasting_glucose_mg_dl?: number;
  active_conditions: Array<{
    code: string;
    description: string;
    status: string;
    onset_date?: string;
  }>;
  family_history: Array<{
    condition: string;
    relationship: string;
    onset_age?: number;
  }>;
  screening_adherence: {
    last_ifhas_date?: string;
    status: string;
  };
}

// ==========================================
// Phase 6: Abu Dhabi Pathway Types
// ==========================================
export interface PathwaySummary {
  pathway_id: string;
  pathway_name: string;
  version: string;
  authoritative_body: string;
  guideline_reference: string;
  target_population: string;
  cycle_interval_months: number;
  total_milestones: number;
}

export interface CareGapItem {
  milestone_id: string;
  name: string;
  status: 'COMPLETED' | 'DUE' | 'OVERDUE' | 'POTENTIALLY_OVERDUE' | 'NEEDS_DATA' | 'REQUIRES_CLINICAL_REVIEW' | 'NOT_APPLICABLE';
  category: string;
  due_date: string;
  grace_period_days: number;
  clinical_guideline: string;
  screening_code?: string;
  last_completed_date?: string;
  next_due_date?: string;
  reason?: string;
}

export interface ComprehensiveEvaluation {
  patient_id: string;
  pathway_id: string;
  pathway_name: string;
  evaluation_date: string;
  eligibility: {
    is_eligible: boolean;
    reasons: string[];
    criteria_evaluated: {
      patient_age_months: number;
      patient_age_years: number;
      patient_gender: string;
    };
  };
  care_gaps: CareGapItem[];
}

// ==========================================
// Phase 7: Preventive Assessment Types
// ==========================================
export type ClinicalRiskTier = 'LOW_RISK' | 'MODERATE_RISK' | 'ELEVATED_RISK' | 'INSUFFICIENT_DATA';
export type AssessmentStatus = 'COMPLETED' | 'REQUIRES_REVIEW' | 'NEEDS_DATA' | 'SUPERSEDED';
export type HumanReviewStatus = 'NOT_REQUIRED' | 'AWAITING_REVIEW' | 'UNDER_REVIEW' | 'APPROVED' | 'MODIFIED' | 'REJECTED';

export interface PreventiveAssessmentResponse {
  id?: string;
  assessment_id?: string;
  patient_id: string;
  assessment_type?: string;
  population_category: string;
  pathway_id: string;
  pathway_version: string;
  assessment_status: AssessmentStatus;
  clinical_risk_tier: ClinicalRiskTier;
  eligibility_summary?: string;
  care_gaps_summary?: string;
  contributing_factors?: any;
  data_quality_limitations?: any;
  eligibility?: any;
  care_gaps?: any;
  human_review_status: HumanReviewStatus;
  reviewed_by_user_id?: string;
  reviewed_at?: string;
  reviewer_notes?: string;
  recommended_action: string;
  created_at: string;
  updated_at?: string;
}

// ==========================================
// Phase 8: Genomic Intelligence Types
// ==========================================
export type VariantClassification = 'PATHOGENIC' | 'LIKELY_PATHOGENIC' | 'VUS' | 'LIKELY_BENIGN' | 'BENIGN';
export type GenomicReviewStatus = 'PENDING_REVIEW' | 'GENETIC_COUNSELOR_VERIFIED' | 'REJECTED';

export interface GenomicFindingResponse {
  id: string;
  patient_id: string;
  gene_symbol: string;
  variant_id: string;
  genomic_coordinates?: string;
  variant_classification: VariantClassification;
  clinical_interpretation: string;
  actionable_guidance?: string;
  source_laboratory: string;
  report_reference?: string;
  review_status: GenomicReviewStatus;
  reviewed_by_user_id?: string;
  reviewed_at?: string;
  reviewer_notes?: string;
  created_at: string;
}
