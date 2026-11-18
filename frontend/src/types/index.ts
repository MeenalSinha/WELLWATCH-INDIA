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

// ==========================================
// Phase 9: Recommendation & Care Plan Types
// ==========================================
export type RecommendationCategory =
  | 'WELL_CHILD_VISIT'
  | 'DEVELOPMENTAL_SCREENING'
  | 'ADULT_COMPREHENSIVE_SCREENING'
  | 'PREVENTIVE_LABORATORY_SCREENING'
  | 'CARDIOMETABOLIC_SCREENING'
  | 'FAMILY_HISTORY_REVIEW'
  | 'GENOMIC_COUNSELING'
  | 'LIFESTYLE_WELLNESS'
  | 'SPECIALIST_REFERRAL'
  | 'OVERDUE_RECALL'
  | 'ADDITIONAL_INFO_COLLECTION'
  | 'CLINICAL_REASSESSMENT';

export type RecommendationPriority = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'ROUTINE';

export type RecommendationLifecycleStatus =
  | 'DRAFT'
  | 'PENDING_CLINICAL_REVIEW'
  | 'APPROVED'
  | 'MODIFIED'
  | 'OFFERED'
  | 'ACCEPTED'
  | 'DECLINED'
  | 'SCHEDULED'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'EXPIRED'
  | 'CANCELLED'
  | 'SUPERSEDED';

export interface PreventiveRecommendationResponse {
  id: string;
  organization_id: string;
  person_id: string;
  assessment_id?: string;
  pathway_id: string;
  care_gap_id?: string;
  category: RecommendationCategory;
  priority: RecommendationPriority;
  lifecycle_status: RecommendationLifecycleStatus;
  recommended_action: string;
  clinical_rationale: string;
  supporting_evidence: Record<string, any>;
  source_guideline: string;
  uncertainty_status: string;
  rule_id: string;
  rule_version: string;
  target_date?: string;
  expiration_date?: string;
  requires_human_review: boolean;
  declined_reason?: string;
  assigned_provider_id?: string;
  reviewer_notes?: string;
  created_at: string;
  updated_at: string;
}

export interface RecommendationListResponse {
  total: number;
  items: PreventiveRecommendationResponse[];
}

export interface RecommendationExplanation {
  recommendation_id: string;
  category: string;
  priority: string;
  clinical_rationale: string;
  source_guideline: string;
  trigger_rule_id: string;
  trigger_care_gap?: string;
  identified_evidence: Record<string, any>;
  uncertainty_status: string;
  requires_human_review: boolean;
  justification_summary: string;
}

export interface CarePlanItemResponse {
  id: string;
  care_plan_id: string;
  recommendation_id: string;
  status: string;
  target_date?: string;
  completed_at?: string;
  recommendation?: PreventiveRecommendationResponse;
}

export interface CarePlanResponse {
  id: string;
  organization_id: string;
  person_id: string;
  plan_name: string;
  pathway_id: string;
  status: string;
  goals: Array<{ goal: string; target?: string }>;
  responsible_org_id: string;
  assigned_care_team: Array<{ role: string; user_id?: string; name?: string }>;
  target_date?: string;
  review_schedule?: string;
  created_at: string;
  updated_at: string;
  items: CarePlanItemResponse[];
}

// ==========================================
// Phase 10: Cultural Outreach Types
// ==========================================
export type CommunicationChannel = 'IN_APP' | 'SMS' | 'EMAIL' | 'WHATSAPP';
export type OutreachDeliveryStatus =
  | 'DRAFT'
  | 'PENDING'
  | 'SCHEDULED'
  | 'SENDING'
  | 'SENT'
  | 'DELIVERED'
  | 'FAILED'
  | 'RETRY_PENDING'
  | 'CANCELLED'
  | 'SUPPRESSED'
  | 'EXPIRED';

export interface CommunicationPreferenceResponse {
  id: string;
  person_id: string;
  preferred_language: 'ar' | 'en';
  preferred_channel: CommunicationChannel;
  contact_window_start: string;
  contact_window_end: string;
  quiet_hours_start: string;
  quiet_hours_end: string;
  consent_opt_in: boolean;
  opt_out_timestamp?: string;
  guardian_proxy_required: boolean;
  accessibility_needs?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface OutreachTaskResponse {
  id: string;
  organization_id: string;
  person_id: string;
  recommendation_id?: string;
  recipient_id?: string;
  recipient_role: string;
  channel: CommunicationChannel;
  template_id: string;
  language: string;
  message_body: string;
  scheduled_at: string;
  attempted_at?: string;
  delivery_status: OutreachDeliveryStatus;
  provider_status: string;
  provider_message_id?: string;
  failure_reason?: string;
  retry_count: number;
  max_retries: number;
  consent_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface OutreachTaskListResponse {
  total: number;
  items: OutreachTaskResponse[];
}

export interface OutreachAnalyticsResponse {
  total_generated: number;
  total_attempted: number;
  total_delivered: number;
  total_failed: number;
  total_suppressed: number;
  by_channel: Record<string, number>;
  by_language: Record<string, number>;
  active_providers_count: number;
  honest_boundary_note: string;
}

export interface TemplatePreviewResponse {
  template_id: string;
  language: string;
  direction: 'rtl' | 'ltr';
  rendered_subject: string;
  rendered_body: string;
  governance_approved: boolean;
}

// ==========================================
// Phase 11: Care Navigation Types
// ==========================================
export type AppointmentStatus =
  | 'REQUESTED'
  | 'BOOKED'
  | 'CONFIRMED'
  | 'RESCHEDULED'
  | 'ATTENDED'
  | 'NO_SHOW'
  | 'CANCELLED'
  | 'COMPLETED';

export type ReferralStatus =
  | 'DRAFT'
  | 'SUBMITTED'
  | 'ACCEPTED'
  | 'SCHEDULED'
  | 'COMPLETED'
  | 'REJECTED'
  | 'CANCELLED';

export type FollowUpStatus = 'PENDING' | 'ASSIGNED' | 'IN_PROGRESS' | 'RESOLVED' | 'ESCALATED' | 'CANCELLED';

export interface AppointmentResponse {
  id: string;
  organization_id: string;
  person_id: string;
  recommendation_id?: string;
  care_plan_id?: string;
  service_type: string;
  scheduled_time: string;
  location: string;
  provider_id?: string;
  status: AppointmentStatus;
  attendance_notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ReferralResponse {
  id: string;
  organization_id: string;
  person_id: string;
  recommendation_id?: string;
  referring_provider_id: string;
  receiving_org_id?: string;
  specialty: string;
  clinical_indication: string;
  priority: string;
  status: ReferralStatus;
  outcome_notes?: string;
  created_at: string;
  updated_at: string;
}

export interface FollowUpTaskResponse {
  id: string;
  organization_id: string;
  person_id: string;
  recommendation_id?: string;
  appointment_id?: string;
  referral_id?: string;
  trigger_reason: string;
  assigned_role: string;
  assigned_user_id?: string;
  due_date: string;
  priority: string;
  status: FollowUpStatus;
  resolution_notes?: string;
  resolution_evidence_id?: string;
  created_at: string;
  updated_at: string;
}

export interface CareGapClosureResponse {
  id: string;
  organization_id: string;
  person_id: string;
  care_gap_id: string;
  recommendation_id?: string;
  evidence_source: string;
  evidence_id: string;
  verification_method: string;
  verifying_user_id: string;
  verification_notes?: string;
  verified_at: string;
  is_valid: boolean;
  status: string;
}

export interface NavigationAnalyticsResponse {
  total_appointments: number;
  attended_appointments: number;
  no_show_appointments: number;
  attendance_rate: number;
  total_referrals: number;
  completed_referrals: number;
  active_follow_up_tasks: number;
  closed_care_gaps_count: number;
  data_source_note: string;
}

// ==========================================
// PHASE 12: POPULATION HEALTH INTELLIGENCE
// ==========================================

export interface PopulationOverviewResponse {
  total_registered: number;
  pediatric_population: number;
  adult_population: number;
  geriatric_population: number;
  incomplete_profiles: number;
  eligible_population: number;
  completed_screenings: number;
  screening_coverage_percentage: number;
  confirmed_overdue: number;
  potentially_overdue: number;
  recommendations_generated: number;
  recommendations_approved: number;
  recommendations_accepted: number;
  recommendations_declined: number;
  recommendations_completed: number;
  appointments_booked: number;
  appointments_attended: number;
  no_show_rate: number;
  referrals_created: number;
  referrals_completed: number;
  care_gaps_closed: number;
  reporting_window_days: number;
  generated_at: string;
}

export interface PediatricPopulationResponse {
  total_pediatric: number;
  eligible_for_well_child: number;
  completed_well_child: number;
  well_child_coverage_percentage: number;
  overdue_well_child: number;
  developmental_stages: Record<string, number>;
  open_pediatric_care_gaps: number;
  guardian_authorized_rate: number;
  generated_at: string;
}

export interface AdultPopulationResponse {
  total_adult: number;
  eligible_for_ifhas: number;
  completed_ifhas: number;
  ifhas_coverage_percentage: number;
  overdue_ifhas: number;
  cardiometabolic_risk_distribution: Record<string, number>;
  open_adult_care_gaps: number;
  referral_completion_rate: number;
  generated_at: string;
}

export interface GeographicRegionMetric {
  region_code: string;
  region_name_en: string;
  region_name_ar: string;
  registered_population: number | string;
  eligible_population: number | string;
  completed_screenings: number | string;
  screening_coverage_percentage?: number | null;
  is_suppressed: boolean;
}

export interface GeographicAnalyticsResponse {
  regions: GeographicRegionMetric[];
  small_cell_threshold: number;
  suppression_applied: boolean;
  generated_at: string;
}

export interface PopulationEquityResponse {
  benchmark_region: string;
  subgroup_metrics: Array<Record<string, any>>;
  disparity_ratios: Record<string, number>;
  data_completeness_by_subgroup: Record<string, number>;
  ml_fairness_evaluation: Record<string, any>;
  generated_at: string;
}

export interface TrendPoint {
  period: string;
  completed_screenings: number;
  new_care_gaps: number;
  closed_care_gaps: number;
  screening_coverage_percentage: number;
}

export interface PopulationTrendsResponse {
  timeframe: string;
  trend_points: TrendPoint[];
  generated_at: string;
}

// ==========================================
// PHASE 13: OFFLINE-FIRST CHW APPLICATION
// ==========================================

export interface DeviceRegisterRequest {
  device_id: string;
  device_model?: string;
  os_platform?: string;
  app_version?: string;
}

export interface DeviceRegisterResponse {
  registration_id: string;
  device_id: string;
  user_id: string;
  status: string;
  registered_at: string;
}

export interface OfflinePatientSummary {
  patient_id: string;
  full_name: string;
  date_of_birth?: string | null;
  gender: string;
  emirates_id?: string | null;
  last_screening_date?: string | null;
  assigned_until: string;
  active_care_gaps_count: number;
}

export interface OfflineAssignmentPackageResponse {
  chw_user_id: string;
  organization_id: string;
  assigned_patients: OfflinePatientSummary[];
  package_timestamp: string;
  expires_at: string;
}

export interface OfflineSyncOperationRequest {
  operation_id: string;
  idempotency_key: string;
  resource_type: string;
  operation_type: string;
  client_timestamp: string;
  payload: Record<string, any>;
}

export interface OfflineSyncBatchRequest {
  device_id: string;
  operations: OfflineSyncOperationRequest[];
}

export interface SyncOperationResult {
  operation_id: string;
  status: string;
  resource_id?: string | null;
  server_timestamp: string;
  error_message?: string | null;
}

export interface OfflineSyncBatchResponse {
  batch_id: string;
  total_processed: number;
  synced_count: number;
  conflict_count: number;
  rejected_count: number;
  results: SyncOperationResult[];
  server_time: string;
}

export interface SyncConflictResponse {
  conflict_id: string;
  operation_id: string;
  resource_type: string;
  resource_id: string;
  patient_id: string;
  user_id: string;
  conflict_reason: string;
  client_payload: Record<string, any>;
  server_payload: Record<string, any>;
  resolution_status: string;
  created_at: string;
}

export interface ResolveConflictRequest {
  resolution: 'ACCEPT_CLIENT' | 'KEEP_SERVER';
  notes?: string;
}

// ==========================================
// PHASE 14: HEALTHCARE INTEROPERABILITY
// ==========================================

export interface IntegrationProviderResponse {
  provider_name: string;
  display_name: string;
  provider_type: string;
  status: string;
  endpoint_url?: string | null;
  is_enabled: boolean;
  prerequisites: string[];
  last_heartbeat_at?: string | null;
}

export interface FhirExportResponse {
  patient_id: string;
  bundle: Record<string, any>;
  exported_at: string;
}

export interface FhirIngestResponse {
  status: string;
  patient_id?: string | null;
  resources_ingested: number;
  observations_created: number;
  encounters_created: number;
  provenance_id: string;
  timestamp: string;
}

export interface TerminologyLookupResponse {
  internal_code: string;
  display_name: string;
  target_system: string;
  target_code: string;
  target_display: string;
  ucum_unit?: string | null;
  status: string;
}

export interface IntegrationMonitoringResponse {
  total_providers: number;
  configured_providers: number;
  not_configured_providers: number;
  providers: IntegrationProviderResponse[];
  recent_jobs: Array<Record<string, any>>;
  generated_at: string;
}
