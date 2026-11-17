import {
  AdultProfile,
  AuditLog,
  CareGapItem,
  ComprehensiveEvaluation,
  ConsentRecord,
  GenomicFindingResponse,
  HealthStatus,
  ObservationCreate,
  PathwaySummary,
  PediatricProfile,
  PersonCreate,
  PersonListResponse,
  PersonSummary,
  PreventiveAssessmentResponse,
  ScreeningEncounter,
  TimelineResponse,
  TokenResponse,
  User,
  // Phase 9
  PreventiveRecommendationResponse,
  RecommendationListResponse,
  RecommendationExplanation,
  CarePlanResponse,
  // Phase 10
  CommunicationPreferenceResponse,
  OutreachTaskResponse,
  OutreachTaskListResponse,
  OutreachAnalyticsResponse,
  TemplatePreviewResponse,
  // Phase 11
  AppointmentResponse,
  ReferralResponse,
  FollowUpTaskResponse,
  CareGapClosureResponse,
  NavigationAnalyticsResponse,
} from '@/types';
import { getStoredToken } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1';

export class ApiError extends Error {
  status: number;
  code: string;
  correlationId?: string;

  constructor(status: number, message: string, code: string = 'API_ERROR', correlationId?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.correlationId = correlationId;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  const correlationId = response.headers.get('x-correlation-id') || undefined;

  if (!response.ok) {
    let errorMsg = `API Error ${response.status}: ${response.statusText}`;
    let errorCode = 'API_ERROR';
    try {
      const errJson = await response.json();
      if (errJson.error) {
        errorMsg = errJson.error.message || errorMsg;
        errorCode = errJson.error.code || errorCode;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Fallback
    }
    throw new ApiError(response.status, errorMsg, errorCode, correlationId);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const apiClient = {
  // Health
  getHealth: () => request<HealthStatus>('/health'),
  getReadiness: () => request<{ status: string; database_ready: boolean }>('/ready'),
  listScreenings: (organizationId?: string) => request<ScreeningEncounter[]>('/screenings'),

  // Auth
  login: (email: string, password: string) =>
    request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),
  getMe: () => request<User>('/auth/me'),

  // Patients (Phase 5)
  getPatients: (page = 1, pageSize = 20, search?: string) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (search) params.set('search', search);
    return request<PersonListResponse>(`/patients?${params.toString()}`);
  },
  getPatient: (id: string) => request<PersonSummary>(`/patients/${id}`),
  registerPatient: (data: PersonCreate) =>
    request<PersonSummary>('/patients', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getPediatricProfile: (id: string) => request<PediatricProfile>(`/patients/${id}/pediatric-profile`),
  getAdultProfile: (id: string) => request<AdultProfile>(`/patients/${id}/adult-profile`),
  getPatientTimeline: (id: string) => request<TimelineResponse>(`/patients/${id}/timeline`),
  recordObservation: (patientId: string, data: ObservationCreate) =>
    request<any>(`/patients/${patientId}/observations`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Pathways (Phase 6)
  getPathways: () => request<PathwaySummary[]>('/pathways'),
  getPathway: (id: string) => request<any>(`/pathways/${id}`),
  evaluatePathway: (id: string, payload: any) =>
    request<any>(`/pathways/${id}/evaluate`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  evaluateComprehensive: (patientId: string) =>
    request<ComprehensiveEvaluation>(`/pathways/evaluate-patient/${patientId}`),
  getPatientCareGaps: (patientId: string) =>
    request<CareGapItem[]>(`/pathways/patient-gaps/${patientId}`),

  // Assessments (Phase 7)
  getAssessments: (patientId?: string, skip = 0, limit = 50) => {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
    if (patientId) params.set('patient_id', patientId);
    return request<PreventiveAssessmentResponse[]>(`/assessments?${params.toString()}`);
  },
  getAssessment: (assessmentId: string) =>
    request<PreventiveAssessmentResponse>(`/assessments/${assessmentId}`),
  generateAssessment: (patientId: string, pathwayId: string, pathwayVersion: string) =>
    request<PreventiveAssessmentResponse>('/assessments/generate', {
      method: 'POST',
      body: JSON.stringify({ patient_id: patientId, pathway_id: pathwayId, pathway_version: pathwayVersion }),
    }),
  reviewAssessment: (assessmentId: string, reviewStatus: string, notes: string) =>
    request<{ success: boolean; assessment_id: string; human_review_status: string; reviewed_at: string }>(
      `/assessments/${assessmentId}/review`,
      {
        method: 'POST',
        body: JSON.stringify({ review_status: reviewStatus, notes }),
      }
    ),

  // Genomic Intelligence (Phase 8)
  getPatientGenomics: (patientId: string) =>
    request<GenomicFindingResponse[]>(`/patients/${patientId}/genomics`),
  recordGenomicFinding: (patientId: string, data: any) =>
    request<GenomicFindingResponse>(`/patients/${patientId}/genomics`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  reviewGenomicFinding: (findingId: string, newStatus: string, notes: string) =>
    request<GenomicFindingResponse>(`/genomics/${findingId}/review`, {
      method: 'POST',
      body: JSON.stringify({ new_status: newStatus, notes }),
    }),

  // Phase 9: Recommendations & Care Plans
  generateRecommendations: (patientId: string) =>
    request<PreventiveRecommendationResponse[]>(`/recommendations/generate/${patientId}`, {
      method: 'POST',
    }),
  getRecommendations: (patientId?: string, status?: string, priority?: string, limit = 50, offset = 0) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (patientId) params.set('person_id', patientId);
    if (status) params.set('status', status);
    if (priority) params.set('priority', priority);
    return request<RecommendationListResponse>(`/recommendations?${params.toString()}`);
  },
  getRecommendation: (id: string) =>
    request<PreventiveRecommendationResponse>(`/recommendations/${id}`),
  explainRecommendation: (id: string) =>
    request<RecommendationExplanation>(`/recommendations/${id}/explain`),
  reviewRecommendation: (id: string, reviewDecision: string, rationale: string, modifications?: any) =>
    request<PreventiveRecommendationResponse>(`/recommendations/${id}/review`, {
      method: 'POST',
      body: JSON.stringify({ review_decision: reviewDecision, rationale, modifications }),
    }),
  updateRecommendationStatus: (id: string, lifecycleStatus: string, declinedReason?: string, reviewerNotes?: string) =>
    request<PreventiveRecommendationResponse>(`/recommendations/${id}/status`, {
      method: 'POST',
      body: JSON.stringify({ lifecycle_status: lifecycleStatus, declined_reason: declinedReason, reviewer_notes: reviewerNotes }),
    }),
  createCarePlan: (payload: any) =>
    request<CarePlanResponse>('/care-plans', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getPatientCarePlans: (patientId: string) =>
    request<CarePlanResponse[]>(`/care-plans/patient/${patientId}`),

  // Phase 10: Cultural Outreach
  getCommunicationPreferences: (patientId: string) =>
    request<CommunicationPreferenceResponse>(`/outreach/preferences/${patientId}`),
  updateCommunicationPreferences: (patientId: string, payload: any) =>
    request<CommunicationPreferenceResponse>(`/outreach/preferences/${patientId}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  createOutreachTask: (payload: any) =>
    request<OutreachTaskResponse>('/outreach/tasks', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getOutreachTasks: (patientId?: string, deliveryStatus?: string, limit = 50, offset = 0) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (patientId) params.set('person_id', patientId);
    if (deliveryStatus) params.set('delivery_status', deliveryStatus);
    return request<OutreachTaskListResponse>(`/outreach/tasks?${params.toString()}`);
  },
  retryOutreachTask: (taskId: string) =>
    request<OutreachTaskResponse>(`/outreach/tasks/${taskId}/retry`, {
      method: 'POST',
    }),
  previewTemplate: (payload: { template_id: string; language?: string; variables?: Record<string, string> }) =>
    request<TemplatePreviewResponse>('/outreach/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getOutreachAnalytics: () =>
    request<OutreachAnalyticsResponse>('/outreach/analytics'),

  // Phase 11: Care Navigation
  bookAppointment: (payload: any) =>
    request<AppointmentResponse>('/navigation/appointments', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getAppointments: (patientId?: string, status?: string) => {
    const params = new URLSearchParams();
    if (patientId) params.set('person_id', patientId);
    if (status) params.set('status', status);
    return request<AppointmentResponse[]>(`/navigation/appointments?${params.toString()}`);
  },
  updateAppointmentStatus: (id: string, status: string, attendanceNotes?: string) =>
    request<AppointmentResponse>(`/navigation/appointments/${id}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status, attendance_notes: attendanceNotes }),
    }),
  createReferral: (payload: any) =>
    request<ReferralResponse>('/navigation/referrals', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getReferrals: (patientId?: string, status?: string) => {
    const params = new URLSearchParams();
    if (patientId) params.set('person_id', patientId);
    if (status) params.set('status', status);
    return request<ReferralResponse[]>(`/navigation/referrals?${params.toString()}`);
  },
  updateReferralStatus: (id: string, status: string, outcomeNotes?: string) =>
    request<ReferralResponse>(`/navigation/referrals/${id}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status, outcome_notes: outcomeNotes }),
    }),
  getFollowUps: (patientId?: string, status?: string) => {
    const params = new URLSearchParams();
    if (patientId) params.set('person_id', patientId);
    if (status) params.set('status', status);
    return request<FollowUpTaskResponse[]>(`/navigation/follow-ups?${params.toString()}`);
  },
  resolveFollowUp: (taskId: string, resolutionNotes: string, resolutionEvidenceId?: string) =>
    request<FollowUpTaskResponse>(`/navigation/follow-ups/${taskId}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ resolution_notes: resolutionNotes, resolution_evidence_id: resolutionEvidenceId }),
    }),
  closeCareGapWithEvidence: (payload: {
    person_id: string;
    care_gap_id: string;
    evidence_source: string;
    evidence_id: string;
    verification_method?: string;
    recommendation_id?: string;
    verification_notes?: string;
  }) =>
    request<CareGapClosureResponse>('/navigation/care-gaps/close', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getNavigationAnalytics: () =>
    request<NavigationAnalyticsResponse>('/navigation/analytics'),

  // Consent & Privacy (Phases 1-3)
  getOwnConsents: () => request<ConsentRecord[]>('/consent/me'),
  grantConsent: (data: { purpose: string; data_categories: string[]; consent_version?: string }) =>
    request<ConsentRecord>('/consent', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  revokeConsent: (id: string, reason?: string) =>
    request<ConsentRecord>(`/consent/${id}/revoke`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),

  // Audit Logs (Phases 1-3)
  listAuditLogs: (skip = 0, limit = 50) =>
    request<AuditLog[]>(`/audit/logs?skip=${skip}&limit=${limit}`),
};
