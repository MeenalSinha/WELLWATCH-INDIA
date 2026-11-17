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
  const headers = new Headers(options.headers || {});

  headers.set('Content-Type', 'application/json');
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    const correlationId = res.headers.get('X-Correlation-ID') || undefined;

    if (!res.ok) {
      let errorMsg = `HTTP Error ${res.status}`;
      let errorCode = 'REQUEST_FAILED';
      try {
        const errJson = await res.json();
        if (errJson.error) {
          errorMsg = errJson.error.message || errorMsg;
          errorCode = errJson.error.code || errorCode;
        } else if (errJson.detail) {
          errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // Fallback to generic message
      }
      throw new ApiError(res.status, errorMsg, errorCode, correlationId);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, err.message || 'Network connection failed. Backend service might be unreachable.');
  }
}

export const apiClient = {
  // Health & Readiness
  getHealth: () => request<HealthStatus>('/health'),
  getReadiness: () => request<{ status: string; database_ready: boolean }>('/ready'),

  // Auth & Session
  login: (email: string, password: string) =>
    request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),
  getMe: () => request<User>('/auth/me'),

  // Screenings (Legacy & Direct Enrolment)
  recordScreening: (data: Partial<ScreeningEncounter>) =>
    request<ScreeningEncounter>('/screenings', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  listScreenings: (facilityId?: string) => {
    const qs = facilityId ? `?facility_id=${encodeURIComponent(facilityId)}` : '';
    return request<ScreeningEncounter[]>(`/screenings${qs}`);
  },

  // Patients & Patient Intelligence (Phases 4 & 5)
  listPatients: (classification?: string, q?: string, skip = 0, limit = 50) => {
    const params = new URLSearchParams();
    if (classification) params.set('classification', classification);
    if (q) params.set('q', q);
    params.set('skip', skip.toString());
    params.set('limit', limit.toString());
    return request<PersonListResponse>(`/patients?${params.toString()}`);
  },
  getPatient: (patientId: string) => request<PersonSummary>(`/patients/${patientId}`),
  registerPatient: (data: PersonCreate) =>
    request<PersonSummary>('/patients', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getPediatricProfile: (patientId: string) =>
    request<PediatricProfile>(`/patients/${patientId}/pediatric`),
  getAdultProfile: (patientId: string) =>
    request<AdultProfile>(`/patients/${patientId}/adult`),
  getTimeline: (patientId: string) =>
    request<TimelineResponse>(`/patients/${patientId}/timeline`),
  recordObservation: (patientId: string, data: ObservationCreate) =>
    request<{ success: boolean; observation_id: string; canonical_value: number; canonical_unit: string }>(
      `/patients/${patientId}/observations`,
      {
        method: 'POST',
        body: JSON.stringify(data),
      }
    ),

  // Pathways (Phase 6)
  listPathways: () => request<PathwaySummary[]>('/pathways'),
  getPathway: (pathwayId: string) => request<any>(`/pathways/${pathwayId}`),
  evaluateEligibility: (pathwayId: string, patientId: string, evalDate?: string) => {
    const qs = evalDate ? `?eval_date=${evalDate}` : '';
    return request<any>(`/pathways/${pathwayId}/evaluate-eligibility/${patientId}${qs}`, { method: 'POST' });
  },
  evaluateCareGaps: (pathwayId: string, patientId: string, evalDate?: string) => {
    const qs = evalDate ? `?eval_date=${evalDate}` : '';
    return request<any>(`/pathways/${pathwayId}/evaluate-care-gaps/${patientId}${qs}`, { method: 'POST' });
  },
  evaluateAll: (pathwayId: string, patientId: string, evalDate?: string) => {
    const qs = evalDate ? `?eval_date=${evalDate}` : '';
    return request<ComprehensiveEvaluation>(`/pathways/${pathwayId}/evaluate-all/${patientId}${qs}`, { method: 'POST' });
  },

  // Assessments (Phase 7)
  generateAssessment: (patientId: string, pathwayId?: string, evalDate?: string) => {
    const params = new URLSearchParams();
    if (pathwayId) params.set('pathway_id', pathwayId);
    if (evalDate) params.set('eval_date', evalDate);
    const qs = params.toString() ? `?${params.toString()}` : '';
    return request<PreventiveAssessmentResponse>(`/assessments/generate/${patientId}${qs}`, { method: 'POST' });
  },
  getAssessment: (assessmentId: string) =>
    request<PreventiveAssessmentResponse>(`/assessments/${assessmentId}`),
  listPatientAssessments: (patientId: string) =>
    request<PreventiveAssessmentResponse[]>(`/assessments/patient/${patientId}`),
  submitAssessmentReview: (assessmentId: string, reviewStatus: string, notes: string) =>
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
