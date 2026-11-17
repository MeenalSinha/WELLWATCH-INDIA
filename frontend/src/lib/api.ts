import { AuditLog, ConsentRecord, HealthStatus, ScreeningEncounter, TokenResponse, User } from '@/types';
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
          errorMsg = errJson.detail;
        }
      } catch {
        // Fallback to text
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
  // Health
  getHealth: () => request<HealthStatus>('/health'),
  getReadiness: () => request<{ status: string; database_ready: bool }>('/ready'),

  // Auth
  login: (email: string, password: string) =>
    request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),
  getMe: () => request<User>('/auth/me'),

  // Screenings
  recordScreening: (data: Partial<ScreeningEncounter>) =>
    request<ScreeningEncounter>('/screenings', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  listScreenings: (facilityId?: string) => {
    const qs = facilityId ? `?facility_id=${encodeURIComponent(facilityId)}` : '';
    return request<ScreeningEncounter[]>(`/screenings${qs}`);
  },

  // Consent
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

  // Audit
  listAuditLogs: (skip = 0, limit = 50) =>
    request<AuditLog[]>(`/audit/logs?skip=${skip}&limit=${limit}`),
};
