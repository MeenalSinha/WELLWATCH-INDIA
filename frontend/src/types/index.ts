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
