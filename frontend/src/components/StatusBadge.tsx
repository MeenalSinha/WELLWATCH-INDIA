import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
  showDot?: boolean;
  className?: string;
}

const STATUS_LABELS: Record<string, string> = {
  // Lifecycle & Review states
  PENDING_CLINICAL_REVIEW: 'Pending Review',
  AWAITING_REVIEW: 'Pending Review',
  UNDER_REVIEW: 'Under Review',
  APPROVED: 'Approved',
  MODIFIED: 'Modified',
  REJECTED: 'Rejected',
  OFFERED: 'Offered',
  ACCEPTED: 'Accepted',
  DECLINED: 'Declined',
  SCHEDULED: 'Scheduled',
  COMPLETED: 'Completed',
  IN_PROGRESS: 'In Progress',
  PLANNED: 'Planned',
  CANCELLED: 'Cancelled',
  DRAFT: 'Draft',

  // Priorities
  CRITICAL: 'Critical',
  HIGH: 'High',
  HIGH_RISK: 'High Risk',
  MEDIUM: 'Moderate',
  MODERATE: 'Moderate',
  MODERATE_RISK: 'Moderate Risk',
  LOW: 'Low',
  ROUTINE: 'Routine',

  // Verification & Connectivity
  VERIFIED: 'Verified',
  UNVERIFIED: 'Unverified',
  ACTIVE: 'Active',
  SUCCESS: 'Success',
  FAILED: 'Failed',
  FAILURE: 'Failed',
  ONLINE: 'Online',
  OFFLINE: 'Offline',
  CONNECTED: 'Connected',
  NOT_CONFIGURED: 'Not Configured',
  CONFIGURED: 'Configured',
  SYNCED: 'Synced',
  CONFLICT: 'Conflict',
  DISCREPANT: 'Discrepant',
  EXPIRED: 'Expired',
  REVOKED: 'Revoked',
  DENIED: 'Denied',
  BLOCKED: 'Blocked',
  UNKNOWN: 'Unknown',
  ABSTAIN: 'Abstain',
  INCOMPLETE: 'Incomplete',

  // Personas / Roles
  HEALTHCARE_PROVIDER: 'Physician',
  CARE_COORDINATOR: 'Care Coordinator',
  COMMUNITY_HEALTH_WORKER: 'CHW',
  DATA_GOVERNANCE_OFFICER: 'Governance Officer',
  POPULATION_HEALTH_ADMIN: 'Population Admin',
  SYSTEM_ADMIN: 'System Admin',
  INDIVIDUAL: 'Resident',
  PARENT_GUARDIAN: 'Parent / Guardian',
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'sm',
  showDot = true,
  className = '',
}) => {
  const norm = (status || '').toUpperCase().trim();
  const label = STATUS_LABELS[norm] || status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  // Refined healthcare palette (calm, high-contrast, professional)
  let colorStyles = 'bg-slate-50 text-slate-700 border-slate-200/90';
  let dotColor = 'bg-slate-400';

  // Positive / Verified / Approved
  if (['VERIFIED', 'ACTIVE', 'SUCCESS', 'APPROVED', 'COMPLETED', 'ACCEPTED', 'ONLINE', 'CONNECTED', 'LOW'].includes(norm)) {
    colorStyles = 'bg-emerald-50 text-emerald-800 border-emerald-200/90';
    dotColor = 'bg-emerald-500';
  }
  // Attention / Pending / Moderate
  else if (['PENDING', 'PENDING_CLINICAL_REVIEW', 'AWAITING_REVIEW', 'UNDER_REVIEW', 'MODERATE', 'MODERATE_RISK', 'MEDIUM', 'PLANNED', 'SCHEDULED', 'IN_PROGRESS', 'OFFERED', 'UNVERIFIED', 'SUBMITTED'].includes(norm)) {
    colorStyles = 'bg-amber-50 text-amber-800 border-amber-200/90';
    dotColor = 'bg-amber-500';
  }
  // Critical / High / Risk / Revoked
  else if (['HIGH', 'HIGH_RISK', 'CRITICAL', 'REVOKED', 'DENIED', 'FAILURE', 'DECLINED', 'CONFLICT', 'DISCREPANT', 'EXPIRED', 'FAILED', 'BLOCKED', 'REJECTED'].includes(norm)) {
    colorStyles = 'bg-rose-50 text-rose-800 border-rose-200/90';
    dotColor = 'bg-rose-500';
  }
  // Information / Routine
  else if (['ROUTINE', 'INFORMATIONAL', 'SYNCED', 'INDIVIDUAL', 'PARENT_GUARDIAN', 'DRAFT'].includes(norm)) {
    colorStyles = 'bg-slate-50 text-slate-700 border-slate-200/90';
    dotColor = 'bg-slate-400';
  }
  // Clinical Specialist / Healthcare Role
  else if (['HEALTHCARE_PROVIDER', 'CARE_COORDINATOR', 'COMMUNITY_HEALTH_WORKER', 'DATA_GOVERNANCE_OFFICER', 'POPULATION_HEALTH_ADMIN', 'SYSTEM_ADMIN'].includes(norm)) {
    colorStyles = 'bg-forest-50 text-forest-800 border-forest-200/90';
    dotColor = 'bg-forest-600';
  }
  // Epistemic Uncertainty / Special
  else if (['CONFLICTING', 'UNKNOWN', 'INCOMPLETE', 'ABSTAIN'].includes(norm)) {
    colorStyles = 'bg-purple-50 text-purple-800 border-purple-200/90';
    dotColor = 'bg-purple-500';
  }

  const sizeStyles = size === 'sm' ? 'px-2.5 py-0.5 text-[11px]' : 'px-3 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold border shadow-2xs whitespace-nowrap transition-colors ${sizeStyles} ${colorStyles} ${className}`}
    >
      {showDot && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotColor}`} />}
      <span>{label}</span>
    </span>
  );
};
