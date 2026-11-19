import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
  showDot?: boolean;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'sm',
  showDot = true,
  className = '',
}) => {
  const norm = (status || '').toUpperCase().trim();

  // Color mapping based on healthcare context
  let colorStyles = 'bg-slate-50 text-slate-700 border-slate-200';
  let dotColor = 'bg-slate-400';

  // Positive / Verified / Complete
  if (['VERIFIED', 'ACTIVE', 'SUCCESS', 'APPROVED', 'COMPLETED', 'ACCEPTED', 'ONLINE', 'CONNECTED', 'LOW'].includes(norm)) {
    colorStyles = 'bg-emerald-50 text-emerald-800 border-emerald-200/80';
    dotColor = 'bg-emerald-500';
  }
  // Attention / Pending / Medium
  else if (['PENDING', 'PENDING_CLINICAL_REVIEW', 'MODERATE', 'MODERATE_RISK', 'MEDIUM', 'PLANNED', 'SCHEDULED', 'IN_PROGRESS', 'OFFERED', 'UNVERIFIED', 'SUBMITTED'].includes(norm)) {
    colorStyles = 'bg-amber-50 text-amber-800 border-amber-200/80';
    dotColor = 'bg-amber-500';
  }
  // Critical / High / Risk / Revoked
  else if (['HIGH', 'HIGH_RISK', 'CRITICAL', 'REVOKED', 'DENIED', 'FAILURE', 'DECLINED', 'CONFLICT', 'DISCREPANT', 'EXPIRED', 'FAILED', 'BLOCKED'].includes(norm)) {
    colorStyles = 'bg-rose-50 text-rose-800 border-rose-200/80';
    dotColor = 'bg-rose-500';
  }
  // Information / Routine
  else if (['ROUTINE', 'INFORMATIONAL', 'SYNCED', 'INDIVIDUAL', 'PARENT_GUARDIAN', 'DRAFT'].includes(norm)) {
    colorStyles = 'bg-sky-50 text-sky-800 border-sky-200/80';
    dotColor = 'bg-sky-500';
  }
  // Clinical Specialist / Healthcare Role
  else if (['HEALTHCARE_PROVIDER', 'CARE_COORDINATOR', 'COMMUNITY_HEALTH_WORKER', 'DATA_GOVERNANCE_OFFICER', 'POPULATION_HEALTH_ADMIN', 'SYSTEM_ADMIN'].includes(norm)) {
    colorStyles = 'bg-forest-50 text-forest-800 border-forest-200/80';
    dotColor = 'bg-forest-600';
  }
  // Epistemic Uncertainty / Special
  else if (['CONFLICTING', 'UNKNOWN', 'INCOMPLETE', 'ABSTAIN'].includes(norm)) {
    colorStyles = 'bg-purple-50 text-purple-800 border-purple-200/80';
    dotColor = 'bg-purple-500';
  }

  const sizeStyles = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium border shadow-xs transition-colors ${sizeStyles} ${colorStyles} ${className}`}
    >
      {showDot && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotColor}`} />}
      <span className="truncate">{status.replace(/_/g, ' ')}</span>
    </span>
  );
};
