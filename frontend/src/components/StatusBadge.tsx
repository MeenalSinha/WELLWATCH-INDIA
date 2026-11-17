import React from 'react';

interface StatusBadgeProps {
  status: string;
  variant?: 'epistemic' | 'role' | 'risk';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, variant = 'epistemic' }) => {
  const getColors = () => {
    switch (status.toUpperCase()) {
      case 'VERIFIED':
      case 'ACTIVE':
      case 'SUCCESS':
      case 'LOW':
        return 'bg-emerald-50 text-emerald-800 border-emerald-200';
      case 'UNVERIFIED':
      case 'PENDING':
      case 'MEDIUM':
        return 'bg-amber-50 text-amber-800 border-amber-200';
      case 'MISSING':
      case 'HIGH':
      case 'REVOKED':
      case 'DENIED':
      case 'FAILURE':
        return 'bg-rose-50 text-rose-800 border-rose-200';
      case 'UNKNOWN':
      case 'CONFLICTING':
        return 'bg-purple-50 text-purple-800 border-purple-200';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-200';
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getColors()}`}
    >
      {status.replace(/_/g, ' ')}
    </span>
  );
};
