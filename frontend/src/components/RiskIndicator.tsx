import React from 'react';

export type RiskTier = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL' | 'UNKNOWN';

interface RiskIndicatorProps {
  tier: string;
  score?: number | null;
  requiresReview?: boolean;
  reviewed?: boolean;
  className?: string;
}

export const RiskIndicator: React.FC<RiskIndicatorProps> = ({
  tier,
  score,
  requiresReview,
  reviewed,
  className = '',
}) => {
  const norm = (tier || '').toUpperCase();

  const configs: Record<string, { label: string; bg: string; text: string; border: string; barColor: string }> = {
    LOW: {
      label: 'Low Preventive Risk',
      bg: 'bg-emerald-50',
      text: 'text-emerald-800',
      border: 'border-emerald-200',
      barColor: 'bg-emerald-500',
    },
    MODERATE: {
      label: 'Moderate Risk',
      bg: 'bg-amber-50',
      text: 'text-amber-800',
      border: 'border-amber-200',
      barColor: 'bg-amber-500',
    },
    MODERATE_RISK: {
      label: 'Moderate Risk',
      bg: 'bg-amber-50',
      text: 'text-amber-800',
      border: 'border-amber-200',
      barColor: 'bg-amber-500',
    },
    HIGH: {
      label: 'High Clinical Risk',
      bg: 'bg-rose-50',
      text: 'text-rose-800',
      border: 'border-rose-200',
      barColor: 'bg-rose-500',
    },
    HIGH_RISK: {
      label: 'High Clinical Risk',
      bg: 'bg-rose-50',
      text: 'text-rose-800',
      border: 'border-rose-200',
      barColor: 'bg-rose-500',
    },
    CRITICAL: {
      label: 'Critical / Urgent',
      bg: 'bg-rose-100',
      text: 'text-rose-900',
      border: 'border-rose-300',
      barColor: 'bg-rose-600',
    },
    UNKNOWN: {
      label: 'Epistemic Abstention',
      bg: 'bg-purple-50',
      text: 'text-purple-800',
      border: 'border-purple-200',
      barColor: 'bg-purple-400',
    },
  };

  const conf = configs[norm] || configs.UNKNOWN;

  return (
    <div
      className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-lg border text-xs font-semibold ${conf.bg} ${conf.text} ${conf.border} ${className}`}
    >
      <span className={`w-2 h-2 rounded-full ${conf.barColor}`} />
      <span>{conf.label}</span>
      {score !== undefined && score !== null && (
        <span className="font-mono text-[11px] opacity-80">({(score * 100).toFixed(0)}%)</span>
      )}
      {requiresReview && (
        <span className="text-[10px] uppercase font-bold tracking-wider px-1 py-0.2 rounded bg-white/70 border border-slate-300 ml-1">
          Review Req.
        </span>
      )}
      {reviewed && (
        <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300 ml-1">
          Attested
        </span>
      )}
    </div>
  );
};
