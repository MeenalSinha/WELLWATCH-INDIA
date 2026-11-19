import React from 'react';
import { LucideIcon } from 'lucide-react';

export interface MetricCardProps {
  label?: string;
  title?: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  trend?: string;
  trendDirection?: 'up' | 'down' | 'neutral';
  icon?: LucideIcon | any;
  variant?: 'default' | 'forest' | 'emerald' | 'teal' | 'amber' | 'blue' | 'rose';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  title,
  value,
  unit,
  subtitle,
  trend,
  trendDirection,
  icon: Icon,
  variant = 'default',
  className = '',
}) => {
  const displayLabel = label || title || '';

  // Border accent tinting
  const variantStyles: Record<string, string> = {
    default: 'hover:border-slate-300',
    forest: 'border-l-4 border-l-forest-700',
    emerald: 'border-l-4 border-l-emerald-600',
    teal: 'border-l-4 border-l-teal-600',
    amber: 'border-l-4 border-l-amber-500',
    blue: 'border-l-4 border-l-sky-600',
    rose: 'border-l-4 border-l-rose-500',
  };

  const borderStyle = variantStyles[variant] || variantStyles.default;

  return (
    <div
      className={`bg-white p-5 rounded-xl border border-slate-200/90 shadow-xs hover:shadow-sm transition-all duration-150 ${borderStyle} ${className}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
          {displayLabel}
        </span>
        {Icon && (
          <div className="w-7 h-7 rounded-lg bg-slate-50 border border-slate-100 flex items-center justify-center text-slate-600">
            <Icon className="w-4 h-4 text-forest-700" />
          </div>
        )}
      </div>

      <div className="mt-2.5 flex items-baseline gap-1.5">
        <span className="text-2xl font-bold tracking-tight text-slate-900">{value}</span>
        {unit && <span className="text-xs font-medium text-slate-500">{unit}</span>}
      </div>

      {(subtitle || trend) && (
        <div className="mt-2.5 pt-2 border-t border-slate-100/80 text-[11px] text-slate-500 flex items-center justify-between gap-1">
          {subtitle && <span className="truncate">{subtitle}</span>}
          {trend && (
            <span
              className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                trendDirection === 'down'
                  ? 'bg-rose-50 text-rose-700'
                  : 'bg-emerald-50 text-emerald-700'
              }`}
            >
              {trend}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
