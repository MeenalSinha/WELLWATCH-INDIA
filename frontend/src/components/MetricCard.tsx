import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label?: string;
  title?: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  trend?: string;
  trendDirection?: 'up' | 'down' | 'neutral';
  icon?: LucideIcon | React.ComponentType<{ className?: string }>;
  variant?: 'default' | 'forest' | 'amber' | 'blue' | 'emerald' | 'teal';
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
  const displayLabel = title || label || '';

  // Border accent tinting
  const variantStyles = {
    default: 'hover:border-slate-300',
    forest: 'border-l-4 border-l-forest-700',
    emerald: 'border-l-4 border-l-emerald-600',
    teal: 'border-l-4 border-l-teal-600',
    amber: 'border-l-4 border-l-amber-500',
    blue: 'border-l-4 border-l-sky-600',
  }[variant] || 'hover:border-slate-300';

  return (
    <div
      className={`card-healthcare p-5 flex flex-col justify-between transition-all duration-150 hover:shadow-md ${variantStyles} ${className}`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider line-clamp-1">
          {displayLabel}
        </span>
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-forest-50/80 border border-forest-100 flex items-center justify-center text-forest-800 shrink-0">
            <Icon className="w-4 h-4 text-forest-700" />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight font-sans">
          {value}
        </span>
        {unit && <span className="text-xs font-medium text-slate-500">{unit}</span>}
      </div>

      {(subtitle || trend) && (
        <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px]">
          {subtitle && <span className="text-slate-500 line-clamp-1">{subtitle}</span>}
          {trend && (
            <span
              className={`font-semibold shrink-0 ${
                trendDirection === 'down'
                  ? 'text-rose-700'
                  : trendDirection === 'neutral'
                  ? 'text-slate-600'
                  : 'text-emerald-700'
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
