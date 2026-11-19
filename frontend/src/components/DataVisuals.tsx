import React from 'react';

// ---------------------------------------------------------------------------
// 1. PROGRESS BAR
// ---------------------------------------------------------------------------
interface ProgressBarProps {
  label?: string;
  value: number; // 0 to 100
  target?: number;
  color?: 'forest' | 'emerald' | 'amber' | 'rose' | 'blue';
  showPercentage?: boolean;
  size?: 'sm' | 'md';
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  label,
  value,
  target,
  color = 'forest',
  showPercentage = true,
  size = 'md',
}) => {
  const clamped = Math.min(100, Math.max(0, Math.round(value)));

  const colorStyles = {
    forest: 'bg-forest-700',
    emerald: 'bg-emerald-600',
    amber: 'bg-amber-500',
    rose: 'bg-rose-500',
    blue: 'bg-sky-600',
  }[color];

  const heightStyle = size === 'sm' ? 'h-1.5' : 'h-2.5';

  return (
    <div className="w-full space-y-1.5">
      {(label || showPercentage) && (
        <div className="flex items-center justify-between text-xs">
          {label && <span className="font-medium text-slate-700">{label}</span>}
          <div className="flex items-center gap-1.5 text-slate-500">
            {showPercentage && <span className="font-semibold text-slate-900">{clamped}%</span>}
            {target !== undefined && (
              <span className="text-[10px] text-slate-400">(Target: {target}%)</span>
            )}
          </div>
        </div>
      )}
      <div className={`w-full bg-slate-100 rounded-full overflow-hidden ${heightStyle}`}>
        <div
          className={`${heightStyle} ${colorStyles} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 2. HORIZONTAL CATEGORY BAR CHART
// ---------------------------------------------------------------------------
interface CategoryBarItem {
  label: string;
  value: number;
  total?: number;
  color?: string;
  badge?: string;
}

interface CategoryBarChartProps {
  items: CategoryBarItem[];
  maxValue?: number;
}

export const CategoryBarChart: React.FC<CategoryBarChartProps> = ({ items, maxValue }) => {
  const max = maxValue || Math.max(...items.map((i) => i.value), 1);

  return (
    <div className="space-y-3 w-full">
      {items.map((item, idx) => {
        const pct = Math.min(100, Math.round((item.value / max) * 100));
        return (
          <div key={idx} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-700 font-medium truncate max-w-[70%]">{item.label}</span>
              <div className="flex items-center gap-2">
                {item.badge && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                    {item.badge}
                  </span>
                )}
                <span className="font-bold text-slate-900">{item.value.toLocaleString()}</span>
              </div>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  item.color || 'bg-forest-700'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

// ---------------------------------------------------------------------------
// 3. DONUT DISTRIBUTION CHART
// ---------------------------------------------------------------------------
interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

interface DonutChartProps {
  segments: DonutSegment[];
  centerLabel?: string;
  centerValue?: string | number;
  size?: number;
}

export const DonutChart: React.FC<DonutChartProps> = ({
  segments,
  centerLabel,
  centerValue,
  size = 140,
}) => {
  const total = segments.reduce((acc, s) => acc + s.value, 0) || 1;
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  let currentOffset = 0;

  return (
    <div className="flex flex-col sm:flex-row items-center gap-6">
      <div className="relative shrink-0" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {segments.map((seg, idx) => {
            const segPct = seg.value / total;
            const strokeDasharray = `${segPct * circumference} ${circumference}`;
            const strokeDashoffset = -currentOffset;
            currentOffset += segPct * circumference;

            return (
              <circle
                key={idx}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="transparent"
                stroke={seg.color}
                strokeWidth={strokeWidth}
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-500"
              />
            );
          })}
        </svg>
        {(centerValue !== undefined || centerLabel) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            {centerValue !== undefined && (
              <span className="text-xl font-bold text-slate-900 leading-tight">{centerValue}</span>
            )}
            {centerLabel && <span className="text-[10px] text-slate-500">{centerLabel}</span>}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="space-y-1.5 w-full">
        {segments.map((seg, idx) => {
          const pct = Math.round((seg.value / total) * 100);
          return (
            <div key={idx} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: seg.color }} />
                <span className="text-slate-600 font-medium truncate">{seg.label}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-900 font-semibold">
                <span>{seg.value.toLocaleString()}</span>
                <span className="text-[11px] font-normal text-slate-400">({pct}%)</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
