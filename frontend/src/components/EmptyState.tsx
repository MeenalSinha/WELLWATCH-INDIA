import React from 'react';
import { LucideIcon } from 'lucide-react';

interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

interface EmptyStateProps {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  action?: EmptyStateAction;
  icon?: LucideIcon | React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  actionText,
  onAction,
  action,
  icon: Icon,
  className = '',
}) => {
  const finalActionText = actionText || action?.label;
  const finalOnAction = onAction || action?.onClick;

  return (
    <div
      className={`flex flex-col items-center justify-center p-8 sm:p-12 text-center bg-white border border-dashed border-slate-200 rounded-xl my-4 shadow-2xs ${className}`}
    >
      <div className="w-12 h-12 flex items-center justify-center rounded-xl bg-forest-50/80 border border-forest-100 text-forest-700 mb-3.5">
        {Icon ? (
          typeof Icon === 'function' ? (
            <Icon className="w-6 h-6 text-forest-700" />
          ) : (
            Icon
          )
        ) : (
          <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        )}
      </div>
      <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
      <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4 leading-relaxed">{description}</p>
      {finalActionText && finalOnAction && (
        <button
          onClick={finalOnAction}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 border border-transparent text-xs font-semibold rounded-lg text-white bg-forest-800 hover:bg-forest-900 shadow-xs transition"
        >
          {finalActionText}
        </button>
      )}
    </div>
  );
};
