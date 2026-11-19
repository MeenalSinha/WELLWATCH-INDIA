import React from 'react';
import Link from 'next/link';

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface ActionItem {
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  onClick?: () => void;
  href?: string;
}

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  action?: ActionItem;
  breadcrumbs?: (string | BreadcrumbItem)[];
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  badge,
  actions,
  action,
  breadcrumbs,
}) => {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-5 border-b border-slate-200/80 mb-6">
      <div className="space-y-1">
        {breadcrumbs && breadcrumbs.length > 0 && (
          <nav className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            {breadcrumbs.map((crumb, idx) => {
              const label = typeof crumb === 'string' ? crumb : crumb.label;
              const href = typeof crumb === 'string' ? undefined : crumb.href;
              return (
                <React.Fragment key={idx}>
                  {idx > 0 && <span className="text-slate-300">/</span>}
                  {href ? (
                    <Link href={href} className="hover:text-forest-700 transition">
                      {label}
                    </Link>
                  ) : (
                    <span className="text-slate-600 font-medium">{label}</span>
                  )}
                </React.Fragment>
              );
            })}
          </nav>
        )}
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">{title}</h1>
          {badge}
        </div>
        {subtitle && (
          <p className="text-xs sm:text-sm text-slate-500 max-w-3xl leading-relaxed">{subtitle}</p>
        )}
      </div>

      <div className="flex items-center gap-2.5 flex-wrap shrink-0">
        {actions}
        {action && (
          action.href ? (
            <Link
              href={action.href}
              className="btn-healthcare-primary text-xs py-2 px-3.5 inline-flex items-center gap-1.5"
            >
              {action.icon && <action.icon className="w-3.5 h-3.5" />}
              <span>{action.label}</span>
            </Link>
          ) : (
            <button
              onClick={action.onClick}
              className="btn-healthcare-primary text-xs py-2 px-3.5 inline-flex items-center gap-1.5"
            >
              {action.icon && <action.icon className="w-3.5 h-3.5" />}
              <span>{action.label}</span>
            </button>
          )
        )}
      </div>
    </div>
  );
};
