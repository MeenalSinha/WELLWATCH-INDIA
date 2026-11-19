'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { Menu, LogOut, Shield, Bell } from 'lucide-react';
import { clearSession } from '@/lib/auth';
import { TokenResponse } from '@/types';
import { StatusBadge } from './StatusBadge';

interface TopHeaderProps {
  user: TokenResponse | null;
  onOpenMobileSidebar: () => void;
  pageTitle?: string;
  breadcrumbs?: { label: string; href?: string }[];
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  user,
  onOpenMobileSidebar,
  pageTitle,
  breadcrumbs,
}) => {
  const router = useRouter();

  const handleSignOut = () => {
    clearSession();
    router.push('/login');
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200/90 sticky top-0 z-30 flex items-center justify-between px-4 sm:px-6 lg:px-8">
      {/* Left: Mobile Toggle & Page Context */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobileSidebar}
          className="lg:hidden p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition"
          aria-label="Open sidebar menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          {breadcrumbs && breadcrumbs.length > 0 && (
            <div className="hidden sm:flex items-center gap-1.5 text-[11px] text-slate-400 font-medium">
              {breadcrumbs.map((b, i) => (
                <React.Fragment key={i}>
                  {i > 0 && <span>/</span>}
                  {b.href ? (
                    <a href={b.href} className="hover:text-forest-700 transition">
                      {b.label}
                    </a>
                  ) : (
                    <span className="text-slate-600">{b.label}</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          )}
          {pageTitle && (
            <h2 className="text-sm sm:text-base font-bold text-slate-900 tracking-tight leading-tight">
              {pageTitle}
            </h2>
          )}
        </div>
      </div>

      {/* Right: Operational Status & User Identity */}
      <div className="flex items-center gap-3 sm:gap-4">
        {/* DoH Jurisdiction Pill */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-forest-50 border border-forest-100 text-[11px] font-medium text-forest-800">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>Abu Dhabi, UAE (DoH Guidelines)</span>
        </div>

        {/* Notifications Icon (Direct link to outreach/notifications) */}
        <button
          onClick={() => router.push('/outreach')}
          className="relative p-2 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition"
          title="Outreach & Alerts"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-emerald-500" />
        </button>

        <div className="h-6 w-px bg-slate-200" />

        {/* User Profile Pill */}
        {user ? (
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex flex-col text-right">
              <span className="text-xs font-semibold text-slate-900 leading-tight">
                {user.full_name}
              </span>
              <span className="text-[10px] text-slate-500">
                {user.role ? user.role.replace(/_/g, ' ') : 'Healthcare Staff'}
              </span>
            </div>

            <div className="w-8 h-8 rounded-full bg-forest-800 text-white flex items-center justify-center font-bold text-xs shadow-xs">
              {user.full_name ? user.full_name.charAt(0).toUpperCase() : 'U'}
            </div>

            <button
              onClick={handleSignOut}
              className="p-1.5 rounded-lg text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <button
            onClick={() => router.push('/login')}
            className="px-3 py-1.5 rounded-lg bg-forest-800 text-white text-xs font-semibold shadow-xs hover:bg-forest-900 transition"
          >
            Sign In
          </button>
        )}
      </div>
    </header>
  );
};
