'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Users,
  Stethoscope,
  GitBranch,
  ShieldAlert,
  Sparkles,
  ClipboardList,
  MessageSquare,
  Navigation,
  Smartphone,
  BarChart3,
  Network,
  FileCheck2,
  ScrollText,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Building2,
} from 'lucide-react';
import { TokenResponse, UserRole } from '@/types';

interface SidebarProps {
  user: TokenResponse | null;
  isOpenMobile: boolean;
  onCloseMobile: () => void;
}

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  allowedRoles?: UserRole[];
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

export const Sidebar: React.FC<SidebarProps> = ({ user, isOpenMobile, onCloseMobile }) => {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  const userRole = user?.role;

  const navGroups: NavGroup[] = [
    {
      title: 'OVERVIEW',
      items: [
        { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      ],
    },
    {
      title: 'PATIENT INTELLIGENCE',
      items: [
        { href: '/patients', label: 'Patients Directory', icon: Users },
        { href: '/screening', label: 'Clinical Screening', icon: Stethoscope },
        { href: '/pathways', label: 'Care Pathways', icon: GitBranch },
        { href: '/assessments', label: 'Risk Assessments', icon: ShieldAlert },
        { href: '/recommendations', label: 'Recommendations', icon: Sparkles },
        { href: '/care-plans', label: 'Care Plans', icon: ClipboardList },
      ],
    },
    {
      title: 'CARE OPERATIONS',
      items: [
        { href: '/outreach', label: 'Outreach & RTL', icon: MessageSquare },
        { href: '/navigation', label: 'Care Navigation', icon: Navigation },
        { href: '/chw', label: 'CHW Field Workspace', icon: Smartphone, badge: 'Offline' },
      ],
    },
    {
      title: 'POPULATION HEALTH',
      items: [
        { href: '/population', label: 'Population Analytics', icon: BarChart3 },
      ],
    },
    {
      title: 'PLATFORM & GOVERNANCE',
      items: [
        { href: '/integrations', label: 'Interoperability Hub', icon: Network },
        { href: '/consent', label: 'Consent & Sovereign Privacy', icon: FileCheck2 },
        { href: '/audit', label: 'Audit & Accountability', icon: ScrollText },
      ],
    },
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpenMobile && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-xs lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col bg-forest-950 text-slate-200 border-r border-forest-900 transition-all duration-300 ease-in-out lg:static lg:z-auto ${
          collapsed ? 'w-20' : 'w-64'
        } ${isOpenMobile ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
      >
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-forest-900/80 shrink-0">
          <Link href="/dashboard" className="flex items-center gap-3 overflow-hidden">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-forest-700 text-white flex items-center justify-center font-bold text-lg shadow-sm shrink-0">
              W
            </div>
            {!collapsed && (
              <div className="leading-tight shrink-0">
                <div className="font-extrabold text-sm tracking-tight text-white flex items-center gap-1.5">
                  WELLWATCH
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                    PREVENT
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-medium tracking-wide">
                  Life-Course Intelligence
                </div>
              </div>
            )}
          </Link>

          {/* Desktop Collapse Toggle */}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="hidden lg:flex w-7 h-7 rounded-lg text-slate-400 hover:text-white hover:bg-forest-900 items-center justify-center transition"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Scrollable Navigation Groups */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {navGroups.map((group, gIdx) => (
            <div key={gIdx} className="space-y-1">
              {!collapsed && (
                <div className="px-3 text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  {group.title}
                </div>
              )}
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href || (item.href !== '/dashboard' && pathname.startsWith(item.href));

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={onCloseMobile}
                      className={`group relative flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-forest-800 text-white shadow-xs font-semibold'
                          : 'text-slate-300 hover:bg-forest-900/60 hover:text-white'
                      } ${collapsed ? 'justify-center' : ''}`}
                      title={collapsed ? item.label : undefined}
                    >
                      <Icon
                        className={`w-4 h-4 shrink-0 transition-colors ${
                          isActive ? 'text-emerald-400' : 'text-slate-400 group-hover:text-slate-200'
                        }`}
                      />
                      {!collapsed && <span className="truncate flex-1">{item.label}</span>}
                      {!collapsed && item.badge && (
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-forest-900 text-emerald-300 border border-emerald-500/20">
                          {item.badge}
                        </span>
                      )}
                      {/* Active Left Indicator Bar */}
                      {isActive && (
                        <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-emerald-400" />
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Bottom Jurisdiction & Facility Badge */}
        {!collapsed ? (
          <div className="p-3 mx-3 mb-3 rounded-xl bg-forest-900/50 border border-forest-800/80 text-[11px] space-y-1">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold text-[10px] uppercase tracking-wider">
              <ShieldCheck className="w-3.5 h-3.5" />
              Abu Dhabi Healthcare
            </div>
            <div className="text-slate-300 text-[10px] leading-tight">
              Aligned with DoH & ADHICS Sovereign Standards
            </div>
          </div>
        ) : (
          <div className="p-2 mb-2 flex justify-center text-emerald-400" title="Abu Dhabi DoH Healthcare">
            <ShieldCheck className="w-5 h-5" />
          </div>
        )}
      </aside>
    </>
  );
};
