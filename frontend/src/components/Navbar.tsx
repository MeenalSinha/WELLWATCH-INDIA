'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { clearSession, getStoredUser } from '@/lib/auth';
import { TokenResponse } from '@/types';
import { StatusBadge } from './StatusBadge';

export const Navbar: React.FC = () => {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<TokenResponse | null>(null);

  useEffect(() => {
    setUser(getStoredUser());
  }, [pathname]);

  const handleLogout = () => {
    clearSession();
    setUser(null);
    router.push('/login');
  };

  const navLinks = [
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/patients', label: 'Patients' },
    { href: '/pathways', label: 'Pathways' },
    { href: '/population', label: 'Population' },
    { href: '/chw', label: 'CHW Field' },
    { href: '/integrations', label: 'Integrations' },
    { href: '/assessments', label: 'Risk Assessments' },
    { href: '/recommendations', label: 'Recommendations' },
    { href: '/care-plans', label: 'Care Plans' },
    { href: '/outreach', label: 'Outreach & RTL' },
    { href: '/navigation', label: 'Care Navigation' },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
      {/* UAE Jurisdiction Sub-Bar */}
      <div className="bg-slate-900 text-slate-300 text-[11px] px-6 py-1 flex items-center justify-between font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>Target Market: Abu Dhabi, UAE (DoH Guidelines / ADHICS Aligned)</span>
        </div>
        <div className="hidden sm:block text-slate-400">
          Future Health Digital Innovation Challenge 2026
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Tagline */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-800 text-white flex items-center justify-center font-bold text-lg shadow-sm">
                W
              </div>
              <div>
                <span className="text-base font-bold text-slate-900 tracking-tight">
                  WELLWATCH <span className="text-emerald-700">PREVENT</span>
                </span>
                <span className="block text-[10px] text-slate-500 font-medium tracking-wider uppercase">
                  Life-Course Preventive Care Intelligence
                </span>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-slate-100 text-slate-900 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* User Status / Auth CTA */}
          <div className="flex items-center gap-3">
            {user ? (
              <div className="flex items-center gap-3">
                <div className="text-right hidden sm:block">
                  <div className="text-xs font-semibold text-slate-900">{user.full_name}</div>
                  <div className="mt-0.5">
                    <StatusBadge status={user.role} />
                  </div>
                </div>
                <button
                  onClick={handleLogout}
                  className="text-xs text-rose-700 hover:text-rose-900 font-medium px-2.5 py-1.5 rounded border border-rose-200 hover:bg-rose-50 transition"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="text-xs text-white bg-emerald-800 hover:bg-emerald-900 font-medium px-4 py-2 rounded-md shadow-sm transition"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
