'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { setSession } from '@/lib/auth';
import { ShieldCheck, Lock, Mail, ArrowLeft, AlertCircle, CheckCircle2, Building2 } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('healthcare_provider@wellwatch.ae');
  const [password, setPassword] = useState('ValidPass123!');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionNotice, setSessionNotice] = useState<string | null>(() => {
    if (typeof window !== 'undefined' && window.location.search.includes('expired=1')) {
      return 'Your clinical session has expired for patient safety. Please re-authenticate.';
    }
    return null;
  });

  const personaOptions = [
    {
      role: 'Healthcare Provider (Physician)',
      email: 'healthcare_provider@wellwatch.ae',
      desc: 'Full clinical decision support, IFHAS screening review & orders',
      badge: 'CLINICAL',
    },
    {
      role: 'Community Health Worker (CHW)',
      email: 'community_health_worker@wellwatch.ae',
      desc: 'Offline-first field screening, community cohort navigation & sync',
      badge: 'FIELD OPS',
    },
    {
      role: 'Individual (Patient)',
      email: 'individual@wellwatch.ae',
      desc: 'Bilingual health records, preventive reminders & consent management',
      badge: 'RESIDENT',
    },
    {
      role: 'Parent / Guardian',
      email: 'parent_guardian@wellwatch.ae',
      desc: 'Pediatric Well-Child tracking, growth percentiles & vaccines',
      badge: 'PEDIATRICS',
    },
    {
      role: 'Care Coordinator',
      email: 'care_coordinator@wellwatch.ae',
      desc: 'Care-gap closures, referral tracking & Malaffi exchange logs',
      badge: 'NAVIGATION',
    },
    {
      role: 'Population Health Admin',
      email: 'population_health_admin@wellwatch.ae',
      desc: 'Epidemiological dashboards, disparity audits & cohort reports',
      badge: 'ANALYTICS',
    },
    {
      role: 'Data Governance Officer',
      email: 'data_governance_officer@wellwatch.ae',
      desc: 'Tamper-evident audit logs, Wadeema law compliance & consents',
      badge: 'GOVERNANCE',
    },
    {
      role: 'System Administrator',
      email: 'system_admin@wellwatch.ae',
      desc: 'Infrastructure monitoring, integrations hub & identity management',
      badge: 'SYS ADMIN',
    },
  ];

  const handlePersonaSelect = (selectedEmail: string) => {
    setEmail(selectedEmail);
    setPassword('ValidPass123!');
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const tokenData = await apiClient.login(email, password);
      setSession(tokenData);
      router.push('/dashboard');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Authentication failed. Please verify credentials or connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAF9] flex flex-col justify-between font-sans">
      {/* UAE DoH Top Ribbon */}
      <header className="bg-forest-950 text-slate-300 text-xs py-2 px-4 border-b border-forest-800">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px]">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold text-white tracking-wide">
              UNITED ARAB EMIRATES &bull; DEPARTMENT OF HEALTH ABU DHABI
            </span>
            <span className="text-slate-400 hidden md:inline">| ADHICS Sovereign Tier 3 Certified</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-slate-300 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Return to Public Overview</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Authentication Card */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-6 my-6">
        <div className="w-full max-w-md bg-white rounded-2xl border border-slate-200 shadow-xl overflow-hidden">
          {/* Card Header */}
          <div className="bg-gradient-to-b from-forest-900 to-forest-950 p-6 text-white text-center relative">
            <div className="w-12 h-12 rounded-xl bg-forest-800/80 border border-emerald-500/30 text-white flex items-center justify-center font-bold text-2xl mx-auto mb-3 shadow-inner">
              W
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">WELLWATCH PREVENT</h1>
            <p className="text-xs text-emerald-200/80 mt-1 font-medium">
              Life-Course Preventive Care Intelligence &bull; Abu Dhabi
            </p>
            <div className="inline-flex items-center gap-1.5 mt-3 px-2.5 py-1 rounded-full bg-forest-800/60 border border-emerald-400/20 text-[11px] text-emerald-300">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Authorized Identity & Access Gateway</span>
            </div>
          </div>

          <div className="p-6 space-y-5">
            {/* Quick-Select Persona Switcher */}
            <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <label className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                  <Building2 className="w-3.5 h-3.5 text-forest-700" />
                  Quick-Select Test Persona
                </label>
                <span className="text-[10px] text-forest-700 font-semibold px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200">
                  Demo Ready
                </span>
              </div>
              <select
                className="w-full text-xs bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-forest-600 focus:border-transparent transition"
                value={email}
                onChange={(e) => handlePersonaSelect(e.target.value)}
              >
                {personaOptions.map((opt) => (
                  <option key={opt.email} value={opt.email}>
                    {opt.role} [{opt.badge}]
                  </option>
                ))}
              </select>
              {(() => {
                const active = personaOptions.find((p) => p.email === email);
                return active ? (
                  <p className="text-[11px] text-slate-500 mt-2 italic">
                    &bull; {active.desc}
                  </p>
                ) : null;
              })()}
            </div>

            {/* Notifications */}
            {sessionNotice && (
              <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <span>{sessionNotice}</span>
              </div>
            )}

            {error && (
              <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Healthcare Professional Email
                </label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg text-slate-900 bg-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-forest-600 focus:border-transparent transition"
                    placeholder="practitioner@wellwatch.ae"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold text-slate-700">Password</label>
                  <span className="text-[11px] text-slate-400">Default: ValidPass123!</span>
                </div>
                <div className="relative">
                  <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg text-slate-900 bg-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-forest-600 focus:border-transparent transition"
                    placeholder="••••••••••••"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 px-4 rounded-lg bg-forest-800 hover:bg-forest-900 text-white font-semibold text-sm transition-all duration-150 shadow hover:shadow-md flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Verifying Clinical Credentials...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Sign In to Clinical Workspace</span>
                  </>
                )}
              </button>
            </form>

            {/* Legal Notice */}
            <div className="pt-3 border-t border-slate-100 text-center">
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Access is restricted to licensed healthcare practitioners and residents under UAE Federal Law No. 2 of 2019 and ADHICS standards.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-4 text-xs text-slate-500 border-t border-slate-200 bg-white">
        &copy; 2026 WELLWATCH PREVENT &bull; Future Health Digital Innovation Challenge 2026 &bull; Abu Dhabi, UAE
      </footer>
    </div>
  );
}
