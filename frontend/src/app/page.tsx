'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/lib/api';
import { HealthStatus } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';
import { MetricCard } from '@/components/MetricCard';

export default function HomePage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiClient
      .getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="bg-white rounded-2xl border border-slate-200 p-8 sm:p-12 shadow-sm">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold mb-6">
            <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse"></span>
            Future Health Digital Innovation Challenge 2026
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            From Health Signals to <span className="text-emerald-700">Preventive Action</span>.
          </h1>
          <p className="mt-4 text-base sm:text-lg text-slate-600 leading-relaxed">
            WELLWATCH PREVENT is an AI-powered population health intelligence and preventive-care navigation
            platform engineered to identify children, adults, and communities with elevated health risks or
            unmet preventive care needs across the Emirate of Abu Dhabi.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/login"
              className="px-6 py-3 rounded-lg bg-emerald-800 text-white font-semibold text-sm shadow hover:bg-emerald-900 transition"
            >
              Sign In to Portal
            </Link>
            <Link
              href="/screening"
              className="px-6 py-3 rounded-lg bg-slate-100 text-slate-800 font-semibold text-sm hover:bg-slate-200 transition border border-slate-200"
            >
              Record Screening Vitals
            </Link>
          </div>
        </div>
      </section>

      {/* Live Backend Telemetry */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
            System Telemetry & Platform Baseline
          </h2>
          {health && <StatusBadge status={health.status} />}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Service Status"
            value={loading ? 'Checking...' : health?.status || 'Offline'}
            subtitle={health ? `API v${health.version}` : 'Backend unreachable'}
          />
          <MetricCard
            label="Database Engine"
            value={loading ? 'Checking...' : health?.database === 'connected' ? 'Connected' : 'Unavailable'}
            subtitle="PostgreSQL / SQLite Dual Engine"
          />
          <MetricCard
            label="Target Jurisdiction"
            value="Abu Dhabi"
            subtitle="United Arab Emirates (DoH)"
          />
          <MetricCard
            label="Clinical Pathways"
            value="2 Primary"
            subtitle="IFHAS Adult + Well-Child Visits"
          />
        </div>
      </section>

      {/* Core Innovation Pillars */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-800 flex items-center justify-center font-bold text-lg">
            1
          </div>
          <h3 className="text-base font-bold text-slate-900">Life-Course Precision</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Tracks preventive continuity from newborn developmental milestones through adolescent transitions,
            adult cardiovascular screening, and senior frailty surveillance.
          </p>
        </div>
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-800 flex items-center justify-center font-bold text-lg">
            2
          </div>
          <h3 className="text-base font-bold text-slate-900">Rules & ML Decoupling</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Deterministic clinical eligibility governed by official DoH guidelines operates independently from
            calibrated, explainable machine learning risk prioritization.
          </p>
        </div>
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-800 flex items-center justify-center font-bold text-lg">
            3
          </div>
          <h3 className="text-base font-bold text-slate-900">Epistemic Uncertainty</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Explicit 8-state data certainty modeling ensures missing records are never falsely interpreted
            as normal findings or confirmed non-compliance.
          </p>
        </div>
      </section>
    </div>
  );
}
