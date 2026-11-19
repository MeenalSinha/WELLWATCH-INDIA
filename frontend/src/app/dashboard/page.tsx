'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Users,
  Activity,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  Stethoscope,
  HeartPulse,
  PlusCircle,
  FileText,
  Clock,
  AlertCircle,
} from 'lucide-react';
import { apiClient } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import {
  ScreeningEncounter,
  TokenResponse,
  PersonSummary,
  PreventiveRecommendationResponse,
} from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import { LoadingSpinner } from '@/components/LoadingSpinner';

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [screenings, setScreenings] = useState<ScreeningEncounter[]>([]);
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [recommendations, setRecommendations] = useState<PreventiveRecommendationResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const activeUser = getStoredUser();
    if (!activeUser) {
      router.push('/login');
      return;
    }
    setUser(activeUser);

    Promise.all([
      apiClient.listScreenings(activeUser.organization_id).catch(() => []),
      apiClient.getPatients(1, 10).catch(() => ({ total: 0, items: [] })),
      apiClient.getRecommendations(undefined, undefined, undefined, 5, 0).catch(() => ({ total: 0, items: [] })),
    ])
      .then(([screenData, patData, recData]) => {
        setScreenings(screenData || []);
        setPatients(patData.items || []);
        setRecommendations(recData.items || []);
      })
      .finally(() => setLoading(false));
  }, [router]);

  const verifiedRate =
    screenings.length > 0
      ? Math.round(
          (screenings.filter((s) => s.epistemic_status === 'VERIFIED').length / screenings.length) * 100
        )
      : 100;

  return (
    <AppShell
      pageTitle="Clinical Operations Dashboard"
      breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'Dashboard' }]}
    >
      <div className="space-y-6">
        {/* Welcome & Facility Banner */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                Welcome back, {user?.full_name}
              </h1>
              {user?.role && <StatusBadge status={user.role} />}
            </div>
            <p className="text-xs text-slate-500">
              {user?.organization_id
                ? `Assigned Facility: SEHA Al Bateen Healthcare Center (ID: ${user.organization_id})`
                : 'Independent Healthcare Resident Portal'}
            </p>
          </div>

          {/* Quick Action Navigation */}
          <div className="flex flex-wrap gap-2 shrink-0">
            <Link
              href="/screening"
              className="px-3.5 py-2 rounded-lg bg-forest-800 hover:bg-forest-900 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5"
            >
              <PlusCircle className="w-3.5 h-3.5" />
              Record Screening Vitals
            </Link>
            <Link
              href="/recommendations"
              className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200 shadow-2xs transition flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              Active Recommendations
            </Link>
            <Link
              href="/population"
              className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200 shadow-2xs transition"
            >
              Population Analytics
            </Link>
          </div>
        </div>

        {/* Real Live Metrics (Live Database Records) */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Assigned Cohort"
            value={loading ? '...' : patients.length}
            unit="patients"
            subtitle="Verified demographic profiles"
            icon={Users}
            variant="forest"
          />
          <MetricCard
            label="Screening Encounters"
            value={loading ? '...' : screenings.length}
            unit="events"
            subtitle="Scoped to facility session"
            icon={Activity}
          />
          <MetricCard
            label="Epistemic Quality Rate"
            value={loading ? '...' : `${verifiedRate}%`}
            subtitle="Verified observation provenance"
            trend={verifiedRate >= 80 ? 'Verified' : 'Review Required'}
            icon={ShieldCheck}
            variant="emerald"
          />
          <MetricCard
            label="Active Recommendations"
            value={loading ? '...' : recommendations.length}
            unit="actions"
            subtitle="IFHAS & Well-Child milestones"
            icon={HeartPulse}
            variant="amber"
          />
        </section>

        {/* Operational Split Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Screening Encounters (2 Columns) */}
          <section className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/90 shadow-xs p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  Clinical Screening Encounters
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Recent adult IFHAS and pediatric developmental screening observations.
                </p>
              </div>
              <Link
                href="/screening"
                className="text-xs font-semibold text-forest-700 hover:text-forest-800 flex items-center gap-1"
              >
                Log New Vitals <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {loading ? (
              <LoadingSpinner />
            ) : screenings.length === 0 ? (
              <EmptyState
                title="No Screening Encounters Registered"
                description="No clinical vitals or screening encounters have been logged for this facility yet."
                actionText="Record First Screening"
                onAction={() => router.push('/screening')}
              />
            ) : (
              <div className="overflow-x-auto">
                <table className="table-healthcare">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Encounter Type</th>
                      <th>Blood Pressure</th>
                      <th>Glucose</th>
                      <th>BMI</th>
                      <th>Epistemic Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {screenings.slice(0, 6).map((s) => (
                      <tr key={s.id} className="hover:bg-slate-50/80 transition">
                        <td className="font-mono text-slate-500">
                          {new Date(s.encounter_date || s.created_at).toLocaleDateString()}
                        </td>
                        <td className="font-medium text-slate-900">
                          {s.encounter_type.replace(/_/g, ' ')}
                        </td>
                        <td>
                          {s.systolic_bp && s.diastolic_bp ? `${s.systolic_bp}/${s.diastolic_bp} mmHg` : '—'}
                        </td>
                        <td className="font-mono">
                          {s.fasting_glucose ? `${s.fasting_glucose} mg/dL` : '—'}
                        </td>
                        <td className="font-semibold text-slate-800">{s.bmi || '—'}</td>
                        <td>
                          <StatusBadge status={s.epistemic_status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* Active Preventive Recommendations Queue (1 Column) */}
          <section className="bg-white rounded-2xl border border-slate-200/90 shadow-xs p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  Preventive Priorities
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">Clinical actions requiring review or recall.</p>
              </div>
              <Link
                href="/recommendations"
                className="text-xs font-semibold text-forest-700 hover:text-forest-800"
              >
                View All
              </Link>
            </div>

            {loading ? (
              <LoadingSpinner />
            ) : recommendations.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-200 rounded-xl">
                No pending recommendations found for this facility.
              </div>
            ) : (
              <div className="space-y-3">
                {recommendations.slice(0, 4).map((rec) => (
                  <div
                    key={rec.id}
                    className="p-3.5 rounded-xl border border-slate-200/90 hover:border-slate-300 bg-[#F8FAF9]/50 transition space-y-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <StatusBadge status={rec.priority} size="sm" />
                      <span className="text-[10px] font-mono text-slate-400">
                        {rec.rule_id}
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-slate-900 leading-snug line-clamp-2">
                      {rec.recommended_action}
                    </div>
                    <div className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                      {rec.clinical_rationale}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Quick Link to Clinical Pathways */}
            <div className="pt-2 border-t border-slate-100">
              <Link
                href="/pathways"
                className="w-full py-2 px-3 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-semibold text-xs transition flex items-center justify-between"
              >
                <span>Review Active Care Pathway Standards</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
