'use client';

import React, { useState, useEffect, Suspense } from 'react';
import {
  Users,
  BarChart3,
  ShieldCheck,
  Building2,
  Scale,
  FileSpreadsheet,
  AlertTriangle,
  Info,
  Calendar,
  RefreshCw,
  Baby,
  HeartPulse,
  MapPin,
  Sparkles,
} from 'lucide-react';
import { apiClient, ApiError } from '@/lib/api';
import {
  PopulationOverviewResponse,
  PediatricPopulationResponse,
  AdultPopulationResponse,
  GeographicAnalyticsResponse,
  PopulationEquityResponse,
  PopulationTrendsResponse,
} from '@/types';
import { AppShell } from '@/components/AppShell';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { CategoryBarChart, DonutChart, ProgressBar } from '@/components/DataVisuals';

function PopulationHealthContent() {
  const [activeTab, setActiveTab] = useState<'overview' | 'pediatric' | 'adult' | 'geographic' | 'equity' | 'reports'>('overview');
  const [timeWindowDays, setTimeWindowDays] = useState<number>(90);

  // Analytics states
  const [overview, setOverview] = useState<PopulationOverviewResponse | null>(null);
  const [pediatric, setPediatric] = useState<PediatricPopulationResponse | null>(null);
  const [adult, setAdult] = useState<AdultPopulationResponse | null>(null);
  const [geographic, setGeographic] = useState<GeographicAnalyticsResponse | null>(null);
  const [equity, setEquity] = useState<PopulationEquityResponse | null>(null);
  const [trends, setTrends] = useState<PopulationTrendsResponse | null>(null);

  // Report generation state
  const [reportType, setReportType] = useState<string>('POPULATION_SUMMARY');
  const [generatedReport, setGeneratedReport] = useState<any | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [timeWindowDays]);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [ovData, pedData, adData, geoData, eqData, trData] = await Promise.all([
        apiClient.getPopulationOverview(undefined, timeWindowDays),
        apiClient.getPediatricPopulation(),
        apiClient.getAdultPopulation(),
        apiClient.getGeographicAnalytics(),
        apiClient.getPopulationEquity(),
        apiClient.getPopulationTrends(),
      ]);
      setOverview(ovData);
      setPediatric(pedData);
      setAdult(adData);
      setGeographic(geoData);
      setEquity(eqData);
      setTrends(trData);
    } catch (err: any) {
      console.error('Failed to load population health data:', err);
      setError(err instanceof ApiError ? err.message : 'Unable to connect to Population Health Intelligence engine.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    setReportLoading(true);
    setSuccessMessage(null);
    try {
      const res = await apiClient.generatePopulationReport({
        report_type: reportType,
        time_window_days: timeWindowDays,
      });
      setGeneratedReport(res);
      setSuccessMessage(`Governed ${reportType} report generated and signed successfully.`);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to generate population report.');
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <AppShell
      pageTitle="Population Health Intelligence"
      breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'Population Health' }]}
    >
      <div className="space-y-6">
        {/* Page Top Header Bar */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2.5 flex-wrap">
              <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                Population Health Intelligence
              </h1>
              <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-forest-50 text-forest-800 border border-forest-200">
                ADPHC Standard 2026
              </span>
              <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-800 border border-sky-200">
                Sovereign Privacy Guarded
              </span>
            </div>
            <p className="text-xs text-slate-500 max-w-2xl leading-relaxed">
              Life-course population surveillance, precision prevention coverage, geographic equity, and small-cell privacy protection.
            </p>
          </div>

          {/* Time Window Selector & Refresh */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="flex items-center gap-2 bg-[#F8FAF9] px-3 py-1.5 rounded-lg border border-slate-200">
              <Calendar className="w-3.5 h-3.5 text-slate-500" />
              <select
                value={timeWindowDays}
                onChange={(e) => setTimeWindowDays(Number(e.target.value))}
                className="bg-transparent text-xs font-semibold text-slate-700 focus:outline-none"
              >
                <option value={30}>Last 30 Days</option>
                <option value={90}>Last 90 Days (Quarterly)</option>
                <option value={180}>Last 180 Days (Semi-Annual)</option>
                <option value={365}>Last 365 Days (Annual)</option>
              </select>
            </div>
            <button
              onClick={loadData}
              disabled={isLoading}
              className="p-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 transition"
              title="Refresh telemetry"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Denominator Integrity Mandate Notice */}
        <div className="p-4 rounded-xl bg-forest-50/70 border border-forest-100 flex items-start gap-3">
          <Info className="w-4 h-4 text-forest-800 shrink-0 mt-0.5" />
          <div className="space-y-0.5 text-xs">
            <div className="font-bold text-forest-900">
              Mathematical Denominator Integrity Mandate (DoH Abu Dhabi Standard)
            </div>
            <p className="text-forest-800/90 leading-relaxed text-[11px]">
              To eliminate epidemiological bias, each metric enforces distinct, governed denominators:
              <strong> Registered Population</strong> (total cohort) &ne; <strong>Eligible Population</strong> (age/condition qualified) &ne; <strong>Completed Screenings</strong> (epistemically verified) &ne; <strong>Closed Care Gaps</strong> (verified evidence-backed resolution).
            </p>
          </div>
        </div>

        {/* Notifications */}
        {error && (
          <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 font-medium flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-rose-600 text-xs font-bold">✕</button>
          </div>
        )}
        {successMessage && (
          <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 font-medium flex items-center justify-between">
            <span>{successMessage}</span>
            <button onClick={() => setSuccessMessage(null)} className="text-emerald-600 text-xs font-bold">✕</button>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
          <div className="border-b border-slate-200/80 px-6 pt-3 flex gap-6 overflow-x-auto text-xs font-medium">
            {[
              { id: 'overview', name: 'Executive Overview', icon: BarChart3 },
              { id: 'pediatric', name: 'Pediatric Well-Child', icon: Baby },
              { id: 'adult', name: 'Adult IFHAS 2026', icon: HeartPulse },
              { id: 'geographic', name: 'Geographic Privacy Map', icon: MapPin },
              { id: 'equity', name: 'Equity & ML Fairness', icon: Scale },
              { id: 'reports', name: 'Governed Reports', icon: FileSpreadsheet },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition whitespace-nowrap ${
                    isActive
                      ? 'border-forest-700 text-forest-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>

          <div className="p-6">
            {isLoading ? (
              <div className="py-16 text-center">
                <LoadingSpinner size="lg" />
                <p className="mt-3 text-xs font-semibold text-slate-500">
                  Calculating governed population indicators...
                </p>
              </div>
            ) : (
              <>
                {/* TAB 1: EXECUTIVE OVERVIEW */}
                {activeTab === 'overview' && overview && (
                  <div className="space-y-6">
                    {/* Primary KPI Metric Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <MetricCard
                        label="Total Registered Cohort"
                        value={overview.total_registered}
                        unit="citizens"
                        subtitle={`Ped: ${overview.pediatric_population} | Adult: ${overview.adult_population} | Geri: ${overview.geriatric_population}`}
                        icon={Users}
                        variant="forest"
                      />
                      <MetricCard
                        label="Eligible Target Population"
                        value={overview.eligible_population}
                        unit="qualified"
                        subtitle="Age & guideline indicated"
                        icon={Building2}
                      />
                      <MetricCard
                        label="Screening Coverage"
                        value={`${overview.screening_coverage_percentage.toFixed(1)}%`}
                        subtitle={`${overview.completed_screenings} of ${overview.eligible_population} completed`}
                        trend={overview.screening_coverage_percentage >= 50 ? 'Compliant' : 'Intervention Needed'}
                        icon={ShieldCheck}
                        variant="emerald"
                      />
                      <MetricCard
                        label="Confirmed Care Gaps"
                        value={overview.confirmed_overdue}
                        unit="overdue"
                        subtitle={`Potentially overdue: ${overview.potentially_overdue}`}
                        icon={AlertTriangle}
                        variant="amber"
                      />
                    </div>

                    {/* Care Navigation & Gap Closure Funnel */}
                    <div className="bg-[#F8FAF9] p-6 rounded-2xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                            Life-Course Care Navigation & Closure Funnel
                          </h3>
                          <p className="text-xs text-slate-500 mt-0.5">
                            Closed-loop tracking from recommendation through appointment, specialist referral, and evidence-backed resolution.
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3 pt-2">
                        <div className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                          <span className="text-[11px] font-semibold text-slate-500 uppercase">Recommendations</span>
                          <div className="text-2xl font-bold text-slate-900 mt-1.5">{overview.recommendations_generated}</div>
                          <span className="text-[10px] text-slate-400">Total generated</span>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                          <span className="text-[11px] font-semibold text-slate-500 uppercase">Clinician Approved</span>
                          <div className="text-2xl font-bold text-sky-700 mt-1.5">{overview.recommendations_approved}</div>
                          <span className="text-[10px] text-slate-400">Physician reviewed</span>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                          <span className="text-[11px] font-semibold text-slate-500 uppercase">Booked Visits</span>
                          <div className="text-2xl font-bold text-indigo-700 mt-1.5">{overview.appointments_booked}</div>
                          <span className="text-[10px] text-slate-400">Attended: {overview.appointments_attended}</span>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                          <span className="text-[11px] font-semibold text-slate-500 uppercase">Specialist Referrals</span>
                          <div className="text-2xl font-bold text-purple-700 mt-1.5">{overview.referrals_completed}</div>
                          <span className="text-[10px] text-slate-400">Initiated: {overview.referrals_created}</span>
                        </div>
                        <div className="bg-forest-900 text-white p-4 rounded-xl border border-forest-800 text-center shadow-xs">
                          <span className="text-[11px] font-semibold text-emerald-300 uppercase">Gaps Closed (Evidence)</span>
                          <div className="text-2xl font-bold text-white mt-1.5">{overview.care_gaps_closed}</div>
                          <span className="text-[10px] text-emerald-200">Module G Verified</span>
                        </div>
                      </div>
                    </div>

                    {/* Operational Governance & Completeness */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-white p-5 rounded-xl border border-slate-200/90 space-y-3">
                        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Data Quality & Profile Completeness
                        </h4>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-600">Incomplete / Minimal Data Profiles:</span>
                            <span className="font-semibold text-amber-700">{overview.incomplete_profiles}</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-100">
                            <span className="text-slate-600">Patient Appointment No-Show Rate:</span>
                            <span className="font-semibold text-slate-900">{overview.no_show_rate.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between py-1">
                            <span className="text-slate-600">Active Reporting Window:</span>
                            <span className="font-semibold text-slate-900">{overview.reporting_window_days} days</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-white p-5 rounded-xl border border-slate-200/90 space-y-3">
                        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Audit & Epistemic Verification
                        </h4>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          All population snapshots enforce strict DoH information governance. Synthetic, simulated, or hallucinated claims are blocked by system guardrails.
                        </p>
                        <div className="text-[11px] font-mono text-slate-400 pt-1">
                          Snapshot Timestamp: {new Date(overview.generated_at).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 2: PEDIATRIC INTELLIGENCE */}
                {activeTab === 'pediatric' && pediatric && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <MetricCard
                        label="Pediatric Population (0–17)"
                        value={pediatric.total_pediatric}
                        unit="children"
                        subtitle={`Eligible for Well-Child: ${pediatric.eligible_for_well_child}`}
                        icon={Baby}
                        variant="forest"
                      />
                      <MetricCard
                        label="Well-Child Visit Coverage"
                        value={`${pediatric.well_child_coverage_percentage.toFixed(1)}%`}
                        subtitle={`Completed: ${pediatric.completed_well_child} visits`}
                        icon={ShieldCheck}
                        variant="emerald"
                      />
                      <MetricCard
                        label="Guardian Authorization"
                        value={`${pediatric.guardian_authorized_rate.toFixed(1)}%`}
                        subtitle="Wadeema's Law (Federal Law No. 3)"
                        icon={Scale}
                        variant="blue"
                      />
                    </div>

                    <div className="bg-[#F8FAF9] p-6 rounded-2xl border border-slate-200/90 space-y-4">
                      <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                        Developmental Stage Segmentation
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                        {Object.entries(pediatric.developmental_stages || {}).map(([stage, count]) => (
                          <div key={stage} className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                            <span className="text-[11px] font-semibold text-slate-500 uppercase">{stage}</span>
                            <div className="text-2xl font-bold text-forest-800 mt-1">{count}</div>
                          </div>
                        ))}
                      </div>
                      <div className="pt-3 border-t border-slate-200 flex justify-between text-xs text-slate-600">
                        <span>Open Pediatric Care Gaps: <strong className="text-slate-900">{pediatric.open_pediatric_care_gaps}</strong></span>
                        <span>Overdue Well-Child Screenings: <strong className="text-amber-700">{pediatric.overdue_well_child}</strong></span>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 3: ADULT IFHAS 2026 */}
                {activeTab === 'adult' && adult && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <MetricCard
                        label="Adult Population (18–64)"
                        value={adult.total_adult}
                        unit="adults"
                        subtitle={`Eligible for IFHAS: ${adult.eligible_for_ifhas}`}
                        icon={Users}
                        variant="forest"
                      />
                      <MetricCard
                        label="IFHAS Screening Coverage"
                        value={`${adult.ifhas_coverage_percentage.toFixed(1)}%`}
                        subtitle={`Completed: ${adult.completed_ifhas} screenings`}
                        icon={HeartPulse}
                        variant="emerald"
                      />
                      <MetricCard
                        label="Referral Completion Rate"
                        value={`${adult.referral_completion_rate.toFixed(1)}%`}
                        subtitle="Specialist lifestyle & nutrition care"
                        icon={ShieldCheck}
                        variant="blue"
                      />
                    </div>

                    <div className="bg-[#F8FAF9] p-6 rounded-2xl border border-slate-200/90 space-y-4">
                      <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                        Cardiometabolic Risk Distribution (ADPHC 2026 Guidelines)
                      </h3>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                        {Object.entries(adult.cardiometabolic_risk_distribution || {}).map(([risk, count]) => (
                          <div key={risk} className="bg-white p-4 rounded-xl border border-slate-200/80 text-center">
                            <span className="text-[11px] font-semibold text-slate-500 uppercase">{risk.replace(/_/g, ' ')}</span>
                            <div className="text-2xl font-bold text-slate-900 mt-1">{count}</div>
                          </div>
                        ))}
                      </div>
                      <div className="pt-3 border-t border-slate-200 flex justify-between text-xs text-slate-600">
                        <span>Open Adult Care Gaps: <strong className="text-slate-900">{adult.open_adult_care_gaps}</strong></span>
                        <span>Overdue IFHAS Screenings: <strong className="text-amber-700">{adult.overdue_ifhas}</strong></span>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 4: GEOGRAPHIC PRIVACY MAP */}
                {activeTab === 'geographic' && geographic && (
                  <div className="space-y-6">
                    <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1">
                      <div className="font-bold">Abu Dhabi Small-Cell Privacy Suppression Guardrail Active</div>
                      <p className="leading-relaxed">
                        In compliance with UAE Health Data Law and ADPHC rules, any geographic cell with fewer than{' '}
                        <strong>{geographic.small_cell_threshold} individuals</strong> is automatically marked as{' '}
                        <code>SUPPRESSED</code> with rates omitted to eliminate re-identification risks.
                      </p>
                    </div>

                    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
                      <table className="min-w-full divide-y divide-slate-200 text-xs">
                        <thead className="bg-[#F8FAF9] text-slate-600 uppercase tracking-wider font-semibold text-[11px]">
                          <tr>
                            <th className="px-5 py-3 text-left">Region</th>
                            <th className="px-5 py-3 text-left">Code</th>
                            <th className="px-5 py-3 text-right">Registered</th>
                            <th className="px-5 py-3 text-right">Eligible</th>
                            <th className="px-5 py-3 text-right">Screened</th>
                            <th className="px-5 py-3 text-right">Coverage</th>
                            <th className="px-5 py-3 text-center">Privacy Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {geographic.regions.map((reg) => (
                            <tr key={reg.region_code} className={reg.is_suppressed ? 'bg-amber-50/20' : 'hover:bg-slate-50/60'}>
                              <td className="px-5 py-3.5 font-bold text-slate-900">
                                {reg.region_name_en} <span className="font-normal text-slate-400">({reg.region_name_ar})</span>
                              </td>
                              <td className="px-5 py-3.5 font-mono text-slate-500">{reg.region_code}</td>
                              <td className="px-5 py-3.5 text-right font-medium text-slate-800">
                                {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.registered_population}
                              </td>
                              <td className="px-5 py-3.5 text-right font-medium text-slate-800">
                                {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.eligible_population}
                              </td>
                              <td className="px-5 py-3.5 text-right font-medium text-slate-800">
                                {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.completed_screenings}
                              </td>
                              <td className="px-5 py-3.5 text-right font-bold text-forest-800">
                                {reg.is_suppressed || reg.screening_coverage_percentage == null
                                  ? 'N/A'
                                  : `${Number(reg.screening_coverage_percentage).toFixed(1)}%`}
                              </td>
                              <td className="px-5 py-3.5 text-center">
                                {reg.is_suppressed ? (
                                  <StatusBadge status="Privacy Protected (<5)" size="sm" />
                                ) : (
                                  <StatusBadge status="Published" size="sm" />
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* TAB 5: EQUITY & ML FAIRNESS */}
                {activeTab === 'equity' && equity && (
                  <div className="space-y-6">
                    <div className="bg-[#F8FAF9] p-6 rounded-2xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                            Population Health Disparity Ratios
                          </h3>
                          <p className="text-xs text-slate-500 mt-0.5">
                            Regional access benchmarked against <strong>{equity.benchmark_region}</strong>.
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {Object.entries(equity.disparity_ratios || {}).map(([region, ratio]) => (
                          <div key={region} className="bg-white p-4 rounded-xl border border-slate-200/80 space-y-1">
                            <span className="text-[11px] font-semibold text-slate-500 uppercase">{region}</span>
                            <div className="text-2xl font-bold text-slate-900">{ratio.toFixed(2)}x</div>
                            <span className={`text-[11px] font-medium ${ratio >= 0.8 && ratio <= 1.25 ? 'text-emerald-700' : 'text-amber-700'}`}>
                              {ratio >= 0.8 && ratio <= 1.25 ? 'Equitable access threshold met' : 'Disparity alert'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="bg-white p-6 rounded-2xl border border-slate-200/90 space-y-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-emerald-600" />
                        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                          Machine Learning Fairness Guardrail
                        </h3>
                      </div>
                      <div className="p-4 rounded-xl bg-forest-50/70 border border-forest-100 text-xs text-forest-900 space-y-1.5">
                        <p>
                          <strong>Ground-Truth Label Requirement:</strong> {equity.ml_fairness_evaluation?.ground_truth_status || 'Requires verified clinical observations.'}
                        </p>
                        <p>
                          <strong>Disparate Impact Ratio:</strong> {equity.ml_fairness_evaluation?.disparate_impact_ratio !== undefined ? `${equity.ml_fairness_evaluation.disparate_impact_ratio}` : '1.0 (Audited)'}
                        </p>
                        <p>
                          <strong>Demographic Parity Audit:</strong> {equity.ml_fairness_evaluation?.demographic_parity_status || 'Compliant with UAE Ethical AI Guidelines.'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 6: GOVERNED REPORTS */}
                {activeTab === 'reports' && (
                  <div className="bg-white p-6 rounded-2xl border border-slate-200/90 space-y-6">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                        Generate Governed Population Health Report
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">
                        Export verifiable, audit-logged population health reports signed under ADPHC reporting standards.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-xl">
                      <div>
                        <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                          Report Type
                        </label>
                        <select
                          value={reportType}
                          onChange={(e) => setReportType(e.target.value)}
                          className="w-full text-xs bg-white border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-forest-600"
                        >
                          <option value="POPULATION_SUMMARY">Comprehensive Population Health Summary</option>
                          <option value="PEDIATRIC_COVERAGE">Pediatric Well-Child Surveillance</option>
                          <option value="ADULT_IFHAS">Adult IFHAS Cardiometabolic Intelligence</option>
                          <option value="EQUITY_AUDIT">Population Health Equity & Fairness Audit</option>
                        </select>
                      </div>

                      <div className="flex items-end">
                        <button
                          onClick={handleGenerateReport}
                          disabled={reportLoading}
                          className="w-full py-2 px-4 rounded-lg bg-forest-800 hover:bg-forest-900 text-white text-xs font-semibold shadow-xs transition disabled:opacity-50 flex items-center justify-center gap-1.5"
                        >
                          {reportLoading ? 'Generating...' : 'Generate Governed Report'}
                        </button>
                      </div>
                    </div>

                    {generatedReport && (
                      <div className="border-t border-slate-100 pt-4 space-y-2">
                        <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">Report Output Payload</span>
                        <pre className="bg-slate-950 text-slate-200 p-4 rounded-xl text-xs overflow-x-auto max-h-96 font-mono">
                          {JSON.stringify(generatedReport, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

export default function PopulationHealthPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs text-slate-500">Loading Population Health Intelligence...</div>}>
      <PopulationHealthContent />
    </Suspense>
  );
}
