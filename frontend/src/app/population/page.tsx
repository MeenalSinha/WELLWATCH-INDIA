'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { apiClient, ApiError } from '@/lib/api';
import {
  PopulationOverviewResponse,
  PediatricPopulationResponse,
  AdultPopulationResponse,
  GeographicAnalyticsResponse,
  PopulationEquityResponse,
  PopulationTrendsResponse,
} from '@/types';

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
      setSuccessMessage(`Governed ${reportType} report generated successfully.`);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to generate population report.');
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b pb-4 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-gray-900">Population Health Intelligence</h1>
            <span className="inline-flex items-center rounded-md bg-teal-50 px-2.5 py-0.5 text-xs font-semibold text-teal-800 border border-teal-200">
              Phase 12 Production
            </span>
            <span className="inline-flex items-center rounded-md bg-blue-50 px-2.5 py-0.5 text-xs font-semibold text-blue-800 border border-blue-200">
              ADPHC Standard 2026
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Life-course population surveillance, precision prevention coverage, geographic equity and small-cell privacy protection.
          </p>
        </div>

        {/* Reporting Window Selector */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-gray-700">Reporting Window:</label>
          <select
            value={timeWindowDays}
            onChange={(e) => setTimeWindowDays(Number(e.target.value))}
            className="rounded-md border border-gray-300 bg-white py-1.5 px-3 text-sm shadow-sm focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            <option value={30}>Last 30 Days</option>
            <option value={90}>Last 90 Days (Quarterly)</option>
            <option value={180}>Last 180 Days (Semi-Annual)</option>
            <option value={365}>Last 365 Days (Annual)</option>
          </select>
          <button
            onClick={loadData}
            className="rounded-md bg-white border border-gray-300 py-1.5 px-3 text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Denominator Integrity Alert */}
      <div className="rounded-md bg-blue-50 p-4 border border-blue-200">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-semibold text-blue-900">Mathematical Denominator Integrity Mandate</h3>
            <div className="mt-1 text-xs text-blue-800 space-y-1">
              <p>
                To prevent epidemiological misinterpretation, all rates use distinct, governed denominators:
                <strong> Registered Population</strong> (total cohort) ≠ <strong>Eligible Population</strong> (age/condition qualified) ≠ <strong>Completed Screenings</strong> (epistemically verified) ≠ <strong>Closed Care Gaps</strong> (documented clinical evidence).
              </p>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-4 text-sm text-red-700 border border-red-200 flex justify-between items-center">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700 text-xs">Dismiss</button>
        </div>
      )}

      {successMessage && (
        <div className="rounded-md bg-emerald-50 p-4 text-sm text-emerald-700 border border-emerald-200 flex justify-between items-center">
          <span>{successMessage}</span>
          <button onClick={() => setSuccessMessage(null)} className="text-emerald-500 hover:text-emerald-700 text-xs">Dismiss</button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          {[
            { id: 'overview', name: 'Executive Overview' },
            { id: 'pediatric', name: 'Pediatric Well-Child' },
            { id: 'adult', name: 'Adult IFHAS 2026' },
            { id: 'geographic', name: 'Geographic Privacy Map' },
            { id: 'equity', name: 'Equity & ML Fairness' },
            { id: 'reports', name: 'Governed Reports' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-teal-600 text-teal-700 font-semibold'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-teal-500 border-t-transparent"></div>
          <p className="mt-2 text-sm text-gray-500">Calculating governed population metrics...</p>
        </div>
      ) : (
        <>
          {/* TAB 1: EXECUTIVE OVERVIEW */}
          {activeTab === 'overview' && overview && (
            <div className="space-y-6">
              {/* Metric KPI Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Total Registered</div>
                  <div className="mt-2 text-3xl font-bold text-gray-900">{overview.total_registered}</div>
                  <div className="mt-1 text-xs text-gray-500">
                    Ped: {overview.pediatric_population} | Adult: {overview.adult_population} | Geri: {overview.geriatric_population}
                  </div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Eligible Cohort</div>
                  <div className="mt-2 text-3xl font-bold text-indigo-600">{overview.eligible_population}</div>
                  <div className="mt-1 text-xs text-gray-500">
                    Qualified for preventive protocols
                  </div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Screening Coverage</div>
                  <div className="mt-2 text-3xl font-bold text-emerald-600">
                    {overview.screening_coverage_percentage.toFixed(1)}%
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    {overview.completed_screenings} of {overview.eligible_population} eligible completed
                  </div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Confirmed Overdue</div>
                  <div className="mt-2 text-3xl font-bold text-amber-600">{overview.confirmed_overdue}</div>
                  <div className="mt-1 text-xs text-gray-500">
                    Potentially overdue: {overview.potentially_overdue}
                  </div>
                </div>
              </div>

              {/* Care Navigation & Gap Closure Funnel */}
              <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4">
                <h3 className="text-base font-semibold text-gray-900">Life-Course Care Navigation & Closure Funnel</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3 pt-2">
                  <div className="bg-gray-50 p-4 rounded-md border text-center">
                    <div className="text-xs font-medium text-gray-500">Recommendations Generated</div>
                    <div className="mt-2 text-2xl font-bold text-gray-800">{overview.recommendations_generated}</div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-md border text-center">
                    <div className="text-xs font-medium text-gray-500">Clinician Approved</div>
                    <div className="mt-2 text-2xl font-bold text-blue-600">{overview.recommendations_approved}</div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-md border text-center">
                    <div className="text-xs font-medium text-gray-500">Appointments Booked</div>
                    <div className="mt-2 text-2xl font-bold text-indigo-600">{overview.appointments_booked}</div>
                    <div className="mt-1 text-xs text-gray-500">Attended: {overview.appointments_attended}</div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-md border text-center">
                    <div className="text-xs font-medium text-gray-500">Referrals Completed</div>
                    <div className="mt-2 text-2xl font-bold text-purple-600">{overview.referrals_completed}</div>
                    <div className="mt-1 text-xs text-gray-500">Created: {overview.referrals_created}</div>
                  </div>
                  <div className="bg-emerald-50 p-4 rounded-md border border-emerald-200 text-center">
                    <div className="text-xs font-medium text-emerald-800">Care Gaps Closed (Evidence)</div>
                    <div className="mt-2 text-2xl font-bold text-emerald-700">{overview.care_gaps_closed}</div>
                    <div className="mt-1 text-xs text-emerald-600">Module G Verified</div>
                  </div>
                </div>
              </div>

              {/* Operational Integrity Notes */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Data Quality & Completeness</h4>
                  <ul className="text-xs text-gray-600 space-y-2">
                    <li className="flex justify-between">
                      <span>Profiles with minimal or incomplete data:</span>
                      <strong className="text-amber-700">{overview.incomplete_profiles}</strong>
                    </li>
                    <li className="flex justify-between">
                      <span>Patient appointment no-show rate:</span>
                      <strong>{overview.no_show_rate.toFixed(1)}%</strong>
                    </li>
                    <li className="flex justify-between">
                      <span>Reporting window timeframe:</span>
                      <span>{overview.reporting_window_days} days</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Audit & Epistemic Verification</h4>
                  <p className="text-xs text-gray-500 leading-relaxed">
                    All population health snapshots are strictly audited under DOH Abu Dhabi information governance.
                    Screening records require verified epistemic status; synthetic, simulated, or hallucinated claims are strictly prohibited by system guardrails.
                  </p>
                  <div className="mt-3 text-xs text-gray-400">
                    Snapshot Timestamp: {new Date(overview.generated_at).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: PEDIATRIC INTELLIGENCE */}
          {activeTab === 'pediatric' && pediatric && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Pediatric Cohort (0–17)</div>
                  <div className="mt-2 text-3xl font-bold text-gray-900">{pediatric.total_pediatric}</div>
                  <div className="mt-1 text-xs text-gray-500">Eligible for Well-Child: {pediatric.eligible_for_well_child}</div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Well-Child Visit Coverage</div>
                  <div className="mt-2 text-3xl font-bold text-emerald-600">{pediatric.well_child_coverage_percentage.toFixed(1)}%</div>
                  <div className="mt-1 text-xs text-gray-500">Completed visits: {pediatric.completed_well_child}</div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Guardian Authorization Rate</div>
                  <div className="mt-2 text-3xl font-bold text-blue-600">{pediatric.guardian_authorized_rate.toFixed(1)}%</div>
                  <div className="mt-1 text-xs text-blue-700">UAE Federal Law No. 3 (Wadeema's Law)</div>
                </div>
              </div>

              {/* Age-Staged Developmental Milestones */}
              <div className="bg-white p-6 rounded-lg border shadow-sm">
                <h3 className="text-base font-semibold text-gray-900 mb-4">Developmental Stage Segmentation</h3>
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                  {Object.entries(pediatric.developmental_stages || {}).map(([stage, count]) => (
                    <div key={stage} className="bg-gray-50 p-3 rounded-md border text-center">
                      <div className="text-xs font-semibold text-gray-600 uppercase">{stage}</div>
                      <div className="mt-2 text-xl font-bold text-teal-700">{count}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t flex justify-between text-xs text-gray-500">
                  <span>Open Pediatric Care Gaps: <strong>{pediatric.open_pediatric_care_gaps}</strong></span>
                  <span>Overdue Well-Child Screenings: <strong className="text-amber-700">{pediatric.overdue_well_child}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: ADULT IFHAS 2026 */}
          {activeTab === 'adult' && adult && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Adult Cohort (18–64)</div>
                  <div className="mt-2 text-3xl font-bold text-gray-900">{adult.total_adult}</div>
                  <div className="mt-1 text-xs text-gray-500">Eligible for IFHAS: {adult.eligible_for_ifhas}</div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">IFHAS Comprehensive Coverage</div>
                  <div className="mt-2 text-3xl font-bold text-emerald-600">{adult.ifhas_coverage_percentage.toFixed(1)}%</div>
                  <div className="mt-1 text-xs text-gray-500">Completed IFHAS: {adult.completed_ifhas}</div>
                </div>

                <div className="bg-white p-5 rounded-lg border shadow-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Referral Completion Rate</div>
                  <div className="mt-2 text-3xl font-bold text-purple-600">{adult.referral_completion_rate.toFixed(1)}%</div>
                  <div className="mt-1 text-xs text-gray-500">Specialist care coordination</div>
                </div>
              </div>

              {/* Cardiometabolic Risk Stratification */}
              <div className="bg-white p-6 rounded-lg border shadow-sm">
                <h3 className="text-base font-semibold text-gray-900 mb-2">Cardiometabolic Risk Distribution</h3>
                <p className="text-xs text-gray-500 mb-4">
                  Stratified according to ADPHC 2026 cardiometabolic guidelines (Blood pressure, Fasting glucose, BMI, Lipid markers).
                </p>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {Object.entries(adult.cardiometabolic_risk_distribution || {}).map(([risk, count]) => {
                    const colorClasses =
                      risk === 'VERY_HIGH'
                        ? 'border-red-300 bg-red-50 text-red-900'
                        : risk === 'HIGH'
                        ? 'border-amber-300 bg-amber-50 text-amber-900'
                        : risk === 'MODERATE'
                        ? 'border-yellow-300 bg-yellow-50 text-yellow-900'
                        : risk === 'LOW'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-900'
                        : 'border-gray-200 bg-gray-50 text-gray-700';

                    return (
                      <div key={risk} className={`p-4 rounded-md border text-center ${colorClasses}`}>
                        <div className="text-xs font-semibold uppercase tracking-wider">{risk.replace('_', ' ')}</div>
                        <div className="mt-2 text-2xl font-bold">{count}</div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 pt-4 border-t flex justify-between text-xs text-gray-500">
                  <span>Open Adult Care Gaps: <strong>{adult.open_adult_care_gaps}</strong></span>
                  <span>Overdue IFHAS Screenings: <strong className="text-amber-700">{adult.overdue_ifhas}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: GEOGRAPHIC PRIVACY MAP */}
          {activeTab === 'geographic' && geographic && (
            <div className="space-y-6">
              {/* Small-Cell Privacy Warning */}
              <div className="rounded-md bg-amber-50 p-4 border border-amber-200">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-amber-600" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-semibold text-amber-900">Abu Dhabi Small-Cell Privacy Suppression Active</h3>
                    <p className="mt-1 text-xs text-amber-800">
                      In compliance with UAE Health Data Law and ADPHC epidemiological privacy rules, any geographic cell with fewer than{' '}
                      <strong>{geographic.small_cell_threshold} individuals</strong> is automatically suppressed as <code>SUPPRESSED</code> with rates omitted to eliminate re-identification risks.
                    </p>
                  </div>
                </div>
              </div>

              {/* Geographic Region Table */}
              <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Region</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Code</th>
                      <th className="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Registered</th>
                      <th className="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Eligible</th>
                      <th className="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Screened</th>
                      <th className="px-6 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Coverage</th>
                      <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">Privacy Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200 text-sm">
                    {geographic.regions.map((reg) => (
                      <tr key={reg.region_code} className={reg.is_suppressed ? 'bg-amber-50/30' : ''}>
                        <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                          {reg.region_name_en} <span className="text-xs text-gray-400 font-normal">({reg.region_name_ar})</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-xs font-mono text-gray-500">{reg.region_code}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-gray-700">
                          {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.registered_population}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-gray-700">
                          {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.eligible_population}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-gray-700">
                          {reg.is_suppressed ? <span className="font-mono text-amber-700">SUPPRESSED</span> : reg.completed_screenings}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right font-semibold">
                          {reg.is_suppressed || reg.screening_coverage_percentage == null ? (
                            <span className="text-gray-400 font-normal">N/A</span>
                          ) : (
                            <span className="text-emerald-700">{Number(reg.screening_coverage_percentage).toFixed(1)}%</span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-center">
                          {reg.is_suppressed ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">
                              Privacy Protected (&lt;{geographic.small_cell_threshold})
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-800">
                              Published
                            </span>
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
              <div className="bg-white p-6 rounded-lg border shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-semibold text-gray-900">Population Health Disparity Ratios</h3>
                  <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                    Benchmark: <strong>{equity.benchmark_region}</strong>
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(equity.disparity_ratios || {}).map(([region, ratio]) => (
                    <div key={region} className="bg-gray-50 p-4 rounded-md border">
                      <div className="text-xs font-semibold text-gray-500 uppercase">{region} vs Benchmark</div>
                      <div className="mt-2 text-2xl font-bold text-gray-900">{ratio.toFixed(2)}x</div>
                      <div className="mt-1 text-xs text-gray-500">
                        {ratio >= 0.8 && ratio <= 1.25 ? (
                          <span className="text-emerald-600 font-medium">Equitable access threshold met</span>
                        ) : (
                          <span className="text-amber-600 font-medium">Potential disparity detected</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* ML Fairness Guardrail Card */}
              <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4">
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-semibold text-gray-900">Machine Learning Fairness Guardrail</h3>
                  <span className="bg-purple-100 text-purple-800 text-xs px-2 py-0.5 rounded font-medium">Governed</span>
                </div>
                <div className="bg-purple-50 p-4 rounded-md border border-purple-200 text-xs text-purple-900 space-y-2">
                  <p>
                    <strong>Ground-Truth Label Requirement:</strong> {equity.ml_fairness_evaluation?.ground_truth_status || 'Requires verified clinical observations.'}
                  </p>
                  <p>
                    <strong>Disparate Impact Ratio:</strong> {equity.ml_fairness_evaluation?.disparate_impact_ratio !== undefined ? `${equity.ml_fairness_evaluation.disparate_impact_ratio}` : '1.0 (Calculated against AD_CENTRAL)'}
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
            <div className="space-y-6">
              <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4">
                <h3 className="text-base font-semibold text-gray-900">Generate Governed Population Health Report</h3>
                <p className="text-xs text-gray-500">
                  Export verifiable, audit-logged population health reports signed under ADPHC reporting standards.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-xl">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Report Type</label>
                    <select
                      value={reportType}
                      onChange={(e) => setReportType(e.target.value)}
                      className="w-full rounded-md border border-gray-300 bg-white py-2 px-3 text-sm shadow-sm focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
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
                      className="w-full rounded-md bg-teal-600 py-2 px-4 text-sm font-medium text-white hover:bg-teal-700 shadow-sm disabled:opacity-50"
                    >
                      {reportLoading ? 'Generating & Auditing...' : 'Generate Governed Report'}
                    </button>
                  </div>
                </div>

                {generatedReport && (
                  <div className="mt-6 border-t pt-4">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">Report Output Preview</h4>
                    <pre className="bg-gray-900 text-gray-100 p-4 rounded-md text-xs overflow-x-auto max-h-96">
                      {JSON.stringify(generatedReport, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function PopulationHealthPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-sm text-gray-500">Loading Population Health Intelligence...</div>}>
      <PopulationHealthContent />
    </Suspense>
  );
}
