'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import {
  PreventiveRecommendationResponse,
  RecommendationCategory,
  RecommendationPriority,
  RecommendationLifecycleStatus,
  RecommendationExplanation,
  PersonSummary,
} from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import { MetricCard } from '@/components/MetricCard';
import { EmptyState } from '@/components/EmptyState';
import {
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Clock,
  Filter,
  Info,
  Calendar,
  AlertTriangle,
  UserCheck,
  Zap,
  X,
  FileText,
  AlertCircle,
} from 'lucide-react';

function RecommendationsContent() {
  const searchParams = useSearchParams();
  const initialPatientId = searchParams.get('patientId') || '';

  const [patientId, setPatientId] = useState(initialPatientId);
  const [recommendations, setRecommendations] = useState<PreventiveRecommendationResponse[]>([]);
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedRec, setSelectedRec] = useState<PreventiveRecommendationResponse | null>(null);
  const [explanation, setExplanation] = useState<RecommendationExplanation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [filterPriority, setFilterPriority] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('');

  // Review Modal State
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [reviewDecision, setReviewDecision] = useState<string>('APPROVED');
  const [reviewRationale, setReviewRationale] = useState<string>('');
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);

  // Status Update / Decline Modal
  const [showStatusModal, setShowStatusModal] = useState(false);
  const [targetStatus, setTargetStatus] = useState<string>('OFFERED');
  const [declinedReason, setDeclinedReason] = useState<string>('');

  useEffect(() => {
    loadPatients();
  }, []);

  useEffect(() => {
    loadRecommendations();
  }, [patientId, filterPriority, filterStatus]);

  async function loadPatients() {
    try {
      const res = await apiClient.getPatients(1, 100);
      setPatients(res.items);
    } catch (err: any) {
      console.error('Failed to load patients', err);
    }
  }

  async function loadRecommendations() {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiClient.getRecommendations(
        patientId || undefined,
        filterStatus || undefined,
        filterPriority || undefined,
        100,
        0
      );
      setRecommendations(res.items);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to fetch recommendations');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleGenerate() {
    if (!patientId) {
      alert('Please select a resident record first.');
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const newRecs = await apiClient.generateRecommendations(patientId);
      loadRecommendations();
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to generate recommendations');
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleViewExplanation(rec: PreventiveRecommendationResponse) {
    setSelectedRec(rec);
    try {
      const exp = await apiClient.explainRecommendation(rec.id);
      setExplanation(exp);
    } catch (err: any) {
      alert('Failed to load explanation: ' + err.message);
    }
  }

  async function submitReview() {
    if (!selectedRec) return;
    if (!reviewRationale.trim()) {
      alert('Please provide clinical review rationale.');
      return;
    }
    setIsSubmittingReview(true);
    try {
      await apiClient.reviewRecommendation(selectedRec.id, reviewDecision, reviewRationale);
      setShowReviewModal(false);
      setReviewRationale('');
      loadRecommendations();
    } catch (err: any) {
      alert('Clinical review submission failed: ' + err.message);
    } finally {
      setIsSubmittingReview(false);
    }
  }

  async function submitStatusUpdate() {
    if (!selectedRec) return;
    if (targetStatus === 'DECLINED' && !declinedReason.trim()) {
      alert('Please document the reason for declining this recommendation.');
      return;
    }
    try {
      await apiClient.updateRecommendationStatus(
        selectedRec.id,
        targetStatus,
        targetStatus === 'DECLINED' ? declinedReason : undefined
      );
      setShowStatusModal(false);
      setDeclinedReason('');
      loadRecommendations();
    } catch (err: any) {
      alert('Status update failed: ' + err.message);
    }
  }

  // Metric counts
  const criticalCount = recommendations.filter((r) => r.priority === 'CRITICAL').length;
  const pendingCount = recommendations.filter((r) => r.lifecycle_status === 'PENDING_CLINICAL_REVIEW').length;
  const approvedCount = recommendations.filter((r) => r.lifecycle_status === 'APPROVED' || r.lifecycle_status === 'COMPLETED').length;

  return (
    <AppShell activePage="/recommendations">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Intelligence', 'Preventive Guidance']}
          title="Personalized Preventive Recommendations"
          subtitle="Abu Dhabi Public Health Center (ADPHC) &amp; DoH Evidence-Backed Clinical Guidance Engine."
          action={
            patientId
              ? {
                  label: isGenerating ? 'Generating...' : 'Synthesize Recommendations',
                  icon: Zap,
                  onClick: handleGenerate,
                }
              : undefined
          }
        />

        {/* Metric Cards Row */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <MetricCard
            title="Total Active Orders"
            value={recommendations.length}
            icon={Sparkles}
            subtitle="Filtered recommendations"
            variant="forest"
          />
          <MetricCard
            title="Critical Priority"
            value={criticalCount}
            icon={AlertTriangle}
            subtitle="Immediate clinical risk"
            variant="amber"
          />
          <MetricCard
            title="Awaiting MD Review"
            value={pendingCount}
            icon={Clock}
            subtitle="Clinical sign-off needed"
            variant="teal"
          />
          <MetricCard
            title="Clinically Approved"
            value={approvedCount}
            icon={CheckCircle2}
            subtitle="Certified preventive pathways"
            variant="emerald"
          />
        </div>

        {/* Filters and Patient Selector */}
        <div className="card-healthcare p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="flex items-center gap-3 w-full md:w-auto">
            <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex-shrink-0">
              Resident:
            </span>
            <select
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none w-full md:w-80"
            >
              <option value="">-- All Enrolled Residents --</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs)
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Priority:
              </span>
              <select
                value={filterPriority}
                onChange={(e) => setFilterPriority(e.target.value)}
                className="px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
              >
                <option value="">All Priorities</option>
                <option value="CRITICAL">Critical</option>
                <option value="HIGH">High</option>
                <option value="MEDIUM">Medium</option>
                <option value="ROUTINE">Routine</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Lifecycle:
              </span>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
              >
                <option value="">All Lifecycle States</option>
                <option value="PENDING_CLINICAL_REVIEW">Pending Review</option>
                <option value="APPROVED">Clinically Approved</option>
                <option value="OFFERED">Offered</option>
                <option value="ACCEPTED">Accepted</option>
                <option value="DECLINED">Declined</option>
                <option value="SCHEDULED">Scheduled</option>
                <option value="COMPLETED">Completed</option>
              </select>
            </div>

            {(filterPriority || filterStatus || patientId) && (
              <button
                onClick={() => {
                  setFilterPriority('');
                  setFilterStatus('');
                  setPatientId('');
                }}
                className="text-xs text-forest-700 hover:text-forest-900 font-semibold underline ml-1"
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Main Recommendations Table */}
        <div className="card-healthcare overflow-hidden">
          {isLoading ? (
            <div className="p-12 text-center text-slate-400">
              <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <span>Loading evidence-backed recommendations...</span>
            </div>
          ) : recommendations.length === 0 ? (
            <div className="p-8">
              <EmptyState
                title="No Preventive Guidance Found"
                description={
                  patientId
                    ? 'No recommendations currently recorded for this resident. Click "Synthesize Recommendations" above to run the ADPHC guidance engine.'
                    : 'Select a resident to generate or review personalized clinical recommendations.'
                }
                action={
                  patientId
                    ? {
                        label: 'Synthesize Guidance Now',
                        onClick: handleGenerate,
                      }
                    : undefined
                }
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-healthcare">
                <thead>
                  <tr>
                    <th className="w-52 pl-6">Category &amp; Rule</th>
                    <th className="min-w-[280px]">Recommended Action</th>
                    <th className="w-28">Priority</th>
                    <th className="w-44">Lifecycle State</th>
                    <th className="w-36">Target Window</th>
                    <th className="w-56 text-right pr-6">Physician Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {recommendations.map((rec) => (
                    <tr key={rec.id} className="hover:bg-slate-50/70 transition-colors">
                      <td className="pl-6 align-top py-4">
                        <span className="font-semibold text-slate-800 text-xs block leading-tight">
                          {rec.category.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())}
                        </span>
                        <span className="inline-block mt-1.5 font-mono text-[10px] text-slate-500 bg-slate-100/80 px-2 py-0.5 rounded border border-slate-200">
                          {rec.rule_id}
                        </span>
                      </td>
                      <td className="align-top py-4 max-w-md pr-4">
                        <p className="font-semibold text-slate-900 text-xs leading-snug">
                          {rec.recommended_action}
                        </p>
                        <p className="text-[11px] text-slate-500 line-clamp-2 mt-1 leading-relaxed">
                          {rec.clinical_rationale}
                        </p>
                      </td>
                      <td className="align-top py-4 whitespace-nowrap">
                        <StatusBadge status={rec.priority} />
                      </td>
                      <td className="align-top py-4 whitespace-nowrap">
                        <StatusBadge status={rec.lifecycle_status} />
                      </td>
                      <td className="align-top py-4 whitespace-nowrap">
                        <div className="flex items-center gap-1.5 text-xs text-slate-600 font-medium">
                          <Calendar className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                          <span>
                            {rec.target_date
                              ? new Date(rec.target_date).toLocaleDateString()
                              : 'Immediate'}
                          </span>
                        </div>
                      </td>
                      <td className="align-top py-4 pr-6 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleViewExplanation(rec)}
                            className="px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-white text-slate-700 hover:text-forest-800 hover:bg-forest-50 border border-slate-200 shadow-2xs transition"
                            title="View governed clinical explanation and trigger rule"
                          >
                            Explain
                          </button>
                          {rec.lifecycle_status === 'PENDING_CLINICAL_REVIEW' && (
                            <button
                              onClick={() => {
                                setSelectedRec(rec);
                                setShowReviewModal(true);
                              }}
                              className="px-3 py-1.5 text-xs font-bold rounded-lg bg-amber-500 hover:bg-amber-600 text-white shadow-xs transition"
                              title="Clinical physician review sign-off"
                            >
                              Review
                            </button>
                          )}
                          <button
                            onClick={() => {
                              setSelectedRec(rec);
                              setTargetStatus('OFFERED');
                              setShowStatusModal(true);
                            }}
                            className="px-2.5 py-1.5 text-xs font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200/80 transition"
                            title="Update recommendation workflow status"
                          >
                            Status
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Explanation Modal */}
        {explanation && selectedRec && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="flex justify-between items-start border-b border-slate-200 pb-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-1.5">
                    <Info className="w-4 h-4 text-forest-700" />
                    Clinical Evidence &amp; Explanation Trace
                  </h3>
                  <p className="text-xs text-slate-500 font-mono">
                    Governed Rule ID: {explanation.trigger_rule_id}
                  </p>
                </div>
                <button
                  onClick={() => setExplanation(null)}
                  className="p-1 rounded text-slate-400 hover:text-slate-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <span className="font-semibold text-slate-700 uppercase tracking-wider text-[11px] block">
                    Authoritative Standard:
                  </span>
                  <p className="text-slate-900 font-medium mt-0.5">{explanation.source_guideline}</p>
                </div>

                <div>
                  <span className="font-semibold text-slate-700 uppercase tracking-wider text-[11px] block">
                    Clinical Rationale:
                  </span>
                  <p className="text-slate-800 leading-relaxed mt-0.5">
                    {explanation.clinical_rationale}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="font-semibold text-slate-700 uppercase tracking-wider text-[11px] block">
                      Triggered Care Gap:
                    </span>
                    <p className="font-mono text-slate-800 bg-slate-50 p-2 rounded border border-slate-200 mt-0.5">
                      {explanation.trigger_care_gap || 'None'}
                    </p>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-700 uppercase tracking-wider text-[11px] block">
                      Uncertainty State:
                    </span>
                    <p className="font-mono text-slate-800 bg-slate-50 p-2 rounded border border-slate-200 mt-0.5">
                      {explanation.uncertainty_status}
                    </p>
                  </div>
                </div>

                <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200">
                  <span className="font-semibold text-slate-700 uppercase tracking-wider text-[11px] block">
                    Evidence Summary Payload:
                  </span>
                  <pre className="text-[11px] text-slate-700 mt-1 whitespace-pre-wrap font-mono overflow-x-auto max-h-40">
                    {JSON.stringify(explanation.identified_evidence, null, 2)}
                  </pre>
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => setExplanation(null)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Close Trace
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Review Modal */}
        {showReviewModal && selectedRec && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="border-b border-slate-200 pb-3">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-1.5">
                  <UserCheck className="w-4 h-4 text-forest-700" />
                  Licensed Physician Clinical Review
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Human oversight sign-off for: <span className="font-medium text-slate-800">{selectedRec.recommended_action}</span>
                </p>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    Physician Attestation Decision *
                  </label>
                  <select
                    value={reviewDecision}
                    onChange={(e) => setReviewDecision(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  >
                    <option value="APPROVED">Approve (Certified for Resident Care)</option>
                    <option value="MODIFIED">Modify Action</option>
                    <option value="REJECTED">Reject / Cancel Order</option>
                    <option value="INFO_REQUESTED">Request Additional Diagnostic Telemetry</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    Mandatory Clinical Rationale Notes *
                  </label>
                  <textarea
                    required
                    rows={3}
                    value={reviewRationale}
                    onChange={(e) => setReviewRationale(e.target.value)}
                    placeholder="Document medical justification, patient discussion, or modified orders..."
                    className="w-full rounded-lg border-slate-300 text-xs p-3 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  onClick={() => setShowReviewModal(false)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  onClick={submitReview}
                  disabled={isSubmittingReview}
                  className="btn-healthcare-primary text-xs px-4 py-2 disabled:opacity-50"
                >
                  {isSubmittingReview ? 'Submitting...' : 'Sign & Submit Review'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Status Update Modal */}
        {showStatusModal && selectedRec && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="border-b border-slate-200 pb-3">
                <h3 className="text-base font-bold text-slate-900">Update Recommendation Lifecycle</h3>
                <p className="text-xs text-slate-500 mt-0.5 font-mono text-[11px]">
                  Order: {selectedRec.recommended_action}
                </p>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Target Status *</label>
                  <select
                    value={targetStatus}
                    onChange={(e) => setTargetStatus(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  >
                    <option value="OFFERED">OFFERED (Communicated to resident)</option>
                    <option value="ACCEPTED">ACCEPTED (Resident agreed to pathway)</option>
                    <option value="DECLINED">DECLINED (Resident / guardian declined)</option>
                    <option value="CANCELLED">CANCELLED (Superseded)</option>
                  </select>
                </div>

                {targetStatus === 'DECLINED' && (
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Documented Reason for Declining *
                    </label>
                    <textarea
                      required
                      rows={3}
                      value={declinedReason}
                      onChange={(e) => setDeclinedReason(e.target.value)}
                      placeholder="Enter specific reasons provided by resident or legal guardian..."
                      className="w-full rounded-lg border-slate-300 text-xs p-3 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                    />
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  onClick={() => setShowStatusModal(false)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  onClick={submitStatusUpdate}
                  className="btn-healthcare-primary text-xs px-4 py-2"
                >
                  Update Lifecycle Status
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

export default function RecommendationsPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading recommendations...</span>
        </div>
      }
    >
      <RecommendationsContent />
    </Suspense>
  );
}
