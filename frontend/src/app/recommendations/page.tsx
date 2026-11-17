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
        50,
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
      alert('Please select a patient first.');
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const newRecs = await apiClient.generateRecommendations(patientId);
      alert(`Generated ${newRecs.length} evidence-backed recommendations.`);
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Personalized Preventive Recommendations</h1>
          <p className="text-sm text-slate-600 mt-1">
            Abu Dhabi Public Health Center (ADPHC) & DoH Evidence-Backed Clinical Guidance Engine
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="rounded-lg border-slate-300 text-sm py-2 px-3 focus:ring-emerald-500 focus:border-emerald-500"
          >
            <option value="">-- All Patients --</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.full_name} ({p.population_classification})
              </option>
            ))}
          </select>
          {patientId && (
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 shadow-sm disabled:opacity-50"
            >
              {isGenerating ? 'Evaluating...' : '⚡ Generate Recommendations'}
            </button>
          )}
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="flex flex-wrap items-center gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200 text-sm">
        <span className="font-medium text-slate-700">Filters:</span>
        <select
          value={filterPriority}
          onChange={(e) => setFilterPriority(e.target.value)}
          className="rounded-md border-slate-300 text-xs py-1.5 px-2.5"
        >
          <option value="">All Priorities</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="ROUTINE">Routine</option>
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="rounded-md border-slate-300 text-xs py-1.5 px-2.5"
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
        {(filterPriority || filterStatus || patientId) && (
          <button
            onClick={() => {
              setFilterPriority('');
              setFilterStatus('');
              setPatientId('');
            }}
            className="text-xs text-slate-500 hover:text-slate-800 underline"
          >
            Clear Filters
          </button>
        )}
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Main Table */}
      {isLoading ? (
        <div className="text-center py-12 text-slate-500">Loading recommendations from database...</div>
      ) : recommendations.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-xl border border-slate-200 text-slate-500">
          No preventive recommendations found matching criteria. Select a patient and click "Generate Recommendations".
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-slate-600 font-medium">
              <tr>
                <th className="px-6 py-3 text-left">Category</th>
                <th className="px-6 py-3 text-left">Recommended Action</th>
                <th className="px-6 py-3 text-left">Priority</th>
                <th className="px-6 py-3 text-left">Status</th>
                <th className="px-6 py-3 text-left">Target Window</th>
                <th className="px-6 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {recommendations.map((rec) => (
                <tr key={rec.id} className="hover:bg-slate-50/60 transition-colors">
                  <td className="px-6 py-4 font-medium text-slate-800">
                    {rec.category.replace(/_/g, ' ')}
                    <span className="block text-xs font-mono text-slate-400 mt-0.5">{rec.rule_id}</span>
                  </td>
                  <td className="px-6 py-4 text-slate-700 max-w-md">
                    <p className="font-medium text-slate-900">{rec.recommended_action}</p>
                    <p className="text-xs text-slate-500 line-clamp-1 mt-0.5">{rec.clinical_rationale}</p>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex px-2 py-0.5 text-xs font-semibold rounded-full ${
                        rec.priority === 'CRITICAL'
                          ? 'bg-rose-100 text-rose-800'
                          : rec.priority === 'HIGH'
                          ? 'bg-amber-100 text-amber-800'
                          : rec.priority === 'MEDIUM'
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-slate-100 text-slate-800'
                      }`}
                    >
                      {rec.priority}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex px-2 py-0.5 text-xs font-medium rounded-full ${
                        rec.lifecycle_status === 'COMPLETED'
                          ? 'bg-emerald-100 text-emerald-800'
                          : rec.lifecycle_status === 'DECLINED'
                          ? 'bg-rose-100 text-rose-800'
                          : rec.lifecycle_status === 'SCHEDULED'
                          ? 'bg-indigo-100 text-indigo-800'
                          : rec.lifecycle_status === 'APPROVED'
                          ? 'bg-teal-100 text-teal-800'
                          : 'bg-amber-50 text-amber-800 border border-amber-200'
                      }`}
                    >
                      {rec.lifecycle_status.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-xs text-slate-500">
                    {rec.target_date ? new Date(rec.target_date).toLocaleDateString() : 'Immediate'}
                  </td>
                  <td className="px-6 py-4 text-right space-x-2">
                    <button
                      onClick={() => handleViewExplanation(rec)}
                      className="text-xs font-medium text-emerald-600 hover:text-emerald-800"
                    >
                      Explain
                    </button>
                    {rec.lifecycle_status === 'PENDING_CLINICAL_REVIEW' && (
                      <button
                        onClick={() => {
                          setSelectedRec(rec);
                          setShowReviewModal(true);
                        }}
                        className="text-xs font-semibold text-blue-600 hover:text-blue-800"
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
                      className="text-xs font-medium text-slate-600 hover:text-slate-800"
                    >
                      Status
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Explanation Modal */}
      {explanation && selectedRec && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-xl">
            <div className="flex justify-between items-start border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-lg font-bold text-slate-900">Recommendation Explanation</h3>
                <p className="text-xs text-slate-500">Governed Rule ID: {explanation.trigger_rule_id}</p>
              </div>
              <button
                onClick={() => setExplanation(null)}
                className="text-slate-400 hover:text-slate-600 text-lg"
              >
                ✕
              </button>
            </div>
            <div className="space-y-3 text-sm">
              <div>
                <span className="font-semibold text-slate-700">Source Standard:</span>
                <p className="text-slate-900">{explanation.source_guideline}</p>
              </div>
              <div>
                <span className="font-semibold text-slate-700">Clinical Rationale:</span>
                <p className="text-slate-900">{explanation.clinical_rationale}</p>
              </div>
              <div>
                <span className="font-semibold text-slate-700">Identified Care Gap:</span>
                <p className="text-slate-900 font-mono text-xs">{explanation.trigger_care_gap || 'None'}</p>
              </div>
              <div>
                <span className="font-semibold text-slate-700">Clinical Uncertainty State:</span>
                <p className="text-slate-900 font-mono text-xs">{explanation.uncertainty_status}</p>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                <span className="font-semibold text-slate-700 text-xs">Evidence Summary:</span>
                <pre className="text-xs text-slate-700 mt-1 whitespace-pre-wrap">
                  {JSON.stringify(explanation.identified_evidence, null, 2)}
                </pre>
              </div>
            </div>
            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setExplanation(null)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Review Modal */}
      {showReviewModal && selectedRec && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Licensed Physician Clinical Review</h3>
            <p className="text-xs text-slate-500">
              Human oversight attestation for recommendation: {selectedRec.recommended_action}
            </p>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Decision</label>
                <select
                  value={reviewDecision}
                  onChange={(e) => setReviewDecision(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                >
                  <option value="APPROVED">Approve (Certified for Patient Care)</option>
                  <option value="MODIFIED">Modify Action</option>
                  <option value="REJECTED">Reject / Cancel</option>
                  <option value="INFO_REQUESTED">Request Additional Clinical Data</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Rationale Notes</label>
                <textarea
                  value={reviewRationale}
                  onChange={(e) => setReviewRationale(e.target.value)}
                  placeholder="Enter medical justification and doctor notes..."
                  className="w-full rounded-lg border-slate-300 text-sm h-24"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowReviewModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={submitReview}
                disabled={isSubmittingReview}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {isSubmittingReview ? 'Submitting...' : 'Sign & Submit Review'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status Update / Decline Modal */}
      {showStatusModal && selectedRec && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Update Recommendation Lifecycle</h3>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Target Status</label>
                <select
                  value={targetStatus}
                  onChange={(e) => setTargetStatus(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                >
                  <option value="OFFERED">OFFERED (Communicated to patient)</option>
                  <option value="ACCEPTED">ACCEPTED (Patient agreed)</option>
                  <option value="DECLINED">DECLINED (Patient declined)</option>
                  <option value="CANCELLED">CANCELLED</option>
                </select>
              </div>
              {targetStatus === 'DECLINED' && (
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Documented Reason for Declining <span className="text-rose-500">*</span>
                  </label>
                  <textarea
                    value={declinedReason}
                    onChange={(e) => setDeclinedReason(e.target.value)}
                    placeholder="Enter reason provided by patient/guardian..."
                    className="w-full rounded-lg border-slate-300 text-sm h-20"
                  />
                </div>
              )}
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowStatusModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={submitStatusUpdate}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium"
              >
                Update Status
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function RecommendationsPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Recommendations...</div>}>
      <RecommendationsContent />
    </Suspense>
  );
}
