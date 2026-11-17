'use client';

import React, { Suspense, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { PersonSummary, PreventiveAssessmentResponse } from '@/types';
import { getStoredUser } from '@/lib/auth';

function AssessmentsContent() {
  const searchParams = useSearchParams();
  const queryAssessmentId = searchParams.get('id');

  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [assessments, setAssessments] = useState<PreventiveAssessmentResponse[]>([]);
  const [selectedAssessment, setSelectedAssessment] = useState<PreventiveAssessmentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Review Form
  const [reviewStatus, setReviewStatus] = useState<string>('APPROVED');
  const [reviewNotes, setReviewNotes] = useState<string>('');
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [reviewMsg, setReviewMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const currentUser = getStoredUser();
  const isPhysician = currentUser?.role === 'HEALTHCARE_PROVIDER';

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      setError(null);
      try {
        if (queryAssessmentId) {
          const detail = await apiClient.getAssessment(queryAssessmentId);
          setSelectedAssessment(detail);
        }

        const patientList = await apiClient.listPatients(undefined, undefined, 0, 50);
        setPatients(patientList.items);
        if (patientList.items.length > 0 && !queryAssessmentId) {
          setSelectedPatientId(patientList.items[0].id);
          const patientAssessments = await apiClient.listPatientAssessments(patientList.items[0].id);
          setAssessments(patientAssessments);
          if (patientAssessments.length > 0) {
            const firstDetail = await apiClient.getAssessment(patientAssessments[0].assessment_id || patientAssessments[0].id!);
            setSelectedAssessment(firstDetail);
          }
        }
      } catch (err: any) {
        setError(err.message || 'Failed to initialize assessments.');
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [queryAssessmentId]);

  const handlePatientChange = async (patientId: string) => {
    setSelectedPatientId(patientId);
    setLoading(true);
    setError(null);
    try {
      const patientAssessments = await apiClient.listPatientAssessments(patientId);
      setAssessments(patientAssessments);
      if (patientAssessments.length > 0) {
        const detail = await apiClient.getAssessment(patientAssessments[0].assessment_id || patientAssessments[0].id!);
        setSelectedAssessment(detail);
      } else {
        setSelectedAssessment(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch assessments for patient.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectAssessment = async (assessmentId: string) => {
    try {
      const detail = await apiClient.getAssessment(assessmentId);
      setSelectedAssessment(detail);
      setReviewMsg(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load assessment detail.');
    }
  };

  const handleReviewSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedAssessment) return;
    const assessmentId = selectedAssessment.assessment_id || selectedAssessment.id;
    if (!assessmentId) return;

    setReviewSubmitting(true);
    setReviewMsg(null);
    try {
      await apiClient.submitAssessmentReview(assessmentId, reviewStatus, reviewNotes);
      setReviewMsg({ type: 'success', text: `Review submitted: Status updated to ${reviewStatus}.` });
      // Reload detail
      const refreshed = await apiClient.getAssessment(assessmentId);
      setSelectedAssessment(refreshed);
      setReviewNotes('');
    } catch (err: any) {
      setReviewMsg({ type: 'error', text: err.message || 'Failed to submit clinical review.' });
    } finally {
      setReviewSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Explainable Preventive Risk Intelligence & Human Review
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Transparent multi-factor risk stratification, DoH/ADPHC guideline attribution, data completeness bounds, and licensed physician oversight.
        </p>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 text-sm p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Patient Selector */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Patient:</span>
          <select
            value={selectedPatientId}
            onChange={(e) => handlePatientChange(e.target.value)}
            className="px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.full_name} ({p.population_classification}) — {p.id.slice(0, 8)}...
              </option>
            ))}
          </select>
        </div>

        <div className="text-xs text-slate-500">
          Physician Oversight Mode:{' '}
          <strong className={isPhysician ? 'text-emerald-700' : 'text-slate-600'}>
            {isPhysician ? 'Active (Dr. Reviewer)' : 'Read-Only (Non-Physician Role)'}
          </strong>
        </div>
      </div>

      {/* Main Grid: Assessments list & Explainability Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Assessments List */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <h2 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
            Assessment Snapshots ({assessments.length})
          </h2>
          {assessments.length === 0 ? (
            <p className="text-xs text-slate-500 py-4">No assessments generated for this patient yet.</p>
          ) : (
            <div className="space-y-2">
              {assessments.map((a) => {
                const aId = a.assessment_id || a.id!;
                const isSelected = selectedAssessment && (selectedAssessment.assessment_id === aId || selectedAssessment.id === aId);
                return (
                  <div
                    key={aId}
                    onClick={() => handleSelectAssessment(aId)}
                    className={`p-3 rounded-lg border text-xs cursor-pointer transition ${
                      isSelected
                        ? 'border-emerald-700 bg-emerald-50/50 ring-1 ring-emerald-700'
                        : 'border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between font-medium">
                      <span className="text-slate-900 font-mono text-[11px]">{aId.slice(0, 12)}...</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        a.clinical_risk_tier === 'ELEVATED_RISK'
                          ? 'bg-rose-100 text-rose-800'
                          : a.clinical_risk_tier === 'MODERATE_RISK'
                          ? 'bg-amber-100 text-amber-800'
                          : a.clinical_risk_tier === 'INSUFFICIENT_DATA'
                          ? 'bg-purple-100 text-purple-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {a.clinical_risk_tier}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-1 flex justify-between">
                      <span>{a.pathway_id}</span>
                      <span>{a.created_at?.slice(0, 10)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Selected Assessment Deep Explainability & Review */}
        <div className="lg:col-span-2 space-y-6">
          {selectedAssessment ? (
            <>
              {/* Snapshot Overview Card */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-4 gap-2">
                  <div>
                    <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">
                      Immutable Snapshot ID: {selectedAssessment.assessment_id || selectedAssessment.id}
                    </span>
                    <h2 className="text-xl font-bold text-slate-900 mt-0.5">
                      Preventive Risk Evaluation
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Evaluated under Pathway: <strong>{selectedAssessment.pathway_id}</strong> (v{selectedAssessment.pathway_version})
                    </p>
                  </div>
                  <div className="text-right">
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                      selectedAssessment.clinical_risk_tier === 'ELEVATED_RISK'
                        ? 'bg-rose-100 text-rose-800 ring-1 ring-rose-300'
                        : selectedAssessment.clinical_risk_tier === 'MODERATE_RISK'
                        ? 'bg-amber-100 text-amber-800 ring-1 ring-amber-300'
                        : selectedAssessment.clinical_risk_tier === 'INSUFFICIENT_DATA'
                        ? 'bg-purple-100 text-purple-800 ring-1 ring-purple-300'
                        : 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-300'
                    }`}>
                      Tier: {selectedAssessment.clinical_risk_tier}
                    </span>
                    <div className="text-[11px] text-slate-400 mt-1 font-mono">
                      Status: {selectedAssessment.assessment_status}
                    </div>
                  </div>
                </div>

                <div className="text-xs text-slate-700 bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="font-bold text-slate-900">Recommended Clinical Action:</span>{' '}
                  {selectedAssessment.recommended_action}
                </div>

                {/* Deep Contributing Factors & Clinical Citations */}
                <div>
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    Contributing Clinical Factors & Guidelines Attributions
                  </h3>
                  {Array.isArray(selectedAssessment.contributing_factors) && selectedAssessment.contributing_factors.length > 0 ? (
                    <div className="space-y-2">
                      {selectedAssessment.contributing_factors.map((cf: any, idx: number) => (
                        <div key={idx} className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-900">{cf.factor}</span>
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-50 text-amber-800">
                              Severity: {cf.severity}
                            </span>
                          </div>
                          <p className="text-slate-600">{cf.detail}</p>
                          <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-100 flex items-center justify-between">
                            <span>Authoritative Guideline: <strong className="text-slate-700">{cf.cited_guideline}</strong></span>
                            {cf.standard_code && <span className="font-mono text-[10px] text-slate-500">Code: {cf.standard_code}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">No abnormal contributing clinical factors identified.</p>
                  )}
                </div>

                {/* Data Limitations & Epistemic Boundaries */}
                <div>
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    Data Quality, Staleness & Abstention Analysis
                  </h3>
                  {Array.isArray(selectedAssessment.data_quality_limitations) && selectedAssessment.data_quality_limitations.length > 0 ? (
                    <ul className="text-xs space-y-1 bg-amber-50/60 border border-amber-200 p-3 rounded-lg text-amber-900">
                      {selectedAssessment.data_quality_limitations.map((lim: string, idx: number) => (
                        <li key={idx} className="flex items-center gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                          <span>{lim}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-500">All required baseline clinical observations are recent and validated.</p>
                  )}
                </div>
              </div>

              {/* Physician Review Workflow Card */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <h3 className="text-sm font-bold text-slate-900">
                    Licensed Clinician Review & Attestation
                  </h3>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded bg-slate-100 text-slate-700">
                    Current Review Status: {selectedAssessment.human_review_status}
                  </span>
                </div>

                {selectedAssessment.reviewed_at && (
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1">
                    <div className="flex justify-between text-slate-500">
                      <span>Reviewed by: <strong className="text-slate-700">{selectedAssessment.reviewed_by_user_id}</strong></span>
                      <span className="font-mono">{selectedAssessment.reviewed_at}</span>
                    </div>
                    <p className="text-slate-800"><strong>Notes:</strong> {selectedAssessment.reviewer_notes || 'No review notes recorded.'}</p>
                  </div>
                )}

                {isPhysician ? (
                  <form onSubmit={handleReviewSubmit} className="space-y-4 pt-2">
                    {reviewMsg && (
                      <div className={`p-3 text-xs rounded-lg border ${
                        reviewMsg.type === 'success' ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-rose-50 border-rose-200 text-rose-800'
                      }`}>
                        {reviewMsg.text}
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-slate-700 mb-1">Physician Decision</label>
                        <select
                          value={reviewStatus}
                          onChange={(e) => setReviewStatus(e.target.value)}
                          className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                        >
                          <option value="APPROVED">Approve Recommended Pathway</option>
                          <option value="MODIFIED">Modify Preventive Orders</option>
                          <option value="REJECTED">Reject / Defer Action</option>
                          <option value="MORE_INFO_REQUIRED">Request Additional Diagnostics</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-700 mb-1">Mandatory Clinical Rationale *</label>
                      <textarea
                        required
                        rows={3}
                        value={reviewNotes}
                        onChange={(e) => setReviewNotes(e.target.value)}
                        placeholder="Document medical justification, patient discussion, or altered clinical management plan..."
                        className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg"
                      />
                    </div>

                    <div className="flex justify-end">
                      <button
                        type="submit"
                        disabled={reviewSubmitting}
                        className="px-4 py-2 bg-emerald-800 text-white rounded-lg text-xs font-semibold hover:bg-emerald-900 disabled:opacity-50 transition"
                      >
                        {reviewSubmitting ? 'Submitting Attestation...' : 'Submit Physician Attestation'}
                      </button>
                    </div>
                  </form>
                ) : (
                  <p className="text-xs text-slate-500 italic">
                    Human review submission is restricted to licensed Healthcare Providers (Physicians). Your current account does not have clinical sign-off authorization.
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white p-12 text-center text-slate-500 rounded-xl border border-slate-200">
              Select an assessment snapshot on the left to inspect explainable factors and submit clinical reviews.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AssessmentsPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-slate-400">Loading assessments...</div>}>
      <AssessmentsContent />
    </Suspense>
  );
}
