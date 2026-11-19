'use client';

import React, { Suspense, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { PersonSummary, PreventiveAssessmentResponse } from '@/types';
import { getStoredUser } from '@/lib/auth';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { RiskIndicator } from '@/components/RiskIndicator';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  FileCheck2,
  ShieldCheck,
  UserCheck,
  AlertTriangle,
  BookOpen,
  Send,
  Layers,
  Info,
  Clock,
  CheckCircle2,
  XCircle,
  FileText,
} from 'lucide-react';

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
            const aId = patientAssessments[0].assessment_id || patientAssessments[0].id || '';
            if (aId) {
              const firstDetail = await apiClient.getAssessment(aId);
              setSelectedAssessment(firstDetail);
            }
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
        const aId = patientAssessments[0].assessment_id || patientAssessments[0].id || '';
        if (aId) {
          const detail = await apiClient.getAssessment(aId);
          setSelectedAssessment(detail);
        } else {
          setSelectedAssessment(null);
        }
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
    <AppShell activePage="/assessments">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Intelligence', 'Clinical Assessments']}
          title="Explainable Clinical Risk &amp; Human Review"
          subtitle="Transparent multi-factor risk stratification, DoH/ADPHC guideline attribution, data completeness bounds, and licensed physician oversight."
        />

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Patient Selection & Role Banner */}
        <div className="card-healthcare p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex-shrink-0">
              Resident Target:
            </label>
            <select
              value={selectedPatientId}
              onChange={(e) => handlePatientChange(e.target.value)}
              className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none w-full sm:w-96"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs &bull; {p.primary_identifier_masked || p.identifier_value || 'No EID'})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500">Physician Oversight:</span>
            <span
              className={`px-2.5 py-0.5 rounded-full font-semibold border ${
                isPhysician
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-slate-100 text-slate-700 border-slate-200'
              }`}
            >
              {isPhysician ? 'Licensed Clinician (Dr. Reviewer)' : 'Read-Only (Non-Physician)'}
            </span>
          </div>
        </div>

        {/* 2-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Assessment Snapshots */}
          <div className="card-healthcare p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-200 pb-2">
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-forest-700" />
                Assessment Snapshots
              </h3>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                {assessments.length}
              </span>
            </div>

            {assessments.length === 0 ? (
              <p className="text-xs text-slate-400 py-6 text-center">
                No assessments recorded for this patient yet. Run a care pathway evaluation or screening intake to generate a snapshot.
              </p>
            ) : (
              <div className="space-y-2 max-h-[580px] overflow-y-auto pr-1">
                {assessments.map((a, idx) => {
                  const aId = a.assessment_id || a.id || `assessment-${idx}`;
                  const isSelected =
                    selectedAssessment &&
                    (selectedAssessment.assessment_id === aId || selectedAssessment.id === aId);
                  return (
                    <div
                      key={aId}
                      onClick={() => handleSelectAssessment(aId)}
                      className={`p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                        isSelected
                          ? 'border-forest-700 bg-forest-50/40 ring-1 ring-forest-700/30 shadow-sm'
                          : 'border-slate-200 hover:border-slate-300 bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between font-semibold">
                        <span className="text-slate-900 font-mono text-[11px]">
                          {aId.slice(0, 12)}...
                        </span>
                        <RiskIndicator tier={a.clinical_risk_tier} />
                      </div>
                      <div className="text-[11px] text-slate-500 mt-2 flex justify-between">
                        <span className="font-medium text-slate-700">{a.pathway_id}</span>
                        <span>{a.created_at?.slice(0, 10)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Column: Detailed Explainability & Review */}
          <div className="lg:col-span-2 space-y-6">
            {selectedAssessment ? (
              <>
                {/* Snapshot Card */}
                <div className="card-healthcare p-6 space-y-5">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-4 gap-3">
                    <div>
                      <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider block">
                        Snapshot ID: {selectedAssessment.assessment_id || selectedAssessment.id}
                      </span>
                      <h2 className="text-lg font-bold text-slate-900 mt-1">
                        Preventive Risk &amp; Clinical Guidance Profile
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Pathway: <strong className="text-slate-800">{selectedAssessment.pathway_id}</strong> (v{selectedAssessment.pathway_version})
                      </p>
                    </div>
                    <div className="flex flex-col items-start sm:items-end gap-1">
                      <RiskIndicator
                        tier={selectedAssessment.clinical_risk_tier}
                        reviewed={selectedAssessment.human_review_status === 'APPROVED'}
                      />
                      <span className="text-[11px] text-slate-400 font-mono">
                        Lifecycle: {selectedAssessment.assessment_status}
                      </span>
                    </div>
                  </div>

                  {/* Recommended Action */}
                  <div className="p-4 rounded-xl bg-forest-50/50 border border-forest-200/60 text-xs space-y-1">
                    <span className="font-bold text-forest-900 uppercase tracking-wide text-[11px] flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-forest-700" />
                      Recommended Preventive Clinical Action
                    </span>
                    <p className="text-slate-800 leading-relaxed">
                      {selectedAssessment.recommended_action}
                    </p>
                  </div>

                  {/* Contributing Clinical Factors */}
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-forest-700" />
                      Contributing Factors &amp; Guidelines Attribution
                    </h3>
                    {Array.isArray(selectedAssessment.contributing_factors) &&
                    selectedAssessment.contributing_factors.length > 0 ? (
                      <div className="space-y-2.5">
                        {selectedAssessment.contributing_factors.map((cf: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-1.5"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-bold text-slate-900">{cf.factor}</span>
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                                Severity: {cf.severity}
                              </span>
                            </div>
                            <p className="text-slate-600 leading-relaxed">{cf.detail}</p>
                            <div className="text-[11px] text-slate-500 pt-2 border-t border-slate-200/60 flex items-center justify-between">
                              <span>
                                Cited Guideline: <strong className="text-slate-700">{cf.cited_guideline}</strong>
                              </span>
                              {cf.standard_code && (
                                <span className="font-mono text-[10px] text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
                                  {cf.standard_code}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500 italic">
                        No abnormal contributing clinical factors identified.
                      </p>
                    )}
                  </div>

                  {/* Data Limitations */}
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <Info className="w-3.5 h-3.5 text-forest-700" />
                      Epistemic Boundaries &amp; Data Quality Limits
                    </h3>
                    {Array.isArray(selectedAssessment.data_quality_limitations) &&
                    selectedAssessment.data_quality_limitations.length > 0 ? (
                      <ul className="text-xs space-y-1.5 bg-amber-50/70 border border-amber-200 p-3.5 rounded-xl text-amber-900">
                        {selectedAssessment.data_quality_limitations.map((lim: string, idx: number) => (
                          <li key={idx} className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-600 flex-shrink-0" />
                            <span>{lim}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-slate-500 italic">
                        All baseline clinical observations meet freshness and completeness criteria.
                      </p>
                    )}
                  </div>
                </div>

                {/* Licensed Physician Review Card */}
                <div className="card-healthcare p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                      <UserCheck className="w-4 h-4 text-forest-700" />
                      Licensed Physician Review &amp; Attestation
                    </h3>
                    <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                      Attestation: {selectedAssessment.human_review_status}
                    </span>
                  </div>

                  {selectedAssessment.reviewed_at && (
                    <div className="p-3.5 bg-emerald-50/50 border border-emerald-200 rounded-xl text-xs space-y-1.5">
                      <div className="flex justify-between text-slate-600">
                        <span>
                          Reviewed by: <strong className="text-forest-900">{selectedAssessment.reviewed_by_user_id}</strong>
                        </span>
                        <span className="font-mono text-slate-500">{selectedAssessment.reviewed_at}</span>
                      </div>
                      <p className="text-slate-800">
                        <strong>Clinical Rationale:</strong> {selectedAssessment.reviewer_notes || 'No review notes recorded.'}
                      </p>
                    </div>
                  )}

                  {isPhysician ? (
                    <form onSubmit={handleReviewSubmit} className="space-y-4 pt-1">
                      {reviewMsg && (
                        <div
                          className={`p-3 text-xs rounded-xl border ${
                            reviewMsg.type === 'success'
                              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                              : 'bg-rose-50 border-rose-200 text-rose-800'
                          }`}
                        >
                          {reviewMsg.text}
                        </div>
                      )}

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs font-semibold text-slate-700 mb-1">
                            Physician Verdict *
                          </label>
                          <select
                            value={reviewStatus}
                            onChange={(e) => setReviewStatus(e.target.value)}
                            className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                          >
                            <option value="APPROVED">Approve Recommended Pathway</option>
                            <option value="MODIFIED">Modify Preventive Orders</option>
                            <option value="REJECTED">Reject / Defer Action</option>
                            <option value="MORE_INFO_REQUIRED">Request Additional Diagnostics</option>
                          </select>
                        </div>
                      </div>

                      <div>
                        <label className="block text-xs font-semibold text-slate-700 mb-1">
                          Mandatory Clinical Rationale *
                        </label>
                        <textarea
                          required
                          rows={3}
                          value={reviewNotes}
                          onChange={(e) => setReviewNotes(e.target.value)}
                          placeholder="Document clinical reasoning, patient discussion, or altered preventive strategy..."
                          className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                        />
                      </div>

                      <div className="flex justify-end">
                        <button
                          type="submit"
                          disabled={reviewSubmitting}
                          className="btn-healthcare-primary text-xs py-2 px-5 flex items-center gap-1.5 disabled:opacity-50"
                        >
                          {reviewSubmitting ? (
                            'Submitting Attestation...'
                          ) : (
                            <>
                              <Send className="w-3.5 h-3.5" />
                              <span>Submit Physician Attestation</span>
                            </>
                          )}
                        </button>
                      </div>
                    </form>
                  ) : (
                    <p className="text-xs text-slate-500 italic p-3 bg-slate-50 rounded-xl border border-slate-200">
                      Human review submission is restricted to licensed Healthcare Providers (Physicians). Your current credentials are in audit read-only mode.
                    </p>
                  )}
                </div>
              </>
            ) : (
              <div className="p-8 card-healthcare">
                <EmptyState
                  title="No Snapshot Selected"
                  description="Select an assessment snapshot from the left column to view contributing clinical factors and submit reviews."
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

export default function AssessmentsPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading clinical assessments...</span>
        </div>
      }
    >
      <AssessmentsContent />
    </Suspense>
  );
}
