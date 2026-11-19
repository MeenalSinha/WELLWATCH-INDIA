'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/lib/api';
import { CareGapItem, ComprehensiveEvaluation, PathwaySummary, PersonSummary } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import {
  GitBranch,
  ShieldCheck,
  CheckCircle2,
  Calendar,
  Play,
  FileCode,
  AlertCircle,
  Clock,
  Layers,
  ChevronRight,
  BookOpen,
} from 'lucide-react';

export default function PathwaysPage() {
  const [pathways, setPathways] = useState<PathwaySummary[]>([]);
  const [selectedPathwayId, setSelectedPathwayId] = useState<string>('');
  const [selectedPathwayDetail, setSelectedPathwayDetail] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Evaluation Sandbox
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<ComprehensiveEvaluation | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const list = await apiClient.listPathways();
        setPathways(list);
        if (list.length > 0) {
          setSelectedPathwayId(list[0].pathway_id);
          const detail = await apiClient.getPathway(list[0].pathway_id);
          setSelectedPathwayDetail(detail);
        }

        const patientList = await apiClient.listPatients(undefined, undefined, 0, 50);
        setPatients(patientList.items);
        if (patientList.items.length > 0) {
          setSelectedPatientId(patientList.items[0].id);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load preventive pathways catalog.');
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const handleSelectPathway = async (pathwayId: string) => {
    setSelectedPathwayId(pathwayId);
    setEvaluationResult(null);
    try {
      const detail = await apiClient.getPathway(pathwayId);
      setSelectedPathwayDetail(detail);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch pathway details.');
    }
  };

  const handleEvaluate = async () => {
    if (!selectedPathwayId || !selectedPatientId) return;
    setEvaluating(true);
    setEvalError(null);
    try {
      const result = await apiClient.evaluateAll(selectedPathwayId, selectedPatientId);
      setEvaluationResult(result);
    } catch (err: any) {
      setEvalError(err.message || 'Pathway evaluation failed.');
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <AppShell activePage="/pathways">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Clinical Protocols', 'Care Pathways']}
          title="Preventive Care Pathway Engine"
          subtitle="Deterministic clinical screening and surveillance pathways aligned with Department of Health Abu Dhabi (DoH) and ADPHC guidelines."
        />

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Pathway Catalog Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {pathways.map((pw) => {
            const isSelected = pw.pathway_id === selectedPathwayId;
            return (
              <div
                key={pw.pathway_id}
                onClick={() => handleSelectPathway(pw.pathway_id)}
                className={`card-healthcare p-5 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-forest-700 bg-forest-50/30 ring-2 ring-forest-700/20 shadow-md'
                    : 'hover:border-slate-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-bold text-forest-800 bg-forest-100/60 px-2.5 py-0.5 rounded border border-forest-200">
                    VERSION {pw.version}
                  </span>
                  <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                    {pw.target_population}
                  </span>
                </div>
                <h3 className="text-base font-bold text-slate-900 mt-2.5 flex items-center justify-between">
                  <span>{pw.pathway_name}</span>
                  {isSelected && <CheckCircle2 className="w-4 h-4 text-forest-700" />}
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  Authority: <strong className="text-slate-800">{pw.authoritative_body}</strong>
                </p>
                <p className="text-[11px] text-slate-500 font-mono mt-0.5">{pw.guideline_reference}</p>
                <div className="flex items-center justify-between pt-3 mt-3 border-t border-slate-200/80 text-xs text-slate-600">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    {pw.cycle_interval_months > 0
                      ? `Cadence: ${pw.cycle_interval_months} months`
                      : 'Cadence: Continuous (0–72 mos)'}
                  </span>
                  <span className="font-semibold text-forest-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    {pw.total_milestones} Defined Milestones
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Pathway Details & Milestones */}
        {selectedPathwayDetail && (
          <div className="card-healthcare p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-3 gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-forest-700" />
                  <h2 className="text-base font-bold text-slate-900">{selectedPathwayDetail.name}</h2>
                </div>
                <p className="text-xs text-slate-500 font-mono mt-0.5">
                  Identifier: {selectedPathwayDetail.pathway_id} &bull; Reference: {selectedPathwayDetail.guideline_reference}
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded bg-slate-100 font-semibold text-slate-700 border border-slate-200">
                {selectedPathwayDetail.publishing_authority}
              </span>
            </div>

            <div>
              <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5 text-forest-700" />
                Standard Clinical Milestones ({selectedPathwayDetail.milestones?.length || 0})
              </h3>
              <div className="overflow-x-auto">
                <table className="table-healthcare text-xs">
                  <thead>
                    <tr>
                      <th>Milestone ID</th>
                      <th>Milestone Name</th>
                      <th>Category</th>
                      <th>Age / Timing</th>
                      <th>Grace Period</th>
                      <th>Clinical Code</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedPathwayDetail.milestones?.map((m: any) => (
                      <tr key={m.milestone_id}>
                        <td className="font-mono font-medium text-slate-900">{m.milestone_id}</td>
                        <td className="font-semibold text-slate-800">{m.name}</td>
                        <td>
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">
                            {m.category}
                          </span>
                        </td>
                        <td className="text-slate-600">
                          {m.target_age_months !== undefined
                            ? `${m.target_age_months} months`
                            : m.cycle_interval_months !== undefined
                            ? `Every ${m.cycle_interval_months} mos`
                            : 'As indicated'}
                        </td>
                        <td className="text-slate-600">{m.grace_period_days} days</td>
                        <td className="font-mono text-slate-500">{m.screening_code || 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Live Pathway Evaluation Sandbox */}
        <div className="card-healthcare p-6 space-y-5">
          <div className="border-b border-slate-200 pb-3">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Play className="w-4 h-4 text-forest-700" />
              Live Deterministic Eligibility &amp; Care Gap Evaluation Sandbox
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Evaluates live patient observations against the formal rules engine in real time to calculate gap status.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-3">
            <div className="flex-1 w-full">
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Select Resident Record to Evaluate:
              </label>
              <select
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs &bull; {p.primary_identifier_masked || p.identifier_value || 'No EID'})
                  </option>
                ))}
              </select>
            </div>

            <div className="sm:self-end w-full sm:w-auto">
              <button
                onClick={handleEvaluate}
                disabled={evaluating || !selectedPatientId}
                className="w-full sm:w-auto btn-healthcare-primary text-xs py-2.5 px-5 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {evaluating ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Evaluating Engine...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    <span>Run Deterministic Evaluation</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {evalError && (
            <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
              <span>{evalError}</span>
            </div>
          )}

          {/* Evaluation Output */}
          {evaluationResult && (
            <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                <div>
                  <span className="text-[11px] text-slate-500 block">Evaluated Resident Target:</span>
                  <span className="font-mono text-xs font-bold text-slate-900">
                    {evaluationResult.patient_id}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Eligibility Verdict:</span>
                  <span
                    className={`px-2.5 py-1 rounded-full text-xs font-bold border ${
                      evaluationResult.eligibility.is_eligible
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : 'bg-rose-50 text-rose-800 border-rose-200'
                    }`}
                  >
                    {evaluationResult.eligibility.is_eligible ? 'ELIGIBLE FOR PROTOCOL' : 'INELIGIBLE'}
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Eligibility Criteria &amp; Decision Rationales:
                </h4>
                <ul className="text-xs space-y-1">
                  {Array.isArray(evaluationResult.eligibility?.reasons) ? (
                    evaluationResult.eligibility.reasons.map((r: string, idx: number) => (
                      <li key={idx} className="text-slate-700 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-forest-700 flex-shrink-0" />
                        <span>{r}</span>
                      </li>
                    ))
                  ) : (
                    <li className="text-slate-700 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-forest-700 flex-shrink-0" />
                      <span>{(evaluationResult.eligibility as any)?.explanation || 'Protocol evaluated against clinical criteria.'}</span>
                    </li>
                  )}
                </ul>
              </div>

              {/* Care Gaps Table */}
              {evaluationResult.care_gaps && evaluationResult.care_gaps.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    Identified Pathway Care Gaps ({evaluationResult.care_gaps.length})
                  </h4>
                  <div className="overflow-x-auto bg-white rounded-lg border border-slate-200">
                    <table className="table-healthcare text-xs">
                      <thead>
                        <tr>
                          <th>Milestone</th>
                          <th>Category</th>
                          <th>Gap Status</th>
                          <th>Due Date</th>
                          <th>Guideline Authority</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evaluationResult.care_gaps.map((g: any) => (
                          <tr key={g.milestone_id}>
                            <td className="font-semibold text-slate-900">
                              <div>{g.milestone_name || g.name}</div>
                              <span className="font-mono text-[10px] text-slate-400">{g.milestone_id}</span>
                            </td>
                            <td>{g.category || 'Screening Protocol'}</td>
                            <td>
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                                  g.status === 'OVERDUE' || g.status === 'POTENTIALLY_OVERDUE'
                                    ? 'bg-rose-50 text-rose-800 border-rose-200'
                                    : g.status === 'DUE'
                                    ? 'bg-amber-50 text-amber-800 border-amber-200'
                                    : g.status === 'COMPLETED'
                                    ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                    : 'bg-slate-100 text-slate-700 border-slate-200'
                                }`}
                              >
                                {g.status.replace(/_/g, ' ')}
                              </span>
                            </td>
                            <td className="font-mono text-slate-600 text-xs">
                              {g.due_date || (g.status === 'DUE' ? 'Target Window (18m)' : 'Scheduled')}
                            </td>
                            <td className="text-slate-600 text-xs max-w-xs">{g.reason || g.clinical_guideline}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
