'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api';
import { CareGapItem, ComprehensiveEvaluation, PathwaySummary, PersonSummary } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';

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

        const patientList = await apiClient.listPatients(undefined, undefined, 0, 20);
        setPatients(patientList.items);
        if (patientList.items.length > 0) {
          setSelectedPatientId(patientList.items[0].id);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load pathways.');
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

  if (loading) {
    return <div className="p-12 text-center text-slate-400">Loading Abu Dhabi Preventive Pathways...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Abu Dhabi Preventive Care Pathway Catalog
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Governed, deterministic clinical care pathways aligned with the Department of Health Abu Dhabi (DoH) and Abu Dhabi Public Health Center (ADPHC).
        </p>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 text-sm p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Pathway Selector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {pathways.map((pw) => {
          const isSelected = pw.pathway_id === selectedPathwayId;
          return (
            <div
              key={pw.pathway_id}
              onClick={() => handleSelectPathway(pw.pathway_id)}
              className={`p-5 rounded-xl border cursor-pointer transition ${
                isSelected
                  ? 'border-emerald-700 bg-emerald-50/50 shadow-sm ring-1 ring-emerald-700'
                  : 'border-slate-200 bg-white hover:border-slate-300 shadow-sm'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-slate-500 uppercase tracking-wider">
                  v{pw.version}
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                  {pw.target_population}
                </span>
              </div>
              <h2 className="text-base font-bold text-slate-900 mt-2">{pw.pathway_name}</h2>
              <p className="text-xs text-slate-600 mt-1">Authority: <strong>{pw.authoritative_body}</strong></p>
              <p className="text-xs text-slate-500 mt-0.5 font-mono">{pw.guideline_reference}</p>
              <div className="flex items-center justify-between pt-3 mt-3 border-t border-slate-200/60 text-xs text-slate-600">
                <span>Cycle: {pw.cycle_interval_months > 0 ? `${pw.cycle_interval_months} months` : 'Continuous (0-72 mos)'}</span>
                <span className="font-semibold text-emerald-800">{pw.total_milestones} Defined Milestones</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Pathway Details & Milestones */}
      {selectedPathwayDetail && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-lg font-bold text-slate-900">{selectedPathwayDetail.name}</h2>
              <p className="text-xs text-slate-500 font-mono mt-0.5">
                ID: {selectedPathwayDetail.pathway_id} &bull; Clinical Reference: {selectedPathwayDetail.guideline_reference}
              </p>
            </div>
            <div className="mt-2 sm:mt-0 text-right">
              <span className="text-xs px-2.5 py-1 rounded bg-slate-100 font-semibold text-slate-700">
                {selectedPathwayDetail.publishing_authority}
              </span>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Standard Milestones ({selectedPathwayDetail.milestones?.length || 0})
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-xs">
                <thead className="bg-slate-50 font-semibold text-slate-600 text-left">
                  <tr>
                    <th className="px-4 py-2">Milestone ID</th>
                    <th className="px-4 py-2">Name</th>
                    <th className="px-4 py-2">Category</th>
                    <th className="px-4 py-2">Age / Timing</th>
                    <th className="px-4 py-2">Grace Period</th>
                    <th className="px-4 py-2">Clinical Code</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {selectedPathwayDetail.milestones?.map((m: any) => (
                    <tr key={m.milestone_id} className="hover:bg-slate-50">
                      <td className="px-4 py-2 font-mono font-medium text-slate-900">{m.milestone_id}</td>
                      <td className="px-4 py-2 font-medium">{m.name}</td>
                      <td className="px-4 py-2">{m.category}</td>
                      <td className="px-4 py-2">
                        {m.target_age_months !== undefined
                          ? `${m.target_age_months} months`
                          : m.cycle_interval_months !== undefined
                          ? `Every ${m.cycle_interval_months} mos`
                          : 'As indicated'}
                      </td>
                      <td className="px-4 py-2">{m.grace_period_days} days</td>
                      <td className="px-4 py-2 font-mono text-slate-500">{m.screening_code || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Live Pathway Evaluation Sandbox */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="border-b border-slate-200 pb-3">
          <h2 className="text-lg font-bold text-slate-900">
            Live Pathway Eligibility & Care Gap Evaluation Sandbox
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Test deterministic pathway evaluation against actual longitudinal patient records in the database.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-center gap-3">
          <div className="flex-1 w-full">
            <label className="block text-xs font-medium text-slate-700 mb-1">Select Patient to Evaluate</label>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name} ({p.population_classification}, {p.age_years} yrs / {p.age_months} mos) — ID: {p.id.slice(0, 8)}...
                </option>
              ))}
            </select>
          </div>

          <div className="sm:self-end w-full sm:w-auto">
            <button
              onClick={handleEvaluate}
              disabled={evaluating || !selectedPatientId}
              className="w-full sm:w-auto px-5 py-2.5 bg-emerald-800 text-white rounded-lg text-sm font-medium hover:bg-emerald-900 disabled:opacity-50 transition"
            >
              {evaluating ? 'Evaluating...' : 'Run Deterministic Evaluation'}
            </button>
          </div>
        </div>

        {evalError && (
          <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-lg">
            {evalError}
          </div>
        )}

        {/* Evaluation Output */}
        {evaluationResult && (
          <div className="mt-6 border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div>
                <span className="text-xs text-slate-500">Evaluation Result for Patient ID:</span>
                <p className="font-mono text-xs font-bold text-slate-800">{evaluationResult.patient_id}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Eligibility Status:</span>
                <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                  evaluationResult.eligibility.is_eligible
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-rose-100 text-rose-800'
                }`}>
                  {evaluationResult.eligibility.is_eligible ? 'ELIGIBLE' : 'INELIGIBLE'}
                </span>
              </div>
            </div>

            {/* Eligibility Criteria Details */}
            <div>
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                Eligibility Reasons & Criteria:
              </h4>
              <ul className="text-xs space-y-1">
                {evaluationResult.eligibility.reasons.map((r, idx) => (
                  <li key={idx} className="text-slate-700 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Care Gaps Table */}
            {evaluationResult.care_gaps && evaluationResult.care_gaps.length > 0 && (
              <div>
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                  Identified Pathway Care Gaps ({evaluationResult.care_gaps.length})
                </h4>
                <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
                  <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead className="bg-slate-50 text-slate-600 font-semibold text-left">
                      <tr>
                        <th className="px-4 py-2.5">Milestone</th>
                        <th className="px-4 py-2.5">Category</th>
                        <th className="px-4 py-2.5">Status</th>
                        <th className="px-4 py-2.5">Due Date</th>
                        <th className="px-4 py-2.5">Guideline Authority</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {evaluationResult.care_gaps.map((g) => (
                        <tr key={g.milestone_id} className="hover:bg-slate-50">
                          <td className="px-4 py-2.5 font-medium text-slate-900">{g.name}</td>
                          <td className="px-4 py-2.5">{g.category}</td>
                          <td className="px-4 py-2.5">
                            <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                              g.status === 'OVERDUE'
                                ? 'bg-rose-100 text-rose-800'
                                : g.status === 'DUE'
                                ? 'bg-amber-100 text-amber-800'
                                : g.status === 'COMPLETED'
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-slate-100 text-slate-700'
                            }`}>
                              {g.status}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 font-mono">{g.due_date}</td>
                          <td className="px-4 py-2.5 text-slate-500">{g.clinical_guideline}</td>
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
  );
}
