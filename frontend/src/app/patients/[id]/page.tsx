'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api';
import {
  AdultProfile,
  GenomicFindingResponse,
  ObservationCreate,
  PediatricProfile,
  PersonSummary,
  PreventiveAssessmentResponse,
  TimelineResponse,
} from '@/types';
import { StatusBadge } from '@/components/StatusBadge';

export default function PatientDetailPage() {
  const params = useParams();
  const router = useRouter();
  const patientId = params?.id as string;

  const [patient, setPatient] = useState<PersonSummary | null>(null);
  const [pediatricProfile, setPediatricProfile] = useState<PediatricProfile | null>(null);
  const [adultProfile, setAdultProfile] = useState<AdultProfile | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [genomics, setGenomics] = useState<GenomicFindingResponse[]>([]);
  const [assessments, setAssessments] = useState<PreventiveAssessmentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Active Tab
  const [activeTab, setActiveTab] = useState<'profile' | 'timeline' | 'genomics' | 'assessments'>('profile');

  // Observation Modal
  const [showObsModal, setShowObsModal] = useState(false);
  const [obsSubmitting, setObsSubmitting] = useState(false);
  const [obsError, setObsError] = useState<string | null>(null);
  const [newObs, setNewObs] = useState<ObservationCreate>({
    observation_type: 'VITAL_SIGN',
    code: '8480-6',
    display_name: 'Systolic Blood Pressure',
    numeric_value: 120,
    unit: 'mmHg',
  });

  // Assessment Trigger
  const [generatingAssessment, setGeneratingAssessment] = useState(false);

  const loadPatientData = async () => {
    if (!patientId) return;
    setLoading(true);
    setError(null);
    try {
      const summary = await apiClient.getPatient(patientId);
      setPatient(summary);

      if (summary.population_classification === 'PEDIATRIC') {
        const ped = await apiClient.getPediatricProfile(patientId).catch(() => null);
        setPediatricProfile(ped);
      } else {
        const adult = await apiClient.getAdultProfile(patientId).catch(() => null);
        setAdultProfile(adult);
      }

      const timeRes = await apiClient.getTimeline(patientId).catch(() => null);
      setTimeline(timeRes);

      const genRes = await apiClient.getPatientGenomics(patientId).catch(() => []);
      setGenomics(genRes);

      const assessRes = await apiClient.listPatientAssessments(patientId).catch(() => []);
      setAssessments(assessRes);
    } catch (err: any) {
      setError(err.message || 'Failed to load patient profile.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPatientData();
  }, [patientId]);

  const handleRecordObservation = async (e: React.FormEvent) => {
    e.preventDefault();
    setObsSubmitting(true);
    setObsError(null);
    try {
      await apiClient.recordObservation(patientId, newObs);
      setShowObsModal(false);
      await loadPatientData();
    } catch (err: any) {
      setObsError(err.message || 'Failed to record observation.');
    } finally {
      setObsSubmitting(false);
    }
  };

  const handleGenerateAssessment = async () => {
    setGeneratingAssessment(true);
    try {
      await apiClient.generateAssessment(patientId);
      await loadPatientData();
      setActiveTab('assessments');
    } catch (err: any) {
      alert(`Assessment Generation Failed: ${err.message}`);
    } finally {
      setGeneratingAssessment(false);
    }
  };

  if (loading) {
    return (
      <div className="p-12 text-center text-slate-400">
        Loading patient record and clinical history...
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="bg-rose-50 border border-rose-200 text-rose-800 p-6 rounded-xl">
        <h2 className="text-lg font-bold">Access Blocked or Record Not Found</h2>
        <p className="mt-1 text-sm">{error || 'Patient does not exist.'}</p>
        <Link href="/patients" className="mt-4 inline-block text-xs font-semibold underline">
          &larr; Return to Patient Directory
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Link href="/patients" className="hover:underline">Patients</Link>
            <span>/</span>
            <span className="font-mono text-slate-700">{patient.id}</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-3">
            {patient.full_name}
            <StatusBadge status={patient.population_classification} />
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            DOB: {patient.date_of_birth} &bull; Age: {patient.age_years} yrs ({patient.age_months} mos) &bull; Gender: {patient.gender} &bull; Emirate: {patient.emirate}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowObsModal(true)}
            className="px-3 py-2 border border-slate-300 rounded-lg text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 shadow-sm"
          >
            + Record Vitals
          </button>
          <button
            onClick={handleGenerateAssessment}
            disabled={generatingAssessment}
            className="px-4 py-2 bg-emerald-800 hover:bg-emerald-900 text-white rounded-lg text-xs font-medium shadow-sm transition disabled:opacity-50"
          >
            {generatingAssessment ? 'Evaluating Risk...' : 'Run Preventive Assessment'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 flex space-x-6">
        <button
          onClick={() => setActiveTab('profile')}
          className={`pb-3 text-sm font-medium border-b-2 transition ${
            activeTab === 'profile'
              ? 'border-emerald-700 text-emerald-800 font-semibold'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Clinical Profile
        </button>
        <button
          onClick={() => setActiveTab('timeline')}
          className={`pb-3 text-sm font-medium border-b-2 transition ${
            activeTab === 'timeline'
              ? 'border-emerald-700 text-emerald-800 font-semibold'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Longitudinal Timeline ({timeline?.total_events || 0})
        </button>
        <button
          onClick={() => setActiveTab('genomics')}
          className={`pb-3 text-sm font-medium border-b-2 transition ${
            activeTab === 'genomics'
              ? 'border-emerald-700 text-emerald-800 font-semibold'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Genomic Intelligence ({genomics.length})
        </button>
        <button
          onClick={() => setActiveTab('assessments')}
          className={`pb-3 text-sm font-medium border-b-2 transition ${
            activeTab === 'assessments'
              ? 'border-emerald-700 text-emerald-800 font-semibold'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Risk Assessments ({assessments.length})
        </button>
      </div>

      {/* Tab 1: Clinical Profile */}
      {activeTab === 'profile' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {patient.population_classification === 'PEDIATRIC' ? (
            <>
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
                  Pediatric Growth & Well-Child Status
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-slate-500">Age in Months:</span>
                    <p className="font-semibold text-slate-800 text-sm">{pediatricProfile?.age_months ?? patient.age_months} mos</p>
                  </div>
                  <div>
                    <span className="text-slate-500">Completed Well-Child Visits:</span>
                    <p className="font-semibold text-slate-800 text-sm">{pediatricProfile?.well_child_visit_count ?? 0}</p>
                  </div>
                  <div>
                    <span className="text-slate-500">Due Milestones:</span>
                    <p className="font-semibold text-amber-600 text-sm">{pediatricProfile?.milestones_due_count ?? 0}</p>
                  </div>
                  <div>
                    <span className="text-slate-500">Last Well-Child Visit:</span>
                    <p className="font-semibold text-slate-800">{pediatricProfile?.last_well_child_visit || 'No prior visits'}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
                  Guardianship & Consent Authority
                </h3>
                {pediatricProfile?.guardian_info ? (
                  <div className="text-xs space-y-2">
                    <div>
                      <span className="text-slate-500">Guardian User ID:</span>
                      <p className="font-mono text-slate-800">{pediatricProfile.guardian_info.guardian_user_id}</p>
                    </div>
                    <div>
                      <span className="text-slate-500">Relationship:</span>
                      <p className="font-semibold text-slate-800">{pediatricProfile.guardian_info.relationship}</p>
                    </div>
                    <div>
                      <span className="text-slate-500">Verification Status:</span>
                      <p className="font-semibold text-emerald-700">{pediatricProfile.guardian_info.status}</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-amber-700">No verified legal guardian on record.</p>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
                  Cardiovascular & Metabolic Vitals
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-slate-500">Blood Pressure:</span>
                    <p className="font-semibold text-slate-800 text-sm">
                      {adultProfile?.blood_pressure?.systolic && adultProfile?.blood_pressure?.diastolic
                        ? `${adultProfile.blood_pressure.systolic}/${adultProfile.blood_pressure.diastolic} mmHg`
                        : 'Not measured'}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-500">Fasting Glucose:</span>
                    <p className="font-semibold text-slate-800 text-sm">
                      {adultProfile?.fasting_glucose_mg_dl ? `${adultProfile.fasting_glucose_mg_dl} mg/dL` : 'Not recorded'}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-500">Body Mass Index (BMI):</span>
                    <p className="font-semibold text-slate-800 text-sm">
                      {adultProfile?.bmi ? `${adultProfile.bmi} kg/m²` : 'Not calculated'}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-500">IFHAS Screening Adherence:</span>
                    <p className="font-semibold text-slate-800 text-sm">
                      {adultProfile?.screening_adherence?.status || 'Unknown'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
                  Documented Conditions & Family History
                </h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Active Diagnoses:</span>
                    {adultProfile?.active_conditions && adultProfile.active_conditions.length > 0 ? (
                      <ul className="mt-1 space-y-1">
                        {adultProfile.active_conditions.map((c, idx) => (
                          <li key={idx} className="text-xs font-medium text-slate-800 bg-slate-50 p-2 rounded">
                            {c.description} ({c.code}) &bull; Status: {c.status}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-slate-400 mt-1">No active chronic conditions documented.</p>
                    )}
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Family Medical History:</span>
                    {adultProfile?.family_history && adultProfile.family_history.length > 0 ? (
                      <ul className="mt-1 space-y-1">
                        {adultProfile.family_history.map((f, idx) => (
                          <li key={idx} className="text-xs text-slate-700 bg-slate-50 p-2 rounded">
                            {f.condition} ({f.relationship})
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-slate-400 mt-1">No family risk history documented.</p>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Tab 2: Longitudinal Timeline */}
      {activeTab === 'timeline' && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 mb-4">
            Unified Longitudinal Health Records
          </h3>
          {timeline && timeline.events.length > 0 ? (
            <div className="relative border-l-2 border-slate-200 ml-4 space-y-6">
              {timeline.events.map((ev) => (
                <div key={ev.event_id} className="relative pl-6">
                  <div className="absolute -left-2 top-1.5 w-4 h-4 rounded-full bg-emerald-600 border-2 border-white shadow-sm" />
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-900">{ev.title}</span>
                    <span className="text-xs font-mono text-slate-400">{ev.timestamp}</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">{ev.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                      Source: {ev.source_name}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-700">
                      Epistemic: {ev.epistemic_status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No longitudinal clinical events recorded yet.</p>
          )}
        </div>
      )}

      {/* Tab 3: Genomic Intelligence */}
      {activeTab === 'genomics' && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-slate-900">
              Authorized Genomic Intelligence (Emirati Genome Program / ACMG)
            </h3>
          </div>
          {genomics.length > 0 ? (
            <div className="divide-y divide-slate-200">
              {genomics.map((g) => (
                <div key={g.id} className="py-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-slate-900">{g.gene_symbol}</span>
                      <span className="text-xs font-mono text-slate-500">({g.variant_id})</span>
                    </div>
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                      g.variant_classification === 'PATHOGENIC'
                        ? 'bg-rose-100 text-rose-800'
                        : g.variant_classification === 'VUS'
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-emerald-100 text-emerald-800'
                    }`}>
                      {g.variant_classification}
                    </span>
                  </div>
                  <p className="text-xs text-slate-700">{g.clinical_interpretation}</p>
                  {g.actionable_guidance && (
                    <div className="p-2 bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 rounded">
                      <strong>Actionable Clinical Guidance:</strong> {g.actionable_guidance}
                    </div>
                  )}
                  <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1">
                    <span>Lab: {g.source_laboratory} &bull; Ref: {g.report_reference || 'N/A'}</span>
                    <span className="font-semibold text-slate-600">Review Status: {g.review_status}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No genomic variant records on file for this individual.</p>
          )}
        </div>
      )}

      {/* Tab 4: Preventive Assessments */}
      {activeTab === 'assessments' && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900">
            Historical Preventive Risk Assessments & Clinical Reviews
          </h3>
          {assessments.length > 0 ? (
            <div className="space-y-4">
              {assessments.map((a, idx) => {
                const aId = a.assessment_id || a.id || `assess-${idx}`;
                return (
                  <div key={aId} className="p-4 border border-slate-200 rounded-lg hover:border-slate-300 transition space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-800">Pathway: {a.pathway_id} ({a.pathway_version})</span>
                        <span className="text-[10px] font-mono text-slate-400">{a.created_at}</span>
                      </div>
                      <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
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
                    <p className="text-xs text-slate-700"><strong>Action:</strong> {a.recommended_action}</p>
                    <div className="text-[11px] text-slate-500 flex items-center justify-between pt-1 border-t border-slate-100">
                      <span>Human Review: <strong className="text-slate-800">{a.human_review_status}</strong></span>
                      <Link
                        href={`/assessments?id=${aId}`}
                        className="text-emerald-700 font-semibold hover:underline"
                      >
                        View Full Explainability & Review &rarr;
                      </Link>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No preventive risk assessments generated yet. Click "Run Preventive Assessment" above.</p>
          )}
        </div>
      )}

      {/* Observation Modal */}
      {showObsModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900 bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-6 shadow-xl border border-slate-200">
            <h2 className="text-base font-bold text-slate-900 mb-3">Record Clinical Observation</h2>
            {obsError && (
              <div className="mb-3 p-2 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded">
                {obsError}
              </div>
            )}
            <form onSubmit={handleRecordObservation} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-700">Observation Type</label>
                <select
                  value={newObs.code}
                  onChange={(e) => {
                    const code = e.target.value;
                    if (code === '8480-6') {
                      setNewObs({ ...newObs, code, display_name: 'Systolic Blood Pressure', unit: 'mmHg', numeric_value: 120 });
                    } else if (code === '8462-4') {
                      setNewObs({ ...newObs, code, display_name: 'Diastolic Blood Pressure', unit: 'mmHg', numeric_value: 80 });
                    } else if (code === '1558-6') {
                      setNewObs({ ...newObs, code, display_name: 'Fasting Blood Glucose', unit: 'mg/dL', numeric_value: 95 });
                    } else if (code === '8302-2') {
                      setNewObs({ ...newObs, code, display_name: 'Body Height', unit: 'cm', numeric_value: 170 });
                    } else if (code === '29463-7') {
                      setNewObs({ ...newObs, code, display_name: 'Body Weight', unit: 'kg', numeric_value: 70 });
                    }
                  }}
                  className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg bg-white"
                >
                  <option value="8480-6">Systolic Blood Pressure (mmHg)</option>
                  <option value="8462-4">Diastolic Blood Pressure (mmHg)</option>
                  <option value="1558-6">Fasting Glucose (mg/dL)</option>
                  <option value="8302-2">Body Height (cm)</option>
                  <option value="29463-7">Body Weight (kg)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">Numeric Value ({newObs.unit})</label>
                <input
                  type="number"
                  step="any"
                  required
                  value={newObs.numeric_value || ''}
                  onChange={(e) => setNewObs({ ...newObs, numeric_value: parseFloat(e.target.value) })}
                  className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg"
                />
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  type="button"
                  onClick={() => setShowObsModal(false)}
                  className="px-4 py-2 border border-slate-300 rounded-lg text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={obsSubmitting}
                  className="px-4 py-2 bg-emerald-800 text-white rounded-lg text-xs font-medium hover:bg-emerald-900 disabled:opacity-50"
                >
                  {obsSubmitting ? 'Saving...' : 'Record Observation'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
