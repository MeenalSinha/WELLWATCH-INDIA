'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import {
  User,
  Heart,
  Activity,
  Calendar,
  Clock,
  Dna,
  ShieldAlert,
  Plus,
  Play,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  FileText,
  BadgeCheck,
  Stethoscope,
  Building2,
  Scale,
} from 'lucide-react';
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
import { AppShell } from '@/components/AppShell';
import { StatusBadge } from '@/components/StatusBadge';
import { RiskIndicator } from '@/components/RiskIndicator';
import { MetricCard } from '@/components/MetricCard';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { EmptyState } from '@/components/EmptyState';

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
      <AppShell pageTitle="Patient Profile" breadcrumbs={[{ label: 'Patients', href: '/patients' }, { label: 'Loading...' }]}>
        <div className="p-16 flex flex-col items-center justify-center text-slate-400">
          <LoadingSpinner size="lg" />
          <span className="text-xs font-semibold text-slate-500 mt-3">Loading longitudinal records and history...</span>
        </div>
      </AppShell>
    );
  }

  if (error || !patient) {
    return (
      <AppShell pageTitle="Patient Profile" breadcrumbs={[{ label: 'Patients', href: '/patients' }, { label: 'Record Not Found' }]}>
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-6 rounded-2xl max-w-xl mx-auto my-8">
          <h2 className="text-base font-bold">Access Blocked or Record Not Found</h2>
          <p className="mt-1 text-xs">{error || 'Patient does not exist or falls outside tenant jurisdiction.'}</p>
          <Link href="/patients" className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold text-forest-700 hover:underline">
            <ArrowLeft className="w-3.5 h-3.5" /> Return to Patient Directory
          </Link>
        </div>
      </AppShell>
    );
  }

  const isPediatric = patient.population_classification === 'PEDIATRIC';

  return (
    <AppShell
      pageTitle={patient.full_name}
      breadcrumbs={[{ label: 'Patients', href: '/patients' }, { label: patient.full_name }]}
    >
      <div className="space-y-6">
        {/* Patient Identity & Primary Metadata Header */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 rounded-2xl bg-forest-800 text-white flex items-center justify-center font-bold text-xl shadow-xs shrink-0">
              {patient.first_name?.charAt(0) || 'P'}
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2.5 flex-wrap">
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">{patient.full_name}</h1>
                <StatusBadge status={patient.population_classification} />
                <StatusBadge status={patient.record_status || 'ACTIVE'} />
              </div>
              <div className="text-xs text-slate-500 flex items-center gap-3 flex-wrap">
                <span>DOB: <strong className="text-slate-700">{patient.date_of_birth}</strong></span>
                <span>&bull;</span>
                <span>Age: <strong className="text-slate-700">{patient.age_years} yrs ({patient.age_months} mos)</strong></span>
                <span>&bull;</span>
                <span>Gender: <strong className="text-slate-700">{patient.gender}</strong></span>
                <span>&bull;</span>
                <span>Emirate: <strong className="text-slate-700">{patient.emirate || 'Abu Dhabi'}</strong></span>
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                Patient UUID: {patient.id}
              </div>
            </div>
          </div>

          {/* Clinical Action Buttons */}
          <div className="flex items-center gap-2.5 flex-wrap shrink-0">
            <button
              onClick={() => setShowObsModal(true)}
              className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200 shadow-2xs transition flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5 text-forest-700" />
              Record Vitals
            </button>
            <button
              onClick={handleGenerateAssessment}
              disabled={generatingAssessment}
              className="px-4 py-2 rounded-lg bg-forest-800 hover:bg-forest-900 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 text-emerald-400" />
              {generatingAssessment ? 'Evaluating Risk...' : 'Run Preventive Assessment'}
            </button>
          </div>
        </div>

        {/* Quick Context Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Life-Course Category"
            value={isPediatric ? 'Pediatric' : 'Adult'}
            subtitle={isPediatric ? 'Child Health & ASQ-3 Standards' : 'IFHAS Periodic Screening Standard'}
            icon={User}
            variant="forest"
          />
          <MetricCard
            label="Clinical Observations"
            value={timeline?.total_events || 0}
            unit="records"
            subtitle="Longitudinal provenance verified"
            icon={Activity}
          />
          <MetricCard
            label="Genomic Findings"
            value={genomics.length}
            unit="variants"
            subtitle="Emirati Genome / ACMG Scope"
            icon={Dna}
            variant="blue"
          />
          <MetricCard
            label="Risk Assessments"
            value={assessments.length}
            unit="evaluations"
            subtitle="Explainable clinical decisions"
            icon={ShieldAlert}
            variant="amber"
          />
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
          <div className="border-b border-slate-200/80 px-6 pt-3 flex gap-6 overflow-x-auto text-xs font-medium">
            <button
              onClick={() => setActiveTab('profile')}
              className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition ${activeTab === 'profile'
                ? 'border-forest-700 text-forest-800'
                : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
            >
              <Stethoscope className="w-4 h-4" />
              <span>Clinical Profile</span>
            </button>
            <button
              onClick={() => setActiveTab('timeline')}
              className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition ${activeTab === 'timeline'
                ? 'border-forest-700 text-forest-800'
                : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
            >
              <Clock className="w-4 h-4" />
              <span>Longitudinal Timeline ({timeline?.total_events || 0})</span>
            </button>
            <button
              onClick={() => setActiveTab('genomics')}
              className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition ${activeTab === 'genomics'
                ? 'border-forest-700 text-forest-800'
                : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
            >
              <Dna className="w-4 h-4" />
              <span>Genomic Intelligence ({genomics.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('assessments')}
              className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition ${activeTab === 'assessments'
                ? 'border-forest-700 text-forest-800'
                : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
            >
              <ShieldAlert className="w-4 h-4" />
              <span>Risk Assessments ({assessments.length})</span>
            </button>
          </div>

          <div className="p-6">
            {/* Tab 1: Clinical Profile */}
            {activeTab === 'profile' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {isPediatric ? (
                  <>
                    {/* Pediatric Growth & Well-Child Status */}
                    <div className="bg-[#F8FAF9] p-5 rounded-xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-200/80 pb-2.5">
                        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Pediatric Growth & Well-Child Status
                        </h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
                          DoH Child Standards
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div>
                          <span className="text-slate-500">Age in Months:</span>
                          <p className="font-bold text-slate-900 text-sm mt-0.5">
                            {pediatricProfile?.age_months ?? patient.age_months} months
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">Completed Well-Child Visits:</span>
                          <p className="font-bold text-slate-900 text-sm mt-0.5">
                            {pediatricProfile?.well_child_visit_count ?? 0}
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">Milestones Due:</span>
                          <p className="font-bold text-amber-600 text-sm mt-0.5">
                            {pediatricProfile?.milestones_due_count ?? 0}
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">Last Well-Child Visit:</span>
                          <p className="font-medium text-slate-800 mt-0.5">
                            {pediatricProfile?.last_well_child_visit || 'No prior visits logged'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Guardianship & Consent Authority */}
                    <div className="bg-[#F8FAF9] p-5 rounded-xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-200/80 pb-2.5">
                        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Guardianship & Consent Authority
                        </h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                          Wadeema's Law
                        </span>
                      </div>
                      {pediatricProfile?.guardian_info ? (
                        <div className="text-xs space-y-2.5">
                          <div>
                            <span className="text-slate-500">Legal Guardian User ID:</span>
                            <p className="font-mono text-slate-800 text-[11px] mt-0.5">
                              {pediatricProfile.guardian_info.guardian_user_id}
                            </p>
                          </div>
                          <div className="flex items-center justify-between">
                            <div>
                              <span className="text-slate-500">Relationship:</span>
                              <p className="font-bold text-slate-900 mt-0.5">{pediatricProfile.guardian_info.relationship}</p>
                            </div>
                            <div>
                              <span className="text-slate-500">Verification:</span>
                              <div className="mt-0.5">
                                <StatusBadge status={pediatricProfile.guardian_info.status} />
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <p className="text-xs text-amber-700">No verified legal guardian on record.</p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    {/* Adult Vitals & IFHAS Metrics */}
                    <div className="bg-[#F8FAF9] p-5 rounded-xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-200/80 pb-2.5">
                        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Cardiovascular & Metabolic Vitals
                        </h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-forest-50 text-forest-800 border border-forest-200">
                          IFHAS Screening
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div>
                          <span className="text-slate-500">Blood Pressure:</span>
                          <p className="font-bold text-slate-900 text-sm mt-0.5">
                            {adultProfile?.blood_pressure?.systolic && adultProfile?.blood_pressure?.diastolic
                              ? `${adultProfile.blood_pressure.systolic}/${adultProfile.blood_pressure.diastolic} mmHg`
                              : 'Not measured'}
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">Fasting Glucose:</span>
                          <p className="font-bold text-slate-900 text-sm mt-0.5">
                            {adultProfile?.fasting_glucose_mg_dl ? `${adultProfile.fasting_glucose_mg_dl} mg/dL` : 'Not recorded'}
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">Body Mass Index (BMI):</span>
                          <p className="font-bold text-slate-900 text-sm mt-0.5">
                            {adultProfile?.bmi ? `${adultProfile.bmi} kg/m²` : 'Not calculated'}
                          </p>
                        </div>
                        <div>
                          <span className="text-slate-500">IFHAS Adherence:</span>
                          <p className="font-medium text-slate-800 mt-0.5">
                            {adultProfile?.screening_adherence?.status || 'Active Screening'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Active Conditions & Family History */}
                    <div className="bg-[#F8FAF9] p-5 rounded-xl border border-slate-200/90 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-200/80 pb-2.5">
                        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                          Conditions & Family Risk
                        </h3>
                      </div>
                      <div className="space-y-3 text-xs">
                        <div>
                          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Active Diagnoses:</span>
                          {adultProfile?.active_conditions && adultProfile.active_conditions.length > 0 ? (
                            <ul className="mt-1 space-y-1">
                              {adultProfile.active_conditions.map((c, idx) => (
                                <li key={idx} className="p-2 rounded bg-white border border-slate-200 font-medium text-slate-800">
                                  {c.description} ({c.code}) &bull; Status: {c.status}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-slate-400 mt-0.5">No active chronic conditions documented.</p>
                          )}
                        </div>
                        <div>
                          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Family Medical History:</span>
                          {adultProfile?.family_history && adultProfile.family_history.length > 0 ? (
                            <ul className="mt-1 space-y-1">
                              {adultProfile.family_history.map((f, idx) => (
                                <li key={idx} className="p-2 rounded bg-white border border-slate-200 text-slate-700">
                                  {f.condition} ({f.relationship})
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-slate-400 mt-0.5">No family risk history documented.</p>
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
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                    Longitudinal Care Continuity & Event Provenance
                  </h3>
                  <span className="text-xs text-slate-500">{timeline?.total_events || 0} chronological milestones</span>
                </div>

                {timeline && timeline.events.length > 0 ? (
                  <div className="relative border-l-2 border-forest-700/20 ml-3 pl-6 space-y-6 pt-2">
                    {timeline.events.map((ev) => (
                      <div key={ev.event_id} className="relative group">
                        <div className="absolute -left-[31px] top-1.5 w-3.5 h-3.5 rounded-full bg-forest-700 border-2 border-white shadow-xs" />
                        <div className="bg-[#F8FAF9] p-4 rounded-xl border border-slate-200/80 space-y-1.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-bold text-slate-900">{ev.title}</span>
                            <span className="text-[11px] font-mono text-slate-400">{ev.timestamp}</span>
                          </div>
                          <p className="text-xs text-slate-600 leading-relaxed">{ev.description}</p>
                          <div className="flex items-center gap-2 pt-1 text-[10px]">
                            <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
                              Source: {ev.source_name}
                            </span>
                            <StatusBadge status={ev.epistemic_status} size="sm" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No Longitudinal Events Recorded"
                    description="No prior encounters or verified observations logged for this patient."
                    actionText="Record First Vitals"
                    onAction={() => setShowObsModal(true)}
                  />
                )}
              </div>
            )}

            {/* Tab 3: Genomic Intelligence */}
            {activeTab === 'genomics' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Authorized Genomic Intelligence
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Emirati Genome Program, ACMG classification, and authorized pharmacogenomics.
                    </p>
                  </div>
                  <span className="text-xs text-slate-500">{genomics.length} findings</span>
                </div>

                {genomics.length > 0 ? (
                  <div className="space-y-3">
                    {genomics.map((g) => (
                      <div
                        key={g.id}
                        className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9] space-y-2.5"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-slate-900">{g.gene_symbol}</span>
                            <span className="text-xs font-mono text-slate-500">({g.variant_id})</span>
                          </div>
                          <StatusBadge status={g.variant_classification} />
                        </div>
                        <p className="text-xs text-slate-700 leading-relaxed">{g.clinical_interpretation}</p>
                        {g.actionable_guidance && (
                          <div className="p-2.5 bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 rounded-lg">
                            <strong>Actionable Clinical Guidance:</strong> {g.actionable_guidance}
                          </div>
                        )}
                        <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-200/60">
                          <span>Source Lab: {g.source_laboratory} &bull; Ref: {g.report_reference || 'EGP-2026'}</span>
                          <span className="font-semibold text-slate-600">Review: {g.review_status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No Genomic Findings Documented"
                    description="No authorized genetic variants or pharmacogenomic markers registered for this patient profile."
                  />
                )}
              </div>
            )}

            {/* Tab 4: Preventive Risk Assessments */}
            {activeTab === 'assessments' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Preventive Risk Assessments & Clinical Reviews
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Evidence-backed pathway evaluations under DoH standards.
                    </p>
                  </div>
                  <button
                    onClick={handleGenerateAssessment}
                    disabled={generatingAssessment}
                    className="px-3 py-1.5 rounded-lg bg-forest-800 hover:bg-forest-900 text-white text-xs font-semibold shadow-xs transition disabled:opacity-50"
                  >
                    {generatingAssessment ? 'Evaluating...' : '+ Run New Assessment'}
                  </button>
                </div>

                {assessments.length > 0 ? (
                  <div className="space-y-3">
                    {assessments.map((a, idx) => {
                      const aId = a.assessment_id || a.id || `assess-${idx}`;
                      return (
                        <div
                          key={aId}
                          className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9] space-y-2 hover:border-slate-300 transition"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-bold text-slate-900">{a.pathway_id} (v{a.pathway_version})</span>
                              <span className="text-[10px] font-mono text-slate-400">{new Date(a.created_at).toLocaleDateString()}</span>
                            </div>
                            <RiskIndicator tier={a.clinical_risk_tier} requiresReview={a.human_review_status === 'AWAITING_REVIEW'} />
                          </div>
                          <p className="text-xs text-slate-700">
                            <strong>Recommended Action:</strong> {a.recommended_action}
                          </p>
                          <div className="text-[11px] text-slate-500 flex items-center justify-between pt-2 border-t border-slate-200/60">
                            <span>Human Review: <strong className="text-slate-800">{a.human_review_status}</strong></span>
                            <Link
                              href={`/assessments?id=${aId}`}
                              className="text-forest-700 font-semibold hover:underline"
                            >
                              View Full Explainability & Review &rarr;
                            </Link>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <EmptyState
                    title="No Risk Assessments Generated"
                    description="Run an explainable preventive assessment to evaluate cardiometabolic or developmental care gaps."
                    actionText="Run Preventive Assessment"
                    onAction={handleGenerateAssessment}
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {/* Observation Recording Modal */}
        {showObsModal && (
          <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-xl border border-slate-200 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  Record Clinical Vitals
                </h2>
                <button onClick={() => setShowObsModal(false)} className="text-slate-400 hover:text-slate-600 text-xs">
                  ✕
                </button>
              </div>

              {obsError && (
                <div className="p-2.5 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-lg">
                  {obsError}
                </div>
              )}

              <form onSubmit={handleRecordObservation} className="space-y-3.5">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Observation Type</label>
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
                    className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg bg-white focus:outline-none focus:ring-1 focus:ring-forest-600"
                  >
                    <option value="8480-6">Systolic Blood Pressure (mmHg)</option>
                    <option value="8462-4">Diastolic Blood Pressure (mmHg)</option>
                    <option value="1558-6">Fasting Glucose (mg/dL)</option>
                    <option value="8302-2">Body Height (cm)</option>
                    <option value="29463-7">Body Weight (kg)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Value ({newObs.unit})
                  </label>
                  <input
                    type="number"
                    step="any"
                    required
                    value={newObs.numeric_value || ''}
                    onChange={(e) => setNewObs({ ...newObs, numeric_value: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-100">
                  <button
                    type="button"
                    onClick={() => setShowObsModal(false)}
                    className="px-3.5 py-2 border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 hover:bg-slate-50 transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={obsSubmitting}
                    className="px-4 py-2 bg-forest-800 hover:bg-forest-900 text-white rounded-lg text-xs font-semibold shadow-xs transition disabled:opacity-50"
                  >
                    {obsSubmitting ? 'Saving...' : 'Record Observation'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
