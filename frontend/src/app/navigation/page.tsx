'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { apiClient, ApiError } from '@/lib/api';
import {
  AppointmentResponse,
  ReferralResponse,
  FollowUpTaskResponse,
  NavigationAnalyticsResponse,
  PersonSummary,
  PreventiveRecommendationResponse,
} from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  Calendar,
  Building2,
  Clock,
  ShieldCheck,
  CheckCircle2,
  Plus,
  AlertCircle,
  FileCheck,
  X,
  UserCheck,
  ArrowRight,
  Activity,
} from 'lucide-react';

function NavigationContent() {
  const [activeTab, setActiveTab] = useState<'appointments' | 'referrals' | 'followups' | 'closure'>('appointments');
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [patientRecs, setPatientRecs] = useState<PreventiveRecommendationResponse[]>([]);

  // Analytics
  const [analytics, setAnalytics] = useState<NavigationAnalyticsResponse | null>(null);

  // Lists
  const [appointments, setAppointments] = useState<AppointmentResponse[]>([]);
  const [referrals, setReferrals] = useState<ReferralResponse[]>([]);
  const [followUps, setFollowUps] = useState<FollowUpTaskResponse[]>([]);

  // Booking Modal
  const [showBookModal, setShowBookModal] = useState(false);
  const [serviceType, setServiceType] = useState('WELL_CHILD_VISIT');
  const [apptTime, setApptTime] = useState('');
  const [selectedRecId, setSelectedRecId] = useState('');

  // Referral Modal
  const [showReferralModal, setShowReferralModal] = useState(false);
  const [specialty, setSpecialty] = useState('Preventive Cardiology');
  const [clinicalIndication, setClinicalIndication] = useState('');

  // Care Gap Closure Form
  const [closureGapId, setClosureGapId] = useState('GAP-IFHAS-COMPREHENSIVE');
  const [evidenceSource, setEvidenceSource] = useState('SCREENING_RECORD');
  const [evidenceId, setEvidenceId] = useState('');
  const [closureNotes, setClosureNotes] = useState('');

  // Follow-up resolve modal
  const [resolvingTaskId, setResolvingTaskId] = useState<string | null>(null);
  const [resolutionNotes, setResolutionNotes] = useState('');

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    loadTabData();
    if (selectedPatientId) {
      loadPatientRecs(selectedPatientId);
    }
  }, [activeTab, selectedPatientId]);

  async function loadInitialData() {
    setIsLoading(true);
    try {
      const [patRes, anRes] = await Promise.all([
        apiClient.getPatients(1, 100),
        apiClient.getNavigationAnalytics(),
      ]);
      setPatients(patRes.items);
      setAnalytics(anRes);
      if (patRes.items.length > 0) {
        setSelectedPatientId(patRes.items[0].id);
      }
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to load navigation data');
    } finally {
      setIsLoading(false);
    }
  }

  async function loadTabData() {
    try {
      if (activeTab === 'appointments') {
        const appts = await apiClient.getAppointments(selectedPatientId || undefined);
        setAppointments(appts);
      } else if (activeTab === 'referrals') {
        const refs = await apiClient.getReferrals(selectedPatientId || undefined);
        setReferrals(refs);
      } else if (activeTab === 'followups') {
        const fus = await apiClient.getFollowUps(selectedPatientId || undefined);
        setFollowUps(fus);
      }
    } catch (err: any) {
      console.error('Failed to load tab data', err);
    }
  }

  async function loadPatientRecs(pId: string) {
    try {
      const res = await apiClient.getRecommendations(pId);
      setPatientRecs(res.items);
    } catch (err: any) {
      console.error('Failed to load recommendations', err);
    }
  }

  async function handleBookAppointment() {
    if (!selectedPatientId || !apptTime) {
      alert('Please fill all required appointment fields.');
      return;
    }
    try {
      await apiClient.bookAppointment({
        person_id: selectedPatientId,
        service_type: serviceType,
        scheduled_time: apptTime,
        recommendation_id: selectedRecId || undefined,
        location: 'Al Bateen Healthcare Center - Abu Dhabi',
      });
      setShowBookModal(false);
      setApptTime('');
      loadTabData();
      const anRes = await apiClient.getNavigationAnalytics();
      setAnalytics(anRes);
    } catch (err: any) {
      alert('Booking failed: ' + err.message);
    }
  }

  async function handleUpdateAttendance(id: string, status: string) {
    const notes = prompt('Enter clinical attendance notes:');
    try {
      await apiClient.updateAppointmentStatus(id, status, notes || undefined);
      loadTabData();
      const anRes = await apiClient.getNavigationAnalytics();
      setAnalytics(anRes);
    } catch (err: any) {
      alert('Status update failed: ' + err.message);
    }
  }

  async function handleCreateReferral() {
    if (!selectedPatientId || !clinicalIndication.trim()) {
      alert('Please enter clinical indication.');
      return;
    }
    try {
      await apiClient.createReferral({
        person_id: selectedPatientId,
        specialty,
        clinical_indication: clinicalIndication,
        recommendation_id: selectedRecId || undefined,
      });
      setShowReferralModal(false);
      setClinicalIndication('');
      loadTabData();
      const anRes = await apiClient.getNavigationAnalytics();
      setAnalytics(anRes);
    } catch (err: any) {
      alert('Referral creation failed: ' + err.message);
    }
  }

  async function handleResolveFollowUp(taskId: string) {
    if (!resolutionNotes.trim()) {
      alert('Please provide resolution notes.');
      return;
    }
    try {
      await apiClient.resolveFollowUp(taskId, resolutionNotes);
      setResolvingTaskId(null);
      setResolutionNotes('');
      loadTabData();
      const anRes = await apiClient.getNavigationAnalytics();
      setAnalytics(anRes);
    } catch (err: any) {
      alert('Follow-up resolution failed: ' + err.message);
    }
  }

  async function handleCloseCareGapWithEvidence() {
    if (!selectedPatientId || !evidenceId.trim()) {
      alert('Verifiable evidence ID is required to close clinical care gap.');
      return;
    }
    try {
      await apiClient.closeCareGapWithEvidence({
        person_id: selectedPatientId,
        care_gap_id: closureGapId,
        evidence_source: evidenceSource,
        evidence_id: evidenceId,
        verification_method: 'CLINICIAN_ATTESTATION',
        recommendation_id: selectedRecId || undefined,
        verification_notes: closureNotes || 'Verified by attending clinical provider.',
      });
      alert('Care gap successfully closed with clinical verification evidence.');
      setEvidenceId('');
      setClosureNotes('');
      const anRes = await apiClient.getNavigationAnalytics();
      setAnalytics(anRes);
      loadTabData();
    } catch (err: any) {
      alert('Care gap closure failed: ' + err.message);
    }
  }

  return (
    <AppShell activePage="/navigation">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Care Operations', 'Navigation & Closures']}
          title="Care Navigation &amp; Closed-Loop Follow-Up"
          subtitle="Abu Dhabi Healthcare Delivery: Coordinating appointments, specialist referrals, automated recall follow-ups, and evidence-backed care gap closure."
        />

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Analytics Row */}
        {analytics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Appointments Booked"
              value={analytics.total_appointments}
              icon={Calendar}
              subtitle="Scheduled encounters"
              variant="forest"
            />
            <MetricCard
              title="Attendance Rate"
              value={`${analytics.attendance_rate}%`}
              icon={CheckCircle2}
              subtitle="Show rate compliance"
              variant="emerald"
            />
            <MetricCard
              title="Active Follow-Ups"
              value={analytics.active_follow_up_tasks}
              icon={Clock}
              subtitle="Missed-care recalls"
              variant="amber"
            />
            <MetricCard
              title="Verified Gaps Closed"
              value={analytics.closed_care_gaps_count}
              icon={ShieldCheck}
              subtitle="Evidence-backed Module G"
              variant="teal"
            />
          </div>
        )}

        {/* Resident Target Bar */}
        <div className="card-healthcare p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex-shrink-0">
              Resident Target:
            </span>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none w-full sm:w-96"
            >
              <option value="">-- All Enrolled Residents --</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs)
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            {activeTab === 'appointments' && (
              <button
                onClick={() => setShowBookModal(true)}
                className="btn-healthcare-primary text-xs py-2 px-3.5 inline-flex items-center gap-1.5"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Book Appointment</span>
              </button>
            )}
            {activeTab === 'referrals' && (
              <button
                onClick={() => setShowReferralModal(true)}
                className="btn-healthcare-primary text-xs py-2 px-3.5 inline-flex items-center gap-1.5"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Issue Referral</span>
              </button>
            )}
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-200 space-x-6 text-xs font-semibold overflow-x-auto pb-1">
          {[
            { id: 'appointments', label: `Appointments (${appointments.length})` },
            { id: 'referrals', label: `Specialist Referrals (${referrals.length})` },
            { id: 'followups', label: `Automated Recalls (${followUps.length})` },
            { id: 'closure', label: 'Verified Care-Gap Closure (Module G)' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`pb-3 font-semibold border-b-2 transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'border-forest-800 text-forest-900'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Appointments */}
        {activeTab === 'appointments' && (
          <div className="card-healthcare overflow-hidden">
            {appointments.length === 0 ? (
              <div className="p-8">
                <EmptyState
                  title="No Appointments Scheduled"
                  description="No preventive appointments are currently booked for this resident."
                  action={{
                    label: 'Book Preventive Visit',
                    onClick: () => setShowBookModal(true),
                  }}
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="table-healthcare text-xs">
                  <thead>
                    <tr>
                      <th>Service Type</th>
                      <th>Scheduled Time</th>
                      <th>Clinic Location</th>
                      <th>Status</th>
                      <th className="text-right">Attendance Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {appointments.map((a) => (
                      <tr key={a.id}>
                        <td className="font-semibold text-slate-900">{a.service_type}</td>
                        <td className="font-mono text-slate-600">
                          {new Date(a.scheduled_time).toLocaleString()}
                        </td>
                        <td className="text-slate-600">{a.location}</td>
                        <td>
                          <StatusBadge status={a.status} />
                        </td>
                        <td className="text-right space-x-2">
                          {a.status === 'BOOKED' && (
                            <>
                              <button
                                onClick={() => handleUpdateAttendance(a.id, 'ATTENDED')}
                                className="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-50 text-emerald-800 hover:bg-emerald-100 border border-emerald-200"
                              >
                                Attended
                              </button>
                              <button
                                onClick={() => handleUpdateAttendance(a.id, 'NO_SHOW')}
                                className="px-2.5 py-1 text-xs font-semibold rounded bg-rose-50 text-rose-800 hover:bg-rose-100 border border-rose-200"
                              >
                                No-Show
                              </button>
                            </>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Referrals */}
        {activeTab === 'referrals' && (
          <div className="card-healthcare overflow-hidden">
            {referrals.length === 0 ? (
              <div className="p-8">
                <EmptyState
                  title="No Specialist Referrals"
                  description="No clinical specialist referrals have been issued for this resident."
                  action={{
                    label: 'Issue Specialist Referral',
                    onClick: () => setShowReferralModal(true),
                  }}
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="table-healthcare text-xs">
                  <thead>
                    <tr>
                      <th>Specialty</th>
                      <th>Clinical Indication</th>
                      <th>Priority</th>
                      <th>Status</th>
                      <th>Referral Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {referrals.map((r) => (
                      <tr key={r.id}>
                        <td className="font-semibold text-slate-900">{r.specialty}</td>
                        <td className="text-slate-700 max-w-sm">{r.clinical_indication}</td>
                        <td>
                          <StatusBadge status={r.priority} />
                        </td>
                        <td>
                          <StatusBadge status={r.status} />
                        </td>
                        <td className="font-mono text-slate-500">
                          {new Date(r.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Follow-Up Queue */}
        {activeTab === 'followups' && (
          <div className="card-healthcare overflow-hidden">
            {followUps.length === 0 ? (
              <div className="p-8">
                <EmptyState
                  title="No Follow-Up Recalls"
                  description="All preventive recalls and follow-ups are currently resolved."
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="table-healthcare text-xs">
                  <thead>
                    <tr>
                      <th>Trigger Reason</th>
                      <th>Assigned Role</th>
                      <th>Priority</th>
                      <th>Due Date</th>
                      <th>Status</th>
                      <th className="text-right">Resolution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {followUps.map((f) => (
                      <tr key={f.id}>
                        <td className="font-semibold text-rose-800">{f.trigger_reason}</td>
                        <td className="text-slate-700">{f.assigned_role}</td>
                        <td>
                          <StatusBadge status={f.priority} />
                        </td>
                        <td className="font-mono text-slate-500">
                          {new Date(f.due_date).toLocaleDateString()}
                        </td>
                        <td>
                          <StatusBadge status={f.status} />
                        </td>
                        <td className="text-right">
                          {f.status !== 'RESOLVED' && (
                            <button
                              onClick={() => setResolvingTaskId(f.id)}
                              className="px-2.5 py-1 text-xs font-semibold rounded bg-forest-50 text-forest-800 hover:bg-forest-100 border border-forest-200"
                            >
                              Resolve
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Verified Care-Gap Closure (Module G) */}
        {activeTab === 'closure' && (
          <div className="card-healthcare p-6 space-y-6">
            <div className="border-b border-slate-200 pb-3">
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-forest-700" />
                Module G: Verified Clinical Care-Gap Closure
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Care gaps cannot be closed by superficial button clicks. Closure strictly requires verified clinical evidence
                (Encounter, Screening Record, Lab Result, or Referral Outcome) attested by an authorized clinician.
              </p>
            </div>

            <div className="max-w-xl space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Target Care Gap to Close *
                </label>
                <select
                  value={closureGapId}
                  onChange={(e) => setClosureGapId(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                >
                  <option value="GAP-IFHAS-COMPREHENSIVE">GAP-IFHAS-COMPREHENSIVE (Adult Periodic Exam)</option>
                  <option value="GAP-WELL-CHILD-DUE">GAP-WELL-CHILD-DUE (Pediatric Well-Child Surveillance)</option>
                  <option value="GAP-BP-OVERDUE">GAP-BP-OVERDUE (Blood Pressure Reassessment)</option>
                  <option value="GAP-DM-GLUCOSE-OVERDUE">GAP-DM-GLUCOSE-OVERDUE (Diabetes Fasting Glucose)</option>
                  <option value="GAP-LIPID-SPECIALIST-EVALUATION">GAP-LIPID-SPECIALIST-EVALUATION (Cardiology Referral)</option>
                </select>
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Verifiable Evidence Source *
                </label>
                <select
                  value={evidenceSource}
                  onChange={(e) => setEvidenceSource(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                >
                  <option value="SCREENING_RECORD">Verified Screening Encounter Record</option>
                  <option value="ENCOUNTER">Completed Clinical Outpatient Encounter</option>
                  <option value="LAB_RESULT">Verified Laboratory Result (LOINC match)</option>
                  <option value="REFERRAL_OUTCOME">Specialist Referral Outcome Report</option>
                </select>
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Verifiable Evidence Record ID *
                </label>
                <input
                  type="text"
                  value={evidenceId}
                  onChange={(e) => setEvidenceId(e.target.value)}
                  placeholder="e.g. enc_ifhas_2026_09118 or lab_glucose_8821"
                  className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 font-mono focus:ring-2 focus:ring-forest-600 focus:outline-none"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Clinician Attestation &amp; Verification Notes *
                </label>
                <textarea
                  rows={3}
                  value={closureNotes}
                  onChange={(e) => setClosureNotes(e.target.value)}
                  placeholder="Attest that verified screening results meet clinical criteria and close the care gap..."
                  className="w-full rounded-lg border-slate-300 text-xs p-3 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                />
              </div>

              <button
                onClick={handleCloseCareGapWithEvidence}
                className="btn-healthcare-primary text-xs py-2.5 px-5 flex items-center gap-1.5"
              >
                <FileCheck className="w-4 h-4" />
                <span>Verify Clinical Evidence &amp; Close Care Gap</span>
              </button>
            </div>
          </div>
        )}

        {/* Booking Modal */}
        {showBookModal && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="flex justify-between items-center border-b border-slate-200 pb-3">
                <h3 className="text-base font-bold text-slate-900">Book Preventive Care Appointment</h3>
                <button
                  onClick={() => setShowBookModal(false)}
                  className="p-1 rounded text-slate-400 hover:text-slate-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Service Type *</label>
                  <select
                    value={serviceType}
                    onChange={(e) => setServiceType(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  >
                    <option value="WELL_CHILD_VISIT">Well-Child Visit (Pediatric)</option>
                    <option value="IFHAS_COMPREHENSIVE">IFHAS Comprehensive Screening (Adult)</option>
                    <option value="CARDIOVASCULAR_RISK_CHECK">Cardiovascular Risk Check</option>
                    <option value="DIABETES_GLUCOSE_SCREEN">Diabetes Fasting Glucose Screen</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Date &amp; Time *</label>
                  <input
                    type="datetime-local"
                    value={apptTime}
                    onChange={(e) => setApptTime(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  onClick={() => setShowBookModal(false)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  onClick={handleBookAppointment}
                  className="btn-healthcare-primary text-xs px-4 py-2"
                >
                  Confirm Booking
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Referral Modal */}
        {showReferralModal && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="flex justify-between items-center border-b border-slate-200 pb-3">
                <h3 className="text-base font-bold text-slate-900">Issue Clinical Specialist Referral</h3>
                <button
                  onClick={() => setShowReferralModal(false)}
                  className="p-1 rounded text-slate-400 hover:text-slate-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Target Specialty *</label>
                  <select
                    value={specialty}
                    onChange={(e) => setSpecialty(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  >
                    <option value="Preventive Cardiology">Preventive Cardiology</option>
                    <option value="Endocrinology & Diabetes">Endocrinology &amp; Diabetes</option>
                    <option value="Pediatric Developmental Medicine">Pediatric Developmental Medicine</option>
                    <option value="Genetics & Genomic Medicine">Genetics &amp; Genomic Medicine</option>
                    <option value="Lifestyle & Preventive Medicine">Lifestyle &amp; Preventive Medicine</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    Clinical Indication &amp; Reason for Referral *
                  </label>
                  <textarea
                    rows={3}
                    value={clinicalIndication}
                    onChange={(e) => setClinicalIndication(e.target.value)}
                    placeholder="Document clinical observations, laboratory anomalies, or risk assessment triggers..."
                    className="w-full rounded-lg border-slate-300 text-xs p-3 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  onClick={() => setShowReferralModal(false)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateReferral}
                  className="btn-healthcare-primary text-xs px-4 py-2"
                >
                  Issue Referral Order
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Resolve Task Modal */}
        {resolvingTaskId && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <h3 className="text-base font-bold text-slate-900">Resolve Follow-Up Recall Task</h3>
              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    Resolution Notes &amp; Outcome *
                  </label>
                  <textarea
                    rows={3}
                    value={resolutionNotes}
                    onChange={(e) => setResolutionNotes(e.target.value)}
                    placeholder="Document outcome of follow-up call, rescheduled appointment, or resident confirmation..."
                    className="w-full rounded-lg border-slate-300 text-xs p-3 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  onClick={() => setResolvingTaskId(null)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleResolveFollowUp(resolvingTaskId)}
                  className="btn-healthcare-primary text-xs px-4 py-2"
                >
                  Confirm Resolution
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

export default function NavigationPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading care navigation...</span>
        </div>
      }
    >
      <NavigationContent />
    </Suspense>
  );
}
