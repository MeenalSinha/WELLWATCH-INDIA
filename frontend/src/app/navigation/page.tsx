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

  // Care Gap Closure Modal (Module G)
  const [showClosureModal, setShowClosureModal] = useState(false);
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
      setShowClosureModal(false);
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Care Navigation & Closed-Loop Follow-Up</h1>
          <p className="text-sm text-slate-600 mt-1">
            Abu Dhabi Healthcare Delivery: Coordinating appointments, referrals, follow-ups, and evidence-backed care gap closure
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedPatientId}
            onChange={(e) => setSelectedPatientId(e.target.value)}
            className="rounded-lg border-slate-300 text-sm py-2 px-3"
          >
            <option value="">-- All Patients --</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.full_name} ({p.population_classification})
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Analytics Cards */}
      {analytics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Total Appointments</span>
            <p className="text-2xl font-bold text-slate-900 mt-1">{analytics.total_appointments}</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Attendance Rate</span>
            <p className="text-2xl font-bold text-emerald-600 mt-1">{analytics.attendance_rate}%</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Active Follow-up Tasks</span>
            <p className="text-2xl font-bold text-amber-600 mt-1">{analytics.active_follow_up_tasks}</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Verified Care Gaps Closed</span>
            <p className="text-2xl font-bold text-blue-600 mt-1">{analytics.closed_care_gaps_count}</p>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex border-b border-slate-200 space-x-6 text-sm">
        <button
          onClick={() => setActiveTab('appointments')}
          className={`pb-3 font-semibold border-b-2 transition-colors ${
            activeTab === 'appointments' ? 'border-emerald-600 text-emerald-600' : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          📅 Appointments ({appointments.length})
        </button>
        <button
          onClick={() => setActiveTab('referrals')}
          className={`pb-3 font-semibold border-b-2 transition-colors ${
            activeTab === 'referrals' ? 'border-emerald-600 text-emerald-600' : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🏥 Specialist Referrals ({referrals.length})
        </button>
        <button
          onClick={() => setActiveTab('followups')}
          className={`pb-3 font-semibold border-b-2 transition-colors ${
            activeTab === 'followups' ? 'border-emerald-600 text-emerald-600' : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🔄 Automated Follow-Up Tasks ({followUps.length})
        </button>
        <button
          onClick={() => setActiveTab('closure')}
          className={`pb-3 font-semibold border-b-2 transition-colors ${
            activeTab === 'closure' ? 'border-emerald-600 text-emerald-600' : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🛡️ Verified Care-Gap Closure (Module G)
        </button>
      </div>

      {/* Tab 1: Appointments */}
      {activeTab === 'appointments' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-base font-bold text-slate-900">Coordinated Preventive Appointments</h3>
            <button
              onClick={() => setShowBookModal(true)}
              className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-medium"
            >
              + Book Preventive Appointment
            </button>
          </div>
          {appointments.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-sm">No appointments scheduled.</div>
          ) : (
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50 text-slate-600 font-medium">
                <tr>
                  <th className="px-4 py-2.5 text-left">Service Type</th>
                  <th className="px-4 py-2.5 text-left">Scheduled Time</th>
                  <th className="px-4 py-2.5 text-left">Location</th>
                  <th className="px-4 py-2.5 text-left">Status</th>
                  <th className="px-4 py-2.5 text-right">Attendance Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {appointments.map((a) => (
                  <tr key={a.id} className="hover:bg-slate-50/60">
                    <td className="px-4 py-3 font-semibold text-slate-800">{a.service_type}</td>
                    <td className="px-4 py-3 text-slate-600">{new Date(a.scheduled_time).toLocaleString()}</td>
                    <td className="px-4 py-3 text-slate-500">{a.location}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex px-2 py-0.5 rounded-full font-medium ${
                          a.status === 'ATTENDED' || a.status === 'COMPLETED'
                            ? 'bg-emerald-100 text-emerald-800'
                            : a.status === 'NO_SHOW'
                            ? 'bg-rose-100 text-rose-800'
                            : 'bg-indigo-100 text-indigo-800'
                        }`}
                      >
                        {a.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right space-x-2">
                      {a.status === 'BOOKED' && (
                        <>
                          <button
                            onClick={() => handleUpdateAttendance(a.id, 'ATTENDED')}
                            className="text-xs font-semibold text-emerald-600 hover:text-emerald-800"
                          >
                            Mark Attended
                          </button>
                          <button
                            onClick={() => handleUpdateAttendance(a.id, 'NO_SHOW')}
                            className="text-xs font-semibold text-rose-600 hover:text-rose-800"
                          >
                            Mark No-Show
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Tab 2: Referrals */}
      {activeTab === 'referrals' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-base font-bold text-slate-900">Clinical Specialist Referrals</h3>
            <button
              onClick={() => setShowReferralModal(true)}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-medium"
            >
              + Issue Clinical Referral
            </button>
          </div>
          {referrals.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-sm">No specialist referrals on record.</div>
          ) : (
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50 text-slate-600 font-medium">
                <tr>
                  <th className="px-4 py-2.5 text-left">Specialty</th>
                  <th className="px-4 py-2.5 text-left">Clinical Indication</th>
                  <th className="px-4 py-2.5 text-left">Priority</th>
                  <th className="px-4 py-2.5 text-left">Status</th>
                  <th className="px-4 py-2.5 text-left">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {referrals.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-50/60">
                    <td className="px-4 py-3 font-semibold text-slate-800">{r.specialty}</td>
                    <td className="px-4 py-3 text-slate-600 max-w-sm">{r.clinical_indication}</td>
                    <td className="px-4 py-3 font-mono">{r.priority}</td>
                    <td className="px-4 py-3">
                      <span className="inline-flex px-2 py-0.5 rounded-full font-medium bg-blue-100 text-blue-800">
                        {r.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-500">{new Date(r.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Tab 3: Follow-Up Tasks */}
      {activeTab === 'followups' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900">Automated Missed-Care Follow-Up Queue</h3>
          {followUps.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-sm">No pending follow-up tasks.</div>
          ) : (
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50 text-slate-600 font-medium">
                <tr>
                  <th className="px-4 py-2.5 text-left">Trigger Reason</th>
                  <th className="px-4 py-2.5 text-left">Assigned Role</th>
                  <th className="px-4 py-2.5 text-left">Priority</th>
                  <th className="px-4 py-2.5 text-left">Due Date</th>
                  <th className="px-4 py-2.5 text-left">Status</th>
                  <th className="px-4 py-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {followUps.map((f) => (
                  <tr key={f.id} className="hover:bg-slate-50/60">
                    <td className="px-4 py-3 font-semibold text-rose-700">{f.trigger_reason}</td>
                    <td className="px-4 py-3 text-slate-600">{f.assigned_role}</td>
                    <td className="px-4 py-3 font-mono">{f.priority}</td>
                    <td className="px-4 py-3 text-slate-500">{new Date(f.due_date).toLocaleDateString()}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex px-2 py-0.5 rounded-full font-medium ${
                          f.status === 'RESOLVED' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {f.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      {f.status !== 'RESOLVED' && (
                        <button
                          onClick={() => setResolvingTaskId(f.id)}
                          className="text-xs font-semibold text-emerald-600 hover:text-emerald-800"
                        >
                          Resolve Task
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Tab 4: Verified Care-Gap Closure */}
      {activeTab === 'closure' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-bold text-slate-900">Module G: Verified Clinical Care-Gap Closure</h3>
            <p className="text-xs text-slate-500 mt-1">
              Care gaps cannot be closed by superficial button clicks. Closure strictly requires verified clinical evidence
              (Encounter, Screening Record, Lab Result, or Referral Outcome).
            </p>
          </div>

          <div className="max-w-xl space-y-4 text-sm">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Target Patient</label>
              <select
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                className="w-full rounded-lg border-slate-300 text-sm"
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.full_name} ({p.population_classification})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Care Gap to Close</label>
              <select
                value={closureGapId}
                onChange={(e) => setClosureGapId(e.target.value)}
                className="w-full rounded-lg border-slate-300 text-sm"
              >
                <option value="GAP-IFHAS-COMPREHENSIVE">GAP-IFHAS-COMPREHENSIVE (Adult Periodic Exam)</option>
                <option value="GAP-WELL-CHILD-DUE">GAP-WELL-CHILD-DUE (Pediatric Well-Child Surveillance)</option>
                <option value="GAP-BP-OVERDUE">GAP-BP-OVERDUE (Blood Pressure Reassessment)</option>
                <option value="GAP-DM-GLUCOSE-OVERDUE">GAP-DM-GLUCOSE-OVERDUE (Diabetes Fasting Glucose)</option>
                <option value="GAP-LIPID-SPECIALIST-EVALUATION">GAP-LIPID-SPECIALIST-EVALUATION (Cardiology Referral)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Verifiable Evidence Source</label>
              <select
                value={evidenceSource}
                onChange={(e) => setEvidenceSource(e.target.value)}
                className="w-full rounded-lg border-slate-300 text-sm"
              >
                <option value="SCREENING_RECORD">Verified Screening Encounter Record</option>
                <option value="ENCOUNTER">Completed Clinical Outpatient Encounter</option>
                <option value="LAB_RESULT">Verified Laboratory Result (LOINC match)</option>
                <option value="REFERRAL_OUTCOME">Specialist Referral Outcome Report</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Evidence Record ID <span className="text-rose-500">*</span>
              </label>
              <input
                type="text"
                value={evidenceId}
                onChange={(e) => setEvidenceId(e.target.value)}
                placeholder="e.g. enc_ifhas_2026_09118 or lab_glucose_8821"
                className="w-full rounded-lg border-slate-300 text-sm font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Clinician Verification Notes</label>
              <textarea
                value={closureNotes}
                onChange={(e) => setClosureNotes(e.target.value)}
                placeholder="Doctor attestation notes: confirmed completed screening and acceptable physiological parameters..."
                className="w-full rounded-lg border-slate-300 text-sm h-20"
              />
            </div>

            <button
              onClick={handleCloseCareGapWithEvidence}
              className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium shadow-sm"
            >
              Verify Evidence & Close Care Gap
            </button>
          </div>
        </div>
      )}

      {/* Book Appointment Modal */}
      {showBookModal && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Schedule Preventive Appointment</h3>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Service Type</label>
                <select
                  value={serviceType}
                  onChange={(e) => setServiceType(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                >
                  <option value="WELL_CHILD_VISIT">Well-Child Examination</option>
                  <option value="ADULT_IFHAS_COMPREHENSIVE">ADPHC IFHAS Screening</option>
                  <option value="CARDIOMETABOLIC_BP_CHECK">Cardiometabolic BP Check</option>
                  <option value="GENOMIC_COUNSELING">Genomic Counseling Consultation</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Appointment Time</label>
                <input
                  type="datetime-local"
                  value={apptTime}
                  onChange={(e) => setApptTime(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowBookModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleBookAppointment}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium"
              >
                Confirm Booking
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Referral Modal */}
      {showReferralModal && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Issue Specialist Referral</h3>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Target Specialty</label>
                <select
                  value={specialty}
                  onChange={(e) => setSpecialty(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                >
                  <option value="Preventive Cardiology">Preventive Cardiology</option>
                  <option value="Endocrinology">Endocrinology & Diabetes</option>
                  <option value="Medical Genetics">Medical Genetics & Genomic Counseling</option>
                  <option value="Pediatrics">Pediatric Specialty Clinic</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Indication</label>
                <textarea
                  value={clinicalIndication}
                  onChange={(e) => setClinicalIndication(e.target.value)}
                  placeholder="Enter medical justification and referral reason..."
                  className="w-full rounded-lg border-slate-300 text-sm h-20"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowReferralModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateReferral}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
              >
                Submit Referral
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Resolve Follow-Up Modal */}
      {resolvingTaskId && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Resolve Follow-Up Task</h3>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Resolution Notes</label>
                <textarea
                  value={resolutionNotes}
                  onChange={(e) => setResolutionNotes(e.target.value)}
                  placeholder="Document actions taken to resolve missed care..."
                  className="w-full rounded-lg border-slate-300 text-sm h-24"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => {
                  setResolvingTaskId(null);
                  setResolutionNotes('');
                }}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={() => handleResolveFollowUp(resolvingTaskId)}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium"
              >
                Confirm Resolution
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function NavigationPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Navigation...</div>}>
      <NavigationContent />
    </Suspense>
  );
}
