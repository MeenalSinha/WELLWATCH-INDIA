'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { apiClient, ApiError } from '@/lib/api';
import {
  CommunicationPreferenceResponse,
  OutreachTaskResponse,
  OutreachAnalyticsResponse,
  TemplatePreviewResponse,
  PersonSummary,
} from '@/types';

function OutreachContent() {
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [preferences, setPreferences] = useState<CommunicationPreferenceResponse | null>(null);
  const [tasks, setTasks] = useState<OutreachTaskResponse[]>([]);
  const [analytics, setAnalytics] = useState<OutreachAnalyticsResponse | null>(null);

  // Template Preview
  const [templateId, setTemplateId] = useState('TMPL-PED-WELL-CHILD-INVITE');
  const [previewLang, setPreviewLang] = useState<'ar' | 'en'>('ar');
  const [preview, setPreview] = useState<TemplatePreviewResponse | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [isUpdatingPref, setIsUpdatingPref] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    loadPreview();
  }, [templateId, previewLang]);

  useEffect(() => {
    if (selectedPatientId) {
      loadPatientPreferences(selectedPatientId);
      loadTasks(selectedPatientId);
    }
  }, [selectedPatientId]);

  async function loadInitialData() {
    setIsLoading(true);
    try {
      const [patRes, anRes] = await Promise.all([
        apiClient.getPatients(1, 100),
        apiClient.getOutreachAnalytics(),
      ]);
      setPatients(patRes.items);
      setAnalytics(anRes);
      if (patRes.items.length > 0) {
        setSelectedPatientId(patRes.items[0].id);
      }
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to load outreach data');
    } finally {
      setIsLoading(false);
    }
  }

  async function loadPatientPreferences(pId: string) {
    try {
      const pref = await apiClient.getCommunicationPreferences(pId);
      setPreferences(pref);
    } catch (err: any) {
      console.error('Failed to load preferences', err);
    }
  }

  async function loadTasks(pId?: string) {
    try {
      const res = await apiClient.getOutreachTasks(pId);
      setTasks(res.items);
    } catch (err: any) {
      console.error('Failed to load outreach tasks', err);
    }
  }

  async function loadPreview() {
    try {
      const p = await apiClient.previewTemplate({
        template_id: templateId,
        language: previewLang,
        variables: { patient_name: 'زايد / Zayed' },
      });
      setPreview(p);
    } catch (err: any) {
      console.error('Preview failed', err);
    }
  }

  async function handleToggleConsent() {
    if (!preferences || !selectedPatientId) return;
    setIsUpdatingPref(true);
    try {
      const updated = await apiClient.updateCommunicationPreferences(selectedPatientId, {
        consent_opt_in: !preferences.consent_opt_in,
      });
      setPreferences(updated);
    } catch (err: any) {
      alert('Failed to update consent preference: ' + err.message);
    } finally {
      setIsUpdatingPref(false);
    }
  }

  async function handleLanguageChange(newLang: 'ar' | 'en') {
    if (!preferences || !selectedPatientId) return;
    setIsUpdatingPref(true);
    try {
      const updated = await apiClient.updateCommunicationPreferences(selectedPatientId, {
        preferred_language: newLang,
      });
      setPreferences(updated);
    } catch (err: any) {
      alert('Failed to update language: ' + err.message);
    } finally {
      setIsUpdatingPref(false);
    }
  }

  async function handleDispatchOutreach() {
    if (!selectedPatientId) return;
    try {
      await apiClient.createOutreachTask({
        person_id: selectedPatientId,
        template_id: templateId,
      });
      alert('Outreach task scheduled/attempted.');
      loadTasks(selectedPatientId);
      const anRes = await apiClient.getOutreachAnalytics();
      setAnalytics(anRes);
    } catch (err: any) {
      alert('Outreach task creation failed: ' + err.message);
    }
  }

  async function handleRetry(taskId: string) {
    try {
      await apiClient.retryOutreachTask(taskId);
      alert('Retry dispatched.');
      loadTasks(selectedPatientId);
    } catch (err: any) {
      alert('Retry failed: ' + err.message);
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-200 pb-5">
        <h1 className="text-2xl font-bold text-slate-900">Cultural Outreach & Multilingual Intelligence</h1>
        <p className="text-sm text-slate-600 mt-1">
          Culturally appropriate, bilingual (العربية / English) preventive healthcare communication for Abu Dhabi
        </p>
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
            <span className="text-xs font-semibold text-slate-500 uppercase">Total Generated</span>
            <p className="text-2xl font-bold text-slate-900 mt-1">{analytics.total_generated}</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Delivered</span>
            <p className="text-2xl font-bold text-emerald-600 mt-1">{analytics.total_delivered}</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Suppressed (Consent/Dedup)</span>
            <p className="text-2xl font-bold text-amber-600 mt-1">{analytics.total_suppressed}</p>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase">Unconfigured / Failed</span>
            <p className="text-2xl font-bold text-slate-600 mt-1">{analytics.total_failed}</p>
          </div>
        </div>
      )}

      {/* Honest Provider Note */}
      {analytics && (
        <div className="bg-blue-50 border border-blue-200 p-4 rounded-xl text-xs text-blue-900 flex items-start gap-2">
          <span>ℹ️</span>
          <div>
            <span className="font-bold">Honest Provider Boundary:</span> {analytics.honest_boundary_note}
          </div>
        </div>
      )}

      {/* Grid: Preferences vs Template Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Preferences */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5">
          <div className="flex justify-between items-center border-b border-slate-100 pb-3">
            <h3 className="text-base font-bold text-slate-900">Communication & Consent Preferences</h3>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="rounded-lg border-slate-300 text-xs py-1.5 px-2.5"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name}
                </option>
              ))}
            </select>
          </div>

          {preferences ? (
            <div className="space-y-4 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Preferred Language</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleLanguageChange('ar')}
                    disabled={isUpdatingPref}
                    className={`px-4 py-1.5 rounded-lg text-xs font-semibold border ${
                      preferences.preferred_language === 'ar'
                        ? 'bg-emerald-600 text-white border-emerald-600'
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                  >
                    العربية (Arabic - Default)
                  </button>
                  <button
                    onClick={() => handleLanguageChange('en')}
                    disabled={isUpdatingPref}
                    className={`px-4 py-1.5 rounded-lg text-xs font-semibold border ${
                      preferences.preferred_language === 'en'
                        ? 'bg-emerald-600 text-white border-emerald-600'
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                  >
                    English (LTR)
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Preferred Channel</label>
                <span className="inline-flex items-center px-3 py-1 bg-slate-100 text-slate-800 text-xs font-medium rounded-full">
                  {preferences.preferred_channel}
                </span>
              </div>

              <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                <div>
                  <span className="block font-semibold text-slate-800 text-xs">Preventive Communication Consent</span>
                  <span className="text-xs text-slate-500">Opt-in status for health screening reminders</span>
                </div>
                <button
                  onClick={handleToggleConsent}
                  disabled={isUpdatingPref}
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    preferences.consent_opt_in
                      ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200'
                      : 'bg-rose-100 text-rose-800 hover:bg-rose-200'
                  }`}
                >
                  {preferences.consent_opt_in ? '✓ Opted In' : '✕ Opted Out'}
                </button>
              </div>

              <div className="border-t border-slate-100 pt-3">
                <button
                  onClick={handleDispatchOutreach}
                  className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium shadow-sm"
                >
                  Schedule / Send Previewed Template
                </button>
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm">Select a patient to view preferences.</div>
          )}
        </div>

        {/* Right: Live Governed Template Preview */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex justify-between items-center border-b border-slate-100 pb-3">
            <h3 className="text-base font-bold text-slate-900">Live Governed Message Preview</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPreviewLang('ar')}
                className={`px-2.5 py-1 text-xs font-semibold rounded ${
                  previewLang === 'ar' ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-700'
                }`}
              >
                العربية (RTL)
              </button>
              <button
                onClick={() => setPreviewLang('en')}
                className={`px-2.5 py-1 text-xs font-semibold rounded ${
                  previewLang === 'en' ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-700'
                }`}
              >
                English (LTR)
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Select Approved Template</label>
            <select
              value={templateId}
              onChange={(e) => setTemplateId(e.target.value)}
              className="w-full rounded-lg border-slate-300 text-xs py-1.5"
            >
              <option value="TMPL-PED-WELL-CHILD-INVITE">Well-Child Examination Invitation</option>
              <option value="TMPL-PED-VACCINE-DUE">National Immunization Reminder (Wadeema's Law)</option>
              <option value="TMPL-ADULT-IFHAS-INVITE">ADPHC IFHAS Comprehensive Screening</option>
              <option value="TMPL-ADULT-BP-RECALL">Clinical Blood Pressure Follow-Up</option>
              <option value="TMPL-ADULT-DM-RECALL">Periodic Diabetes Screening Reminder</option>
              <option value="TMPL-GENOMIC-COUNSELING">Genomic Finding Consultation (EGP)</option>
            </select>
          </div>

          {preview && (
            <div
              dir={preview.direction}
              className={`p-4 rounded-xl border ${
                preview.direction === 'rtl' ? 'bg-emerald-50/40 border-emerald-200 text-right' : 'bg-slate-50 border-slate-200 text-left'
              }`}
            >
              <p className="font-bold text-sm text-slate-900 border-b border-slate-200/60 pb-2">
                {preview.rendered_subject}
              </p>
              <p className="text-sm text-slate-800 mt-3 whitespace-pre-wrap leading-relaxed">
                {preview.rendered_body}
              </p>
              <div className="mt-4 pt-2 border-t border-slate-200/60 flex items-center justify-between text-xs text-slate-500">
                <span className="font-mono">Direction: {preview.direction.toUpperCase()}</span>
                <span className="font-semibold text-emerald-700">✓ Governed Standard Approved</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Dispatched / Scheduled Outreach Tasks */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
        <h3 className="text-base font-bold text-slate-900">Outreach Tasks & Delivery Audit</h3>
        {tasks.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-sm">No outreach tasks found for selected patient.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50 text-slate-600 font-medium">
                <tr>
                  <th className="px-4 py-2.5 text-left">Channel</th>
                  <th className="px-4 py-2.5 text-left">Recipient Role</th>
                  <th className="px-4 py-2.5 text-left">Language</th>
                  <th className="px-4 py-2.5 text-left">Delivery Status</th>
                  <th className="px-4 py-2.5 text-left">Provider Status</th>
                  <th className="px-4 py-2.5 text-left">Scheduled / Attempted</th>
                  <th className="px-4 py-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {tasks.map((t) => (
                  <tr key={t.id} className="hover:bg-slate-50/60">
                    <td className="px-4 py-3 font-medium text-slate-800">{t.channel}</td>
                    <td className="px-4 py-3 text-slate-600">{t.recipient_role}</td>
                    <td className="px-4 py-3 font-mono">{t.language.toUpperCase()}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex px-2 py-0.5 rounded-full font-medium ${
                          t.delivery_status === 'DELIVERED'
                            ? 'bg-emerald-100 text-emerald-800'
                            : t.delivery_status === 'SUPPRESSED'
                            ? 'bg-amber-100 text-amber-800'
                            : t.delivery_status === 'FAILED'
                            ? 'bg-rose-100 text-rose-800'
                            : 'bg-slate-100 text-slate-800'
                        }`}
                      >
                        {t.delivery_status}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-600">
                      {t.provider_status}
                      {t.failure_reason && (
                        <span className="block text-[10px] text-rose-600 line-clamp-1">{t.failure_reason}</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-500">
                      {new Date(t.scheduled_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {t.delivery_status === 'FAILED' && (
                        <button
                          onClick={() => handleRetry(t.id)}
                          className="text-xs font-semibold text-blue-600 hover:text-blue-800"
                        >
                          Retry ({t.retry_count}/{t.max_retries})
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
    </div>
  );
}

export default function OutreachPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Outreach...</div>}>
      <OutreachContent />
    </Suspense>
  );
}
