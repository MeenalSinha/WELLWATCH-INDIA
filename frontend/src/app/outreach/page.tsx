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
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  Send,
  Languages,
  CheckCircle2,
  Clock,
  ShieldAlert,
  Info,
  RefreshCw,
  MessageSquare,
  Globe,
  Bell,
  Smartphone,
  AlertCircle,
} from 'lucide-react';

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
      alert('Outreach task queued for delivery.');
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
    <AppShell activePage="/outreach">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Care Operations', 'Outreach & Localization']}
          title="Cultural Outreach &amp; Multilingual Intelligence"
          subtitle="Culturally attuned, bilingual (العربية / English) preventive outreach with consent verification and quiet-hours enforcement."
          action={{
            label: 'Dispatch Previewed Message',
            icon: Send,
            onClick: handleDispatchOutreach,
          }}
        />

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Analytics Cards */}
        {analytics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Total Generated"
              value={analytics.total_generated}
              icon={MessageSquare}
              subtitle="Synthesized messages"
              variant="forest"
            />
            <MetricCard
              title="Delivered"
              value={analytics.total_delivered}
              icon={CheckCircle2}
              subtitle="Confirmed channels"
              variant="emerald"
            />
            <MetricCard
              title="Suppressed"
              value={analytics.total_suppressed}
              icon={ShieldAlert}
              subtitle="Consent &amp; deduplication"
              variant="amber"
            />
            <MetricCard
              title="Pending Gateway"
              value={analytics.total_failed}
              icon={Clock}
              subtitle="Sandbox boundary hold"
              variant="teal"
            />
          </div>
        )}

        {/* Honest Provider Banner */}
        {analytics && (
          <div className="p-4 rounded-xl bg-emerald-50/60 border border-emerald-200 text-xs text-forest-900 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-forest-700 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Honest Provider Boundary:</span> {analytics.honest_boundary_note}
            </div>
          </div>
        )}

        {/* 2-Column Grid: Communication Preferences & Bilingual Template Preview */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Communication Preferences */}
          <div className="card-healthcare p-6 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-3 gap-2">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                <Smartphone className="w-4 h-4 text-forest-700" />
                Resident Consent &amp; Channel Directives
              </h3>
              <select
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                className="px-2.5 py-1 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.full_name}
                  </option>
                ))}
              </select>
            </div>

            {preferences ? (
              <div className="space-y-4 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1.5">
                    Preferred Communication Language
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleLanguageChange('ar')}
                      disabled={isUpdatingPref}
                      className={`px-3.5 py-2 rounded-lg font-semibold border transition ${
                        preferences.preferred_language === 'ar'
                          ? 'bg-forest-800 text-white border-forest-800 shadow-sm'
                          : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      العربية (Arabic - Default)
                    </button>
                    <button
                      onClick={() => handleLanguageChange('en')}
                      disabled={isUpdatingPref}
                      className={`px-3.5 py-2 rounded-lg font-semibold border transition ${
                        preferences.preferred_language === 'en'
                          ? 'bg-forest-800 text-white border-forest-800 shadow-sm'
                          : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      English (LTR)
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Preferred Channel
                    </label>
                    <span className="inline-flex items-center px-3 py-1.5 bg-slate-100 text-slate-800 text-xs font-semibold rounded-lg border border-slate-200">
                      {preferences.preferred_channel}
                    </span>
                  </div>
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Enforced Quiet Hours
                    </label>
                    <span className="inline-flex items-center px-3 py-1.5 bg-slate-100 text-slate-700 text-xs font-mono rounded-lg border border-slate-200">
                      {preferences.quiet_hours_start} &ndash; {preferences.quiet_hours_end}
                    </span>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-200 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-slate-900 block">Outreach Opt-In Consent</span>
                    <span className="text-[11px] text-slate-500">
                      Explicit consent required under UAE Health Data Law &amp; ADHICS
                    </span>
                  </div>
                  <button
                    onClick={handleToggleConsent}
                    disabled={isUpdatingPref}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                      preferences.consent_opt_in
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-200 hover:bg-emerald-200'
                        : 'bg-rose-100 text-rose-800 border border-rose-200 hover:bg-rose-200'
                    }`}
                  >
                    {preferences.consent_opt_in ? 'Consent Granted' : 'Opt-Out Active'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-slate-400">Loading communication directives...</div>
            )}
          </div>

          {/* Right Column: Bilingual Template Preview */}
          <div className="card-healthcare p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-3 gap-2">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                <Languages className="w-4 h-4 text-forest-700" />
                Bilingual Template Sandbox
              </h3>
              <div className="flex items-center gap-2">
                <select
                  value={templateId}
                  onChange={(e) => setTemplateId(e.target.value)}
                  className="px-2 py-1 text-xs border border-slate-300 rounded-lg bg-white font-medium"
                >
                  <option value="TMPL-PED-WELL-CHILD-INVITE">Well-Child Invitation</option>
                  <option value="TMPL-ADULT-IFHAS-RECALL">Adult IFHAS Recall</option>
                  <option value="TMPL-GENOMIC-FOLLOWUP">Genomic Counseling Follow-up</option>
                </select>
                <div className="inline-flex rounded-lg p-0.5 bg-slate-100 border border-slate-200 text-xs">
                  <button
                    onClick={() => setPreviewLang('ar')}
                    className={`px-2.5 py-1 rounded font-semibold ${
                      previewLang === 'ar' ? 'bg-white text-forest-900 shadow-sm' : 'text-slate-600'
                    }`}
                  >
                    عربي (RTL)
                  </button>
                  <button
                    onClick={() => setPreviewLang('en')}
                    className={`px-2.5 py-1 rounded font-semibold ${
                      previewLang === 'en' ? 'bg-white text-forest-900 shadow-sm' : 'text-slate-600'
                    }`}
                  >
                    EN (LTR)
                  </button>
                </div>
              </div>
            </div>

            {preview ? (
              <div
                className={`p-5 rounded-xl border border-slate-200 bg-slate-50 space-y-3 ${
                  previewLang === 'ar' ? 'text-right font-sans' : 'text-left'
                }`}
                dir={previewLang === 'ar' ? 'rtl' : 'ltr'}
              >
                <div className="flex items-center justify-between text-[11px] text-slate-400 border-b border-slate-200/80 pb-2">
                  <span className="font-mono">{preview.template_id}</span>
                  <span className="uppercase font-semibold text-forest-800">{preview.language} &bull; {preview.direction}</span>
                </div>
                {preview.rendered_subject && (
                  <h4 className="font-bold text-slate-900 text-sm">{preview.rendered_subject}</h4>
                )}
                <p className="text-xs text-slate-800 leading-relaxed whitespace-pre-wrap">
                  {preview.rendered_body}
                </p>
                <div className="pt-2 border-t border-slate-200/60 text-[10px] text-slate-400">
                  {previewLang === 'ar'
                    ? 'رسالة معتمدة من مركز أبوظبي للصحة العامة (ADPHC)'
                    : 'Validated by Abu Dhabi Public Health Center (ADPHC)'}
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-400">Loading template preview...</div>
            )}
          </div>
        </div>

        {/* Outreach Tasks Log Table */}
        <div className="card-healthcare overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-forest-700" />
              Scheduled &amp; Dispatched Outreach Tasks ({tasks.length})
            </h3>
            <span className="text-[11px] text-slate-500">
              Audit log linked to resident record
            </span>
          </div>

          {tasks.length === 0 ? (
            <div className="p-8">
              <EmptyState
                title="No Outreach History"
                description="No messages or notifications have been dispatched to this resident yet."
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-healthcare text-xs">
                <thead>
                  <tr>
                    <th>Task ID</th>
                    <th>Template ID</th>
                    <th>Delivery Channel</th>
                    <th>Status</th>
                    <th>Dispatch Timestamp</th>
                    <th className="text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {tasks.map((t) => (
                    <tr key={t.id}>
                      <td className="font-mono text-slate-800 font-medium">{t.id.slice(0, 12)}...</td>
                      <td className="font-mono text-slate-600">{t.template_id}</td>
                      <td>
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700">
                          {t.channel}
                        </span>
                      </td>
                      <td>
                        <StatusBadge status={t.delivery_status || t.status || 'QUEUED'} />
                      </td>
                      <td className="font-mono text-slate-500">
                        {t.scheduled_at || t.scheduled_time || t.created_at || 'Instant'}
                      </td>
                      <td className="text-right">
                        {(t.delivery_status === 'FAILED' || t.status === 'FAILED') && (
                          <button
                            onClick={() => handleRetry(t.id)}
                            className="btn-healthcare-secondary text-[11px] py-1 px-2.5 inline-flex items-center gap-1"
                          >
                            <RefreshCw className="w-3 h-3 text-forest-700" />
                            <span>Retry</span>
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
    </AppShell>
  );
}

export default function OutreachPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading outreach intelligence...</span>
        </div>
      }
    >
      <OutreachContent />
    </Suspense>
  );
}
