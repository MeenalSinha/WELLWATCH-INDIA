'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { ConsentRecord, TokenResponse } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  ShieldCheck,
  Lock,
  FileCheck2,
  AlertCircle,
  CheckCircle2,
  Plus,
  Trash2,
  Scale,
  Shield,
} from 'lucide-react';

export default function ConsentPage() {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [consents, setConsents] = useState<ConsentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [purpose, setPurpose] = useState('PREVENTIVE_SCREENING_ELIGIBILITY');
  const [granting, setGranting] = useState(false);
  const [msg, setMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadConsents = () => {
    apiClient
      .getOwnConsents()
      .then(setConsents)
      .catch(() => setConsents([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    const active = getStoredUser();
    if (!active) {
      router.push('/login');
      return;
    }
    setUser(active);
    loadConsents();
  }, [router]);

  const handleGrantConsent = async (e: React.FormEvent) => {
    e.preventDefault();
    setGranting(true);
    setMsg(null);

    try {
      await apiClient.grantConsent({
        purpose,
        data_categories: ['vitals', 'chronic_conditions', 'family_history'],
        consent_version: 'v1.0',
      });
      setMsg({ type: 'success', text: 'Consent directive granted and recorded in tamper-evident audit log.' });
      loadConsents();
    } catch (err: any) {
      setMsg({ type: 'error', text: err.message || 'Failed to grant consent.' });
    } finally {
      setGranting(false);
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm('Are you sure you wish to revoke this consent directive? Revocation halts future preventive processing.')) return;
    setMsg(null);
    try {
      await apiClient.revokeConsent(id, 'Resident portal self-revocation');
      setMsg({ type: 'success', text: 'Consent directive successfully revoked.' });
      loadConsents();
    } catch (err: any) {
      setMsg({ type: 'error', text: err.message || 'Failed to revoke consent.' });
    }
  };

  return (
    <AppShell activePage="/consent">
      <div className="max-w-4xl mx-auto space-y-6">
        <PageHeader
          breadcrumbs={['Platform & Governance', 'Consent Directives']}
          title="Sovereign Consent &amp; Privacy Management"
          subtitle="Purpose-based processing authorization under UAE Personal Data Protection Law (PDPL), Federal Law No. 2 of 2019, and ADHICS standards."
        />

        {/* Regulatory Governance Banner */}
        <div className="p-4 rounded-xl bg-forest-50/70 border border-forest-200 text-xs text-forest-950 flex items-start gap-3">
          <Scale className="w-5 h-5 text-forest-700 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold block text-forest-900">
              Wadeema’s Law &amp; ADHICS Consent Framework
            </span>
            <p className="text-slate-700 leading-relaxed">
              Every preventive care intelligence workflow requires explicit, granular, purpose-bound authorization. Pediatric data access strictly adheres to UAE Federal Law No. 3 of 2016 (Wadeema’s Law) requiring verified parental/guardian consent. Revocations take effect immediately.
            </p>
          </div>
        </div>

        {msg && (
          <div
            className={`p-4 rounded-xl text-xs font-medium border flex items-center gap-2 ${
              msg.type === 'success'
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                : 'bg-rose-50 text-rose-800 border-rose-200'
            }`}
          >
            {msg.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            )}
            <span>{msg.text}</span>
          </div>
        )}

        {/* Grant New Consent Card */}
        <div className="card-healthcare p-6 space-y-4">
          <h2 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
            <Plus className="w-3.5 h-3.5 text-forest-700" />
            Grant New Processing Authorization Directive
          </h2>
          <form onSubmit={handleGrantConsent} className="flex flex-col sm:flex-row gap-3">
            <select
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
              className="flex-1 text-xs px-3 py-2 border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
            >
              <option value="PREVENTIVE_SCREENING_ELIGIBILITY">
                Preventive Screening Eligibility (DoH / IFHAS / WCV)
              </option>
              <option value="CHRONIC_RISK_ASSESSMENT">
                Chronic Risk Assessment (Probabilistic Stratification)
              </option>
              <option value="CARE_COORDINATION_OUTREACH">
                Care Coordination Outreach (SMS / Appointment Invitations)
              </option>
              <option value="POPULATION_HEALTH_ANALYTICS">
                Population Health Analytics (De-Identified Aggregation)
              </option>
            </select>

            <button
              type="submit"
              disabled={granting}
              className="btn-healthcare-primary text-xs py-2 px-5 flex items-center justify-center gap-1.5 disabled:opacity-50"
            >
              <ShieldCheck className="w-4 h-4" />
              <span>{granting ? 'Recording Directive...' : 'Grant Consent'}</span>
            </button>
          </form>
        </div>

        {/* Existing Consents Table */}
        <div className="card-healthcare overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5 text-forest-700" />
              Active &amp; Historical Consent Directives ({consents.length})
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">
              Signed by {user?.full_name || user?.email || user?.user_id}
            </span>
          </div>

          {loading ? (
            <div className="p-12 text-center text-slate-400">
              <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <span>Loading consent directives...</span>
            </div>
          ) : consents.length === 0 ? (
            <div className="p-8">
              <EmptyState
                title="No Active Consent Directives"
                description="You have not yet registered any consent directives. Grant a directive above to authorize preventive care intelligence."
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-healthcare text-xs">
                <thead>
                  <tr>
                    <th>Authorized Purpose</th>
                    <th>Permitted Scopes</th>
                    <th>Effective Date</th>
                    <th>Status</th>
                    <th className="text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {consents.map((c) => (
                    <tr key={c.id}>
                      <td className="font-semibold text-slate-900">
                        {c.purpose.replace(/_/g, ' ')}
                      </td>
                      <td className="text-slate-600">
                        {c.data_categories?.join(', ') || 'Standard clinical scopes'}
                      </td>
                      <td className="font-mono text-slate-500">
                        {new Date(c.valid_from).toLocaleDateString()}
                      </td>
                      <td>
                        <StatusBadge status={c.status} />
                      </td>
                      <td className="text-right">
                        {c.status === 'ACTIVE' && (
                          <button
                            onClick={() => handleRevoke(c.id)}
                            className="px-2.5 py-1 text-xs font-semibold rounded bg-rose-50 text-rose-800 hover:bg-rose-100 border border-rose-200 transition"
                          >
                            Revoke Directive
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
