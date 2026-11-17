'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { ConsentRecord, TokenResponse } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import { LoadingSpinner } from '@/components/LoadingSpinner';

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
      setMsg({ type: 'success', text: 'Consent directive granted and recorded in audit log.' });
      loadConsents();
    } catch (err: any) {
      setMsg({ type: 'error', text: err.message || 'Failed to grant consent.' });
    } finally {
      setGranting(false);
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm('Are you sure you wish to revoke this consent directive?')) return;
    setMsg(null);
    try {
      await apiClient.revokeConsent(id, 'User portal self-revocation');
      setMsg({ type: 'success', text: 'Consent successfully revoked.' });
      loadConsents();
    } catch (err: any) {
      setMsg({ type: 'error', text: err.message || 'Failed to revoke consent.' });
    }
  };

  if (!user) return <LoadingSpinner size="lg" />;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Consent & Privacy Management</h1>
        <p className="text-xs text-slate-500 mt-1">
          Purpose-based processing authorization under UAE Personal Data Protection Law (PDPL) & ADHICS
        </p>
      </div>

      {msg && (
        <div
          className={`p-3.5 rounded-lg text-xs font-medium border ${
            msg.type === 'success'
              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
              : 'bg-rose-50 text-rose-800 border-rose-200'
          }`}
        >
          {msg.text}
        </div>
      )}

      {/* Grant New Consent Card */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
          Grant New Processing Directive
        </h2>
        <form onSubmit={handleGrantConsent} className="flex flex-col sm:flex-row gap-3">
          <select
            value={purpose}
            onChange={(e) => setPurpose(e.target.value)}
            className="flex-1 text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700"
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
            className="px-5 py-2 rounded-lg bg-emerald-800 text-white font-semibold text-xs shadow hover:bg-emerald-900 transition disabled:opacity-50"
          >
            {granting ? 'Recording...' : 'Grant Consent'}
          </button>
        </form>
      </div>

      {/* Existing Consents Table */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
          Active & Historical Consents
        </h2>

        {loading ? (
          <LoadingSpinner />
        ) : consents.length === 0 ? (
          <EmptyState
            title="No Active Consent Directives"
            description="You have not yet registered any consent directives. Grant a directive above to authorize preventive care intelligence."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-600">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-3 px-4">Purpose</th>
                  <th className="py-3 px-4">Scopes</th>
                  <th className="py-3 px-4">Granted Date</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {consents.map((c) => (
                  <tr key={c.id} className="hover:bg-slate-50 transition">
                    <td className="py-3 px-4 font-semibold text-slate-900">
                      {c.purpose.replace(/_/g, ' ')}
                    </td>
                    <td className="py-3 px-4 text-slate-500">
                      {c.data_categories?.join(', ') || 'Standard scopes'}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-500">
                      {new Date(c.valid_from).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <StatusBadge status={c.status} />
                    </td>
                    <td className="py-3 px-4 text-right">
                      {c.status === 'ACTIVE' && (
                        <button
                          onClick={() => handleRevoke(c.id)}
                          className="text-xs font-semibold text-rose-700 hover:text-rose-900"
                        >
                          Revoke
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
