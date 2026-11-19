'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { AuditLog, TokenResponse } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  ShieldAlert,
  ShieldCheck,
  Lock,
  FileText,
  Clock,
  AlertTriangle,
  Fingerprint,
} from 'lucide-react';

export default function AuditPage() {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const active = getStoredUser();
    if (!active) {
      router.push('/login');
      return;
    }
    setUser(active);

    apiClient
      .listAuditLogs()
      .then(setLogs)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 403) {
          setError('Access Restricted: The system audit trail requires DATA_GOVERNANCE_OFFICER permissions.');
        } else {
          setError(err.message || 'Failed to load audit logs.');
        }
      })
      .finally(() => setLoading(false));
  }, [router]);

  return (
    <AppShell activePage="/audit">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Platform & Governance', 'Audit Logs']}
          title="System Audit &amp; Governance Trail"
          subtitle="Immutable, tamper-evident security and data access event logs under ADHICS Tier 3 and UAE Federal Health Data Law."
        />

        {error ? (
          <div className="card-healthcare p-6 border-l-4 border-l-amber-500 bg-amber-50/40 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              <span>Role Authorization Guard Active</span>
            </div>
            <p className="text-xs text-amber-800 leading-relaxed">{error}</p>
            <p className="text-xs text-slate-600 pt-2 border-t border-amber-200">
              To inspect the cryptographic audit trail, switch your active persona to{' '}
              <strong className="text-slate-900">Data Governance Officer</strong> via the Sign In persona picker.
            </p>
          </div>
        ) : (
          <div className="card-healthcare overflow-hidden space-y-4">
            <div className="p-5 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h2 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <Fingerprint className="w-4 h-4 text-forest-700" />
                  Recorded Security &amp; Access Events ({logs.length})
                </h2>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Append-only immutable event stream with correlation IDs
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-forest-50 border border-forest-200 text-[11px] text-forest-900 font-semibold">
                <Lock className="w-3.5 h-3.5 text-forest-700" />
                <span>Zero PHI Invariance Enforced</span>
              </div>
            </div>

            {loading ? (
              <div className="p-12 text-center text-slate-400">
                <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <span>Verifying security logs and audit signatures...</span>
              </div>
            ) : logs.length === 0 ? (
              <div className="p-8">
                <EmptyState
                  title="No Audit Logs Recorded"
                  description="System transactions, screening events, and access attempts will automatically populate this ledger."
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="table-healthcare text-xs">
                  <thead>
                    <tr>
                      <th>Event Timestamp</th>
                      <th>Action</th>
                      <th>Resource Type &amp; ID</th>
                      <th>Outcome</th>
                      <th>Correlation ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((log) => (
                      <tr key={log.id}>
                        <td className="font-mono text-slate-600 whitespace-nowrap">
                          {new Date(log.created_at).toLocaleString()}
                        </td>
                        <td className="font-semibold text-slate-900">
                          {log.action}
                        </td>
                        <td className="text-slate-600 font-mono text-[11px]">
                          {log.resource_type}{' '}
                          {log.resource_id ? `(${log.resource_id.substring(0, 8)}...)` : ''}
                        </td>
                        <td>
                          <StatusBadge status={log.outcome} />
                        </td>
                        <td className="font-mono text-slate-400 text-[11px]">
                          {log.correlation_id.substring(0, 13)}...
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </AppShell>
  );
}
