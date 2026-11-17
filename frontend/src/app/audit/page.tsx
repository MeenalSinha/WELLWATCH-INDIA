'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { AuditLog, TokenResponse } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import { LoadingSpinner } from '@/components/LoadingSpinner';

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
          setError('Access Denied: The system audit trail requires DATA_GOVERNANCE_OFFICER permissions.');
        } else {
          setError(err.message || 'Failed to load audit logs.');
        }
      })
      .finally(() => setLoading(false));
  }, [router]);

  if (!user) return <LoadingSpinner size="lg" />;

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">System Audit & Governance Trail</h1>
        <p className="text-xs text-slate-500 mt-1">
          Immutable, tamper-evident security and data access event logs (ADHICS & UAE Health Data Law)
        </p>
      </div>

      {error ? (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-2">
          <div className="font-bold flex items-center gap-1.5">
            <span>⚠️</span> Role Authorization Guard Active
          </div>
          <p>{error}</p>
          <p className="text-slate-600">
            To inspect the system audit trail, switch your active persona to{' '}
            <span className="font-semibold">Data Governance Officer</span> from the Sign In page.
          </p>
        </div>
      ) : (
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              Recorded Security & Access Events
            </h2>
            <span className="text-xs text-slate-500">{logs.length} events logged</span>
          </div>

          <div className="p-2.5 rounded bg-slate-50 border border-slate-200 text-[11px] text-slate-600">
            🔒 <strong>Zero PHI Invariance:</strong> Audit entries strictly log entity identifiers, actions, and
            outcomes. Protected Health Information, clinical observations, and passwords are cryptographically
            excluded from audit storage.
          </div>

          {loading ? (
            <LoadingSpinner />
          ) : logs.length === 0 ? (
            <EmptyState
              title="No Audit Logs Recorded"
              description="System transactions and access attempts will automatically populate this ledger."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-600">
                <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-3 px-4">Timestamp</th>
                    <th className="py-3 px-4">Action</th>
                    <th className="py-3 px-4">Resource</th>
                    <th className="py-3 px-4">Outcome</th>
                    <th className="py-3 px-4">Correlation ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-slate-50 transition">
                      <td className="py-2.5 px-4 text-slate-500">
                        {new Date(log.created_at).toLocaleString()}
                      </td>
                      <td className="py-2.5 px-4 font-semibold text-slate-900">
                        {log.action}
                      </td>
                      <td className="py-2.5 px-4 text-slate-600">
                        {log.resource_type} {log.resource_id ? `(${log.resource_id.substring(0, 8)}...)` : ''}
                      </td>
                      <td className="py-2.5 px-4">
                        <StatusBadge status={log.outcome} />
                      </td>
                      <td className="py-2.5 px-4 text-slate-400">
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
  );
}
