'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { setSession } from '@/lib/auth';
import { UserRole } from '@/types';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('healthcare_provider@wellwatch.ae');
  const [password, setPassword] = useState('ValidPass123!');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const personaOptions = [
    { label: 'Healthcare Provider (Physician)', email: 'healthcare_provider@wellwatch.ae' },
    { label: 'Community Health Worker (CHW)', email: 'community_health_worker@wellwatch.ae' },
    { label: 'Individual (Patient)', email: 'individual@wellwatch.ae' },
    { label: 'Parent / Guardian', email: 'parent_guardian@wellwatch.ae' },
    { label: 'Care Coordinator', email: 'care_coordinator@wellwatch.ae' },
    { label: 'Population Health Admin', email: 'population_health_admin@wellwatch.ae' },
    { label: 'System Administrator', email: 'system_admin@wellwatch.ae' },
    { label: 'Data Governance Officer', email: 'data_governance_officer@wellwatch.ae' },
  ];

  const handlePersonaSelect = (selectedEmail: string) => {
    setEmail(selectedEmail);
    setPassword('ValidPass123!');
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const tokenData = await apiClient.login(email, password);
      setSession(tokenData);
      router.push('/dashboard');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Authentication failed. Please verify credentials.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto my-8">
      <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
        <div className="text-center">
          <div className="w-10 h-10 rounded-xl bg-emerald-800 text-white flex items-center justify-center font-bold text-xl mx-auto mb-3 shadow">
            W
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Sign In to WELLWATCH PREVENT</h2>
          <p className="text-xs text-slate-500 mt-1">Authorized Healthcare Identity & Access Gateway</p>
        </div>

        {/* Quick-Select Persona Switcher */}
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
          <label className="block text-[11px] font-semibold text-slate-600 mb-1.5 uppercase tracking-wider">
            Quick-Select Test Persona
          </label>
          <select
            className="w-full text-xs bg-white border border-slate-300 rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-emerald-600"
            value={email}
            onChange={(e) => handlePersonaSelect(e.target.value)}
          >
            {personaOptions.map((opt) => (
              <option key={opt.email} value={opt.email}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {error && (
          <div className="p-3 rounded-md bg-rose-50 border border-rose-200 text-xs text-rose-800 font-medium">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Email Address</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700"
              placeholder="user@wellwatch.ae"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 rounded-lg bg-emerald-800 text-white font-semibold text-sm hover:bg-emerald-900 transition shadow disabled:opacity-50"
          >
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>

        <div className="text-center text-[11px] text-slate-400 border-t border-slate-100 pt-4">
          Access is strictly restricted to authorized healthcare personnel and residents under UAE legislation.
        </div>
      </div>
    </div>
  );
}
