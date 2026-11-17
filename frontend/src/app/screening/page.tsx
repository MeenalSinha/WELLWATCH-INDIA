'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { TokenResponse } from '@/types';

export default function ScreeningPage() {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [patientId, setPatientId] = useState('');
  const [encounterType, setEncounterType] = useState('IFHAS_COMPREHENSIVE');
  const [systolic, setSystolic] = useState('128');
  const [diastolic, setDiastolic] = useState('82');
  const [pulse, setPulse] = useState('74');
  const [glucose, setGlucose] = useState('105');
  const [height, setHeight] = useState('175');
  const [weight, setWeight] = useState('76');
  const [epistemicStatus, setEpistemicStatus] = useState('VERIFIED');
  const [notes, setNotes] = useState('Periodic preventive assessment.');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<any | null>(null);

  useEffect(() => {
    const activeUser = getStoredUser();
    if (!activeUser) {
      router.push('/login');
      return;
    }
    setUser(activeUser);
    setPatientId(activeUser.user_id);
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    const sBp = parseFloat(systolic);
    const dBp = parseFloat(diastolic);

    if (dBp >= sBp) {
      setError('Clinical Validation Error: Diastolic BP must be strictly less than Systolic BP.');
      setLoading(false);
      return;
    }

    try {
      const payload = {
        patient_id: patientId,
        facility_id: user?.organization_id || 'SEHA-BATEEN-001',
        encounter_type: encounterType,
        systolic_bp: sBp,
        diastolic_bp: dBp,
        pulse_rate: pulse ? parseInt(pulse) : undefined,
        fasting_glucose: glucose ? parseFloat(glucose) : undefined,
        height_cm: height ? parseFloat(height) : undefined,
        weight_kg: weight ? parseFloat(weight) : undefined,
        epistemic_status: epistemicStatus as any,
        notes,
      };

      const res = await apiClient.recordScreening(payload);
      setSuccess(res);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to record screening encounter.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto my-6">
      <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">Record Clinical Vitals</h1>
          <p className="text-xs text-slate-500 mt-1">
            Standardized Intake for Abu Dhabi DoH Well-Child & IFHAS Screening Pathways
          </p>
        </div>

        {error && (
          <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 font-medium">
            {error}
          </div>
        )}

        {success && (
          <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 space-y-2">
            <div className="font-bold flex items-center gap-1.5">
              <span>✅</span> Screening Encounter Successfully Persisted to PostgreSQL
            </div>
            <div className="text-slate-700">
              Encounter ID: <span className="font-mono">{success.id}</span> · Computed BMI:{' '}
              <span className="font-semibold">{success.bmi}</span>
            </div>
            <button
              onClick={() => router.push('/dashboard')}
              className="mt-2 text-xs font-semibold text-emerald-800 underline"
            >
              ← Return to Clinical Dashboard
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Encounter Pathway</label>
              <select
                value={encounterType}
                onChange={(e) => setEncounterType(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700"
              >
                <option value="IFHAS_COMPREHENSIVE">IFHAS Comprehensive Screening (Adult)</option>
                <option value="WELL_CHILD_VISIT">Well-Child Visit (Pediatric)</option>
                <option value="COMMUNITY_VITALS_CHECK">Community Vitals Check (CHW)</option>
                <option value="CLINIC_FOLLOWUP">Clinic Recall Follow-up</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Epistemic Status</label>
              <select
                value={epistemicStatus}
                onChange={(e) => setEpistemicStatus(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700"
              >
                <option value="VERIFIED">VERIFIED (Clinically Confirmed)</option>
                <option value="UNVERIFIED">UNVERIFIED (CHW Field / Self-Report)</option>
                <option value="MISSING">MISSING (Incomplete Panel)</option>
                <option value="UNKNOWN">UNKNOWN (Unverified Baseline)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Systolic BP</label>
              <div className="relative">
                <input
                  type="number"
                  step="0.1"
                  required
                  value={systolic}
                  onChange={(e) => setSystolic(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  placeholder="120"
                />
                <span className="absolute right-2 top-2 text-[10px] text-slate-400">mmHg</span>
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Diastolic BP</label>
              <div className="relative">
                <input
                  type="number"
                  step="0.1"
                  required
                  value={diastolic}
                  onChange={(e) => setDiastolic(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  placeholder="80"
                />
                <span className="absolute right-2 top-2 text-[10px] text-slate-400">mmHg</span>
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Pulse</label>
              <div className="relative">
                <input
                  type="number"
                  value={pulse}
                  onChange={(e) => setPulse(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  placeholder="72"
                />
                <span className="absolute right-2 top-2 text-[10px] text-slate-400">bpm</span>
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Glucose</label>
              <div className="relative">
                <input
                  type="number"
                  step="0.1"
                  value={glucose}
                  onChange={(e) => setGlucose(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  placeholder="100"
                />
                <span className="absolute right-2 top-2 text-[10px] text-slate-400">mg/dL</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Height (cm)</label>
              <input
                type="number"
                step="0.1"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                placeholder="175"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Weight (kg)</label>
              <input
                type="number"
                step="0.1"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                placeholder="75"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">Clinical Notes & Observations</label>
            <textarea
              rows={2}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
              placeholder="Clinical observations or referral indications..."
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 rounded-lg bg-emerald-800 text-white font-semibold text-sm hover:bg-emerald-900 transition shadow disabled:opacity-50"
          >
            {loading ? 'Recording Encounter...' : 'Submit Screening Encounter'}
          </button>
        </form>
      </div>
    </div>
  );
}
