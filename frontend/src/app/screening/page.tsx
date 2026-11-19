'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { getStoredUser } from '@/lib/auth';
import { TokenResponse, PersonSummary } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import {
  Stethoscope,
  Activity,
  Heart,
  Droplets,
  Scale,
  Ruler,
  FileText,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  ShieldCheck,
  User,
} from 'lucide-react';

export default function ScreeningPage() {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [patientId, setPatientId] = useState('');
  const [encounterType, setEncounterType] = useState('IFHAS_COMPREHENSIVE');
  const [systolic, setSystolic] = useState('122');
  const [diastolic, setDiastolic] = useState('78');
  const [pulse, setPulse] = useState('72');
  const [glucose, setGlucose] = useState('98');
  const [height, setHeight] = useState('174');
  const [weight, setWeight] = useState('74');
  const [epistemicStatus, setEpistemicStatus] = useState('VERIFIED');
  const [notes, setNotes] = useState('Routine preventive health intake under Abu Dhabi DoH standards.');

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

    apiClient
      .listPatients(undefined, undefined, 0, 50)
      .then((res) => {
        setPatients(res.items);
        if (res.items.length > 0) {
          setPatientId(res.items[0].id);
        } else {
          setPatientId(activeUser.user_id);
        }
      })
      .catch(() => {
        setPatientId(activeUser.user_id);
      });
  }, [router]);

  // Real-time BMI calculation
  const hNum = parseFloat(height);
  const wNum = parseFloat(weight);
  const computedBmi =
    hNum > 0 && wNum > 0 ? (wNum / Math.pow(hNum / 100, 2)).toFixed(1) : null;

  const getBmiCategory = (bmi: number) => {
    if (bmi < 18.5) return { label: 'Underweight', color: 'text-amber-600 bg-amber-50 border-amber-200' };
    if (bmi < 25) return { label: 'Normal Weight', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
    if (bmi < 30) return { label: 'Overweight', color: 'text-amber-700 bg-amber-50 border-amber-200' };
    return { label: 'Obese', color: 'text-rose-700 bg-rose-50 border-rose-200' };
  };

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
    <AppShell activePage="/screening">
      <div className="max-w-4xl mx-auto space-y-6">
        <PageHeader
          breadcrumbs={['Preventive Care', 'Clinical Screening']}
          title="Clinical Screening Intake"
          subtitle="Standardized physiological intake for Abu Dhabi DoH Well-Child &amp; IFHAS Comprehensive Screening Pathways."
        />

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Clinical Validation Error:</span> {error}
            </div>
          </div>
        )}

        {success && (
          <div className="card-healthcare p-6 border-l-4 border-l-emerald-600 bg-emerald-50/40 space-y-4">
            <div className="flex items-center gap-2 text-forest-900 font-bold text-sm">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span>Screening Encounter Successfully Persisted to PostgreSQL</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-700 bg-white p-3 rounded-lg border border-slate-200">
              <div>
                <span className="text-slate-400 block">Encounter ID</span>
                <span className="font-mono text-slate-900 font-medium">{success.id}</span>
              </div>
              <div>
                <span className="text-slate-400 block">Calculated BMI</span>
                <span className="font-semibold text-forest-900">{success.bmi || computedBmi || 'N/A'}</span>
              </div>
              <div>
                <span className="text-slate-400 block">Epistemic Status</span>
                <span className="font-medium text-emerald-700">{success.epistemic_status || epistemicStatus}</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-3 pt-2">
              <Link
                href={`/patients/${patientId}`}
                className="btn-healthcare-primary text-xs py-2 px-3 inline-flex items-center gap-1.5"
              >
                <span>View Patient Profile</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
              <Link
                href="/pathways"
                className="btn-healthcare-secondary text-xs py-2 px-3 inline-flex items-center gap-1.5"
              >
                <span>Evaluate Care Pathways</span>
              </Link>
              <button
                onClick={() => setSuccess(null)}
                className="btn-healthcare-secondary text-xs py-2 px-3"
              >
                Record Another Vitals Intake
              </button>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Form */}
          <div className="lg:col-span-2 card-healthcare p-6 space-y-6">
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Patient Selection */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1 flex items-center gap-1.5">
                  <User className="w-3.5 h-3.5 text-forest-700" />
                  Target Resident / Patient *
                </label>
                <select
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white font-medium"
                >
                  {patients.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs &bull; {p.primary_identifier_masked || p.identifier_value || 'No EID'})
                    </option>
                  ))}
                </select>
              </div>

              {/* Encounter Type and Epistemic Status */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Encounter Pathway *
                  </label>
                  <select
                    value={encounterType}
                    onChange={(e) => setEncounterType(e.target.value)}
                    className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white font-medium"
                  >
                    <option value="IFHAS_COMPREHENSIVE">IFHAS Comprehensive Screening (Adult)</option>
                    <option value="WELL_CHILD_VISIT">Well-Child Visit (Pediatric)</option>
                    <option value="COMMUNITY_VITALS_CHECK">Community Vitals Check (CHW)</option>
                    <option value="CLINIC_FOLLOWUP">Clinic Recall Follow-up</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Epistemic Status *
                  </label>
                  <select
                    value={epistemicStatus}
                    onChange={(e) => setEpistemicStatus(e.target.value)}
                    className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white font-medium"
                  >
                    <option value="VERIFIED">VERIFIED (Clinically Confirmed at Facility)</option>
                    <option value="UNVERIFIED">UNVERIFIED (CHW Field / Self-Reported)</option>
                    <option value="PROVISIONAL">PROVISIONAL (Awaiting Laboratory Confirmation)</option>
                  </select>
                </div>
              </div>

              {/* Vitals: BP & Pulse */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="text-xs font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-forest-700" />
                  Cardiovascular Telemetry
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Systolic BP
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        step="0.1"
                        required
                        value={systolic}
                        onChange={(e) => setSystolic(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="120"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">mmHg</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Diastolic BP
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        step="0.1"
                        required
                        value={diastolic}
                        onChange={(e) => setDiastolic(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="80"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">mmHg</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Pulse Rate
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        value={pulse}
                        onChange={(e) => setPulse(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="72"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">bpm</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Metabolic & Biometric */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="text-xs font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <Scale className="w-3.5 h-3.5 text-forest-700" />
                  Metabolic &amp; Anthropometric Panels
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Fasting Glucose
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        step="0.1"
                        value={glucose}
                        onChange={(e) => setGlucose(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="100"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">mg/dL</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Height
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        step="0.1"
                        value={height}
                        onChange={(e) => setHeight(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="175"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">cm</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Weight
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        step="0.1"
                        value={weight}
                        onChange={(e) => setWeight(e.target.value)}
                        className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                        placeholder="75"
                      />
                      <span className="absolute right-2 top-2 text-[10px] text-slate-400">kg</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Clinical Notes &amp; Observations
                </label>
                <textarea
                  rows={2}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  placeholder="Record provider observations, symptoms, or referral indications..."
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full btn-healthcare-primary py-2.5 flex items-center justify-center gap-2 text-sm disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Persisting Screening Encounter...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Submit &amp; Update Clinical Record</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Clinical Context Sidebar */}
          <div className="space-y-4">
            {/* Live BMI Preview Card */}
            <div className="card-healthcare p-5 space-y-3">
              <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Scale className="w-4 h-4 text-forest-700" />
                Live BMI Assessment
              </h3>
              {computedBmi ? (
                <div className="space-y-2">
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-slate-900 font-mono">
                      {computedBmi}
                    </span>
                    <span className="text-xs text-slate-500">kg/m²</span>
                  </div>
                  {(() => {
                    const cat = getBmiCategory(parseFloat(computedBmi));
                    return (
                      <span className={`inline-block px-2.5 py-1 text-xs font-semibold rounded-md border ${cat.color}`}>
                        {cat.label}
                      </span>
                    );
                  })()}
                  <p className="text-[11px] text-slate-500 leading-relaxed mt-2">
                    WHO / DoH classification. Triggers metabolic and lifestyle guidance if &ge; 25.0.
                  </p>
                </div>
              ) : (
                <p className="text-xs text-slate-400">
                  Enter height and weight to calculate BMI in real time.
                </p>
              )}
            </div>

            {/* ADPHC / DoH Pathway Reference */}
            <div className="card-healthcare p-5 space-y-3">
              <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-forest-700" />
                Clinical Guidance Rules
              </h3>
              <ul className="text-xs text-slate-600 space-y-2 leading-relaxed">
                <li className="flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-forest-700 mt-1.5 flex-shrink-0" />
                  <span><strong>Adult IFHAS:</strong> Annual fasting glucose and BP screen for adults aged 18–64.</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 mt-1.5 flex-shrink-0" />
                  <span><strong>Well-Child:</strong> Growth charting, milestone check, and vaccination tracking.</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-teal-600 mt-1.5 flex-shrink-0" />
                  <span><strong>Epistemic Integrity:</strong> All measurements are tagged with data provenance.</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
