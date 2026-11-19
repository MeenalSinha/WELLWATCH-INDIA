'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/lib/api';
import { PersonCreate, PersonSummary, PopulationClassification } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { SearchField } from '@/components/SearchField';
import { EmptyState } from '@/components/EmptyState';
import { getStoredUser } from '@/lib/auth';
import {
  Users,
  Baby,
  Activity,
  HeartPulse,
  UserPlus,
  ArrowRight,
  Filter,
  ShieldAlert,
  MapPin,
  Calendar,
  X,
} from 'lucide-react';

export default function PatientsPage() {
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [classification, setClassification] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Registration Modal State
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [newPatient, setNewPatient] = useState<PersonCreate>({
    first_name: '',
    last_name: '',
    date_of_birth: '',
    gender: 'FEMALE',
    emirate: 'Abu Dhabi',
    district: '',
    preferred_language: 'ar',
    nationality: 'United Arab Emirates',
    identifier_namespace: 'EMIRATES_ID',
    identifier_value: '',
  });

  const currentUser = getStoredUser();

  const fetchPatients = async (query = searchQuery, cls = classification) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.listPatients(
        cls || undefined,
        query || undefined,
        0,
        100
      );
      setPatients(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load patient directory.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients(searchQuery, classification);
  }, [classification]);

  const handleSearch = (term: string) => {
    setSearchQuery(term);
    fetchPatients(term, classification);
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setRegisterError(null);

    try {
      await apiClient.registerPatient(newPatient);
      setShowRegisterModal(false);
      setNewPatient({
        first_name: '',
        last_name: '',
        date_of_birth: '',
        gender: 'FEMALE',
        emirate: 'Abu Dhabi',
        district: '',
        preferred_language: 'ar',
        nationality: 'United Arab Emirates',
        identifier_namespace: 'EMIRATES_ID',
        identifier_value: '',
      });
      await fetchPatients();
    } catch (err: any) {
      setRegisterError(err.message || 'Failed to register patient.');
    } finally {
      setSubmitting(false);
    }
  };

  // Cohort metric calculations
  const pedCount = patients.filter((p) => p.population_classification === 'PEDIATRIC').length;
  const adultCount = patients.filter((p) => p.population_classification === 'ADULT').length;
  const gerCount = patients.filter((p) => p.population_classification === 'GERIATRIC').length;

  return (
    <AppShell activePage="/patients">
      <div className="space-y-6">
        {/* Page Header */}
        <PageHeader
          breadcrumbs={['Patients', 'Directory']}
          title="Patient Intelligence Directory"
          subtitle="Longitudinal life-course records, demographic stratification, and clinical profile completeness for Abu Dhabi residents."
          action={{
            label: 'Register Resident',
            icon: UserPlus,
            onClick: () => setShowRegisterModal(true),
          }}
        />

        {/* Cohort Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Total Roster"
            value={total}
            icon={Users}
            subtitle="Registered individuals"
            variant="forest"
          />
          <MetricCard
            title="Pediatric Cohort"
            value={pedCount}
            icon={Baby}
            subtitle="0–17 yrs (Well-Child)"
            variant="emerald"
          />
          <MetricCard
            title="Adult IFHAS"
            value={adultCount}
            icon={HeartPulse}
            subtitle="18–64 yrs (Comprehensive)"
            variant="teal"
          />
          <MetricCard
            title="Geriatric"
            value={gerCount}
            icon={Activity}
            subtitle="65+ yrs (Senior Care)"
            variant="amber"
          />
        </div>

        {/* Filters and Search Bar */}
        <div className="card-healthcare p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="w-full md:w-96">
            <SearchField
              placeholder="Search by name, Emirates ID, or district..."
              value={searchQuery}
              onChange={handleSearch}
            />
          </div>

          <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 flex-shrink-0">
              <Filter className="w-3.5 h-3.5 text-forest-700" />
              Life-Course:
            </span>
            <div className="inline-flex rounded-lg p-1 bg-slate-100 border border-slate-200">
              {[
                { label: 'All Cohorts', value: '' },
                { label: 'Pediatric (0–17)', value: 'PEDIATRIC' },
                { label: 'Adult (18–64)', value: 'ADULT' },
                { label: 'Geriatric (65+)', value: 'GERIATRIC' },
              ].map((tab) => (
                <button
                  key={tab.value}
                  onClick={() => setClassification(tab.value)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    classification === tab.value
                      ? 'bg-white text-forest-900 shadow-sm font-semibold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-rose-600" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => fetchPatients()}
              className="text-xs font-semibold underline hover:text-rose-950"
            >
              Retry
            </button>
          </div>
        )}

        {/* Patients Table */}
        <div className="card-healthcare overflow-hidden">
          {loading ? (
            <div className="p-12 text-center text-slate-400">
              <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <span>Loading authorized patient records...</span>
            </div>
          ) : patients.length === 0 ? (
            <div className="p-8">
              <EmptyState
                title="No Resident Records Found"
                description={
                  searchQuery || classification
                    ? 'No patient records match the selected search criteria or life-course classification.'
                    : 'No patients have been enrolled yet in this facility partition.'
                }
                action={{
                  label: 'Register New Patient',
                  onClick: () => setShowRegisterModal(true),
                }}
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-healthcare">
                <thead>
                  <tr>
                    <th>Resident / Patient Name</th>
                    <th>Emirates ID &amp; DOB</th>
                    <th>Life-Course Cohort</th>
                    <th>Demographics</th>
                    <th>District / Emirate</th>
                    <th>Profile Completeness</th>
                    <th className="text-right">Clinical Chart</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p) => {
                    const pct = Math.round(p.completeness_score * 100);
                    return (
                      <tr key={p.id}>
                        <td>
                          <Link
                            href={`/patients/${p.id}`}
                            className="font-semibold text-slate-900 hover:text-forest-700 transition"
                          >
                            {p.full_name}
                          </Link>
                          <span className="block text-xs text-slate-400">
                            Language: {p.preferred_language?.toUpperCase() || 'AR'}
                          </span>
                        </td>
                        <td>
                          <span className="font-mono text-xs text-slate-700 block">
                            {p.primary_identifier_masked || p.identifier_value || '784-••••-•••••••-•'}
                          </span>
                          <span className="text-xs text-slate-400 flex items-center gap-1 mt-0.5">
                            <Calendar className="w-3 h-3 text-slate-400" />
                            {p.date_of_birth}
                          </span>
                        </td>
                        <td>
                          <StatusBadge status={p.population_classification} />
                        </td>
                        <td className="text-xs text-slate-600">
                          <span className="font-medium text-slate-800">{p.age_years} yrs</span>
                          {p.population_classification === 'PEDIATRIC' && (
                            <span className="text-slate-400"> ({p.age_months}m)</span>
                          )}{' '}
                          &bull; {p.gender}
                        </td>
                        <td className="text-xs text-slate-600">
                          <div className="flex items-center gap-1">
                            <MapPin className="w-3 h-3 text-slate-400" />
                            <span>
                              {p.district ? `${p.district}, ` : ''}
                              {p.emirate}
                            </span>
                          </div>
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <div className="w-20 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                              <div
                                className={`h-1.5 rounded-full ${
                                  pct >= 80
                                    ? 'bg-emerald-500'
                                    : pct >= 50
                                    ? 'bg-amber-500'
                                    : 'bg-rose-500'
                                }`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono font-medium text-slate-600">
                              {pct}%
                            </span>
                          </div>
                        </td>
                        <td className="text-right">
                          <Link
                            href={`/patients/${p.id}`}
                            className="btn-healthcare-secondary text-xs py-1.5 px-3 inline-flex items-center gap-1"
                          >
                            <span>Open Profile</span>
                            <ArrowRight className="w-3.5 h-3.5 text-forest-700" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Registration Modal */}
        {showRegisterModal && (
          <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200">
              <div className="flex justify-between items-center border-b border-slate-200 pb-3 mb-4">
                <div>
                  <h2 className="text-base font-bold text-slate-900">Enrol New Abu Dhabi Resident</h2>
                  <p className="text-xs text-slate-500">
                    Initiates life-course longitudinal record &amp; preventive pathway eligibility
                  </p>
                </div>
                <button
                  onClick={() => setShowRegisterModal(false)}
                  className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {registerError && (
                <div className="mb-4 p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 flex-shrink-0" />
                  <span>{registerError}</span>
                </div>
              )}

              <form onSubmit={handleRegisterSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      First Name *
                    </label>
                    <input
                      type="text"
                      required
                      value={newPatient.first_name}
                      onChange={(e) => setNewPatient({ ...newPatient, first_name: e.target.value })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                      placeholder="Fatima"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Last Name *
                    </label>
                    <input
                      type="text"
                      required
                      value={newPatient.last_name}
                      onChange={(e) => setNewPatient({ ...newPatient, last_name: e.target.value })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                      placeholder="Al Zaabi"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Date of Birth *
                    </label>
                    <input
                      type="date"
                      required
                      value={newPatient.date_of_birth}
                      onChange={(e) => setNewPatient({ ...newPatient, date_of_birth: e.target.value })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Gender *
                    </label>
                    <select
                      value={newPatient.gender}
                      onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value as any })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white"
                    >
                      <option value="FEMALE">Female</option>
                      <option value="MALE">Male</option>
                      <option value="OTHER">Other</option>
                      <option value="UNKNOWN">Unknown</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Emirates ID (784-XXXX-XXXXXXX-X)
                    </label>
                    <input
                      type="text"
                      placeholder="784-1992-1234567-1"
                      value={newPatient.identifier_value || ''}
                      onChange={(e) =>
                        setNewPatient({ ...newPatient, identifier_value: e.target.value })
                      }
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none font-mono"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      District / Region
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. Al Bateen, Al Ain"
                      value={newPatient.district || ''}
                      onChange={(e) => setNewPatient({ ...newPatient, district: e.target.value })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Emirate
                    </label>
                    <select
                      value={newPatient.emirate}
                      onChange={(e) => setNewPatient({ ...newPatient, emirate: e.target.value })}
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white"
                    >
                      <option value="Abu Dhabi">Abu Dhabi</option>
                      <option value="Al Ain">Al Ain</option>
                      <option value="Al Dhafra">Al Dhafra</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Preferred Language
                    </label>
                    <select
                      value={newPatient.preferred_language}
                      onChange={(e) =>
                        setNewPatient({ ...newPatient, preferred_language: e.target.value })
                      }
                      className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-forest-600 focus:outline-none bg-white"
                    >
                      <option value="ar">Arabic (العربية)</option>
                      <option value="en">English</option>
                    </select>
                  </div>
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                  <button
                    type="button"
                    onClick={() => setShowRegisterModal(false)}
                    className="btn-healthcare-secondary text-xs px-4 py-2"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="btn-healthcare-primary text-xs px-4 py-2 disabled:opacity-50"
                  >
                    {submitting ? 'Registering...' : 'Complete Enrolment'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
