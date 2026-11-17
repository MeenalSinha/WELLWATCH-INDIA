'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient, ApiError } from '@/lib/api';
import { PersonCreate, PersonSummary, PopulationClassification } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';
import { getStoredUser } from '@/lib/auth';

export default function PatientsPage() {
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [classification, setClassification] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal State
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

  const fetchPatients = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.listPatients(
        classification || undefined,
        searchQuery || undefined,
        0,
        50
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
    fetchPatients();
  }, [classification]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchPatients();
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="sm:flex sm:items-center sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Patient Intelligence Directory</h1>
          <p className="mt-1 text-sm text-slate-500">
            Life-course longitudinal health records, population stratification, and clinical profile completeness for Abu Dhabi residents.
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button
            onClick={() => setShowRegisterModal(true)}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-emerald-800 hover:bg-emerald-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-700 transition"
          >
            + Register New Patient
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row gap-4 justify-between items-center">
        <form onSubmit={handleSearchSubmit} className="flex-1 flex gap-2 w-full md:w-auto">
          <input
            type="text"
            placeholder="Search by patient name, Emirates ID, or district..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700 focus:border-transparent"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-slate-800 text-white text-sm font-medium rounded-lg hover:bg-slate-900 transition"
          >
            Search
          </button>
        </form>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Classification:</span>
          <select
            value={classification}
            onChange={(e) => setClassification(e.target.value)}
            className="px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-700 bg-white"
          >
            <option value="">All Populations ({total})</option>
            <option value="PEDIATRIC">Pediatric (0-17 yrs)</option>
            <option value="ADULT">Adult (18-64 yrs)</option>
            <option value="GERIATRIC">Geriatric (65+ yrs)</option>
          </select>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 text-sm p-4 rounded-lg flex items-center justify-between">
          <span>{error}</span>
          <button onClick={fetchPatients} className="underline font-semibold ml-2">Retry</button>
        </div>
      )}

      {/* Patients Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-slate-400">Loading authorized patient records...</div>
        ) : patients.length === 0 ? (
          <div className="p-12 text-center text-slate-500">
            No patient records found matching the active query within your facility boundary.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50 text-slate-600 font-semibold text-left">
                <tr>
                  <th className="px-6 py-3">Patient Name</th>
                  <th className="px-6 py-3">Classification</th>
                  <th className="px-6 py-3">Age / Gender</th>
                  <th className="px-6 py-3">Emirate / District</th>
                  <th className="px-6 py-3">Profile Completeness</th>
                  <th className="px-6 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-slate-700">
                {patients.map((p) => (
                  <tr key={p.id} className="hover:bg-slate-50 transition">
                    <td className="px-6 py-4 font-medium text-slate-900">
                      <Link href={`/patients/${p.id}`} className="hover:text-emerald-800 font-semibold">
                        {p.full_name}
                      </Link>
                      <span className="block text-xs font-mono text-slate-400">DOB: {p.date_of_birth}</span>
                    </td>
                    <td className="px-6 py-4">
                      <StatusBadge status={p.population_classification} />
                    </td>
                    <td className="px-6 py-4">
                      {p.age_years} yrs ({p.age_months} mos) &bull; {p.gender}
                    </td>
                    <td className="px-6 py-4">
                      {p.emirate} {p.district ? `(${p.district})` : ''}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-slate-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              p.completeness_score >= 0.8
                                ? 'bg-emerald-500'
                                : p.completeness_score >= 0.5
                                ? 'bg-amber-500'
                                : 'bg-rose-500'
                            }`}
                            style={{ width: `${Math.round(p.completeness_score * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-slate-500">
                          {Math.round(p.completeness_score * 100)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link
                        href={`/patients/${p.id}`}
                        className="inline-flex items-center px-3 py-1.5 border border-slate-300 rounded-md text-xs font-medium text-slate-700 bg-white hover:bg-slate-100 transition"
                      >
                        Open Chart &rarr;
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Registration Modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900 bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl max-w-lg w-full p-6 shadow-xl border border-slate-200">
            <div className="flex justify-between items-center border-b border-slate-200 pb-3 mb-4">
              <h2 className="text-lg font-bold text-slate-900">Register New Patient</h2>
              <button
                onClick={() => setShowRegisterModal(false)}
                className="text-slate-400 hover:text-slate-600 font-bold"
              >
                &times;
              </button>
            </div>

            {registerError && (
              <div className="mb-4 p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-lg">
                {registerError}
              </div>
            )}

            <form onSubmit={handleRegisterSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-700">First Name *</label>
                  <input
                    type="text"
                    required
                    value={newPatient.first_name}
                    onChange={(e) => setNewPatient({ ...newPatient, first_name: e.target.value })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-700">Last Name *</label>
                  <input
                    type="text"
                    required
                    value={newPatient.last_name}
                    onChange={(e) => setNewPatient({ ...newPatient, last_name: e.target.value })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-700">Date of Birth *</label>
                  <input
                    type="date"
                    required
                    value={newPatient.date_of_birth}
                    onChange={(e) => setNewPatient({ ...newPatient, date_of_birth: e.target.value })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-700">Gender *</label>
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value as any })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700 bg-white"
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
                  <label className="block text-xs font-medium text-slate-700">Emirates ID (784-XXXX-XXXXXXX-X)</label>
                  <input
                    type="text"
                    placeholder="784-1990-1234567-1"
                    value={newPatient.identifier_value || ''}
                    onChange={(e) => setNewPatient({ ...newPatient, identifier_value: e.target.value })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-700">District / Region</label>
                  <input
                    type="text"
                    placeholder="e.g. Al Bateen, Al Ain"
                    value={newPatient.district || ''}
                    onChange={(e) => setNewPatient({ ...newPatient, district: e.target.value })}
                    className="mt-1 w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-emerald-700 focus:border-emerald-700"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  type="button"
                  onClick={() => setShowRegisterModal(false)}
                  className="px-4 py-2 border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-4 py-2 bg-emerald-800 text-white rounded-lg text-sm font-medium hover:bg-emerald-900 disabled:opacity-50"
                >
                  {submitting ? 'Registering...' : 'Complete Enrolment'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
