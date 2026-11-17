'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { CarePlanResponse, PersonSummary, PreventiveRecommendationResponse } from '@/types';

function CarePlansContent() {
  const searchParams = useSearchParams();
  const initialPatientId = searchParams.get('patientId') || '';

  const [patientId, setPatientId] = useState(initialPatientId);
  const [carePlans, setCarePlans] = useState<CarePlanResponse[]>([]);
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [availableRecs, setAvailableRecs] = useState<PreventiveRecommendationResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [planName, setPlanName] = useState('');
  const [pathwayId, setPathwayId] = useState('pediatric_well_child');
  const [goalText, setGoalText] = useState('');
  const [selectedRecIds, setSelectedRecIds] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    loadPatients();
  }, []);

  useEffect(() => {
    if (patientId) {
      loadCarePlans(patientId);
      loadPatientRecs(patientId);
    } else {
      setCarePlans([]);
      setIsLoading(false);
    }
  }, [patientId]);

  async function loadPatients() {
    try {
      const res = await apiClient.getPatients(1, 100);
      setPatients(res.items);
      if (!patientId && res.items.length > 0) {
        setPatientId(res.items[0].id);
      }
    } catch (err: any) {
      console.error('Failed to load patients', err);
    }
  }

  async function loadCarePlans(pId: string) {
    setIsLoading(true);
    setError(null);
    try {
      const plans = await apiClient.getPatientCarePlans(pId);
      setCarePlans(plans);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to fetch care plans');
    } finally {
      setIsLoading(false);
    }
  }

  async function loadPatientRecs(pId: string) {
    try {
      const res = await apiClient.getRecommendations(pId);
      setAvailableRecs(res.items);
    } catch (err: any) {
      console.error('Failed to load patient recommendations', err);
    }
  }

  async function handleCreateCarePlan() {
    if (!planName.trim()) {
      alert('Please provide a care plan name.');
      return;
    }
    setIsSubmitting(true);
    try {
      await apiClient.createCarePlan({
        person_id: patientId,
        plan_name: planName,
        pathway_id: pathwayId,
        goals: [{ goal: goalText || 'Periodic developmental and preventive wellness' }],
        assigned_care_team: [{ role: 'Care Coordinator', name: 'Al Bateen Clinic Coordinator' }],
        recommendation_ids: selectedRecIds,
        review_schedule: 'EVERY_6_MONTHS',
      });
      setShowCreateModal(false);
      setPlanName('');
      setGoalText('');
      setSelectedRecIds([]);
      loadCarePlans(patientId);
    } catch (err: any) {
      alert('Failed to create care plan: ' + err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Preventive Care Plans</h1>
          <p className="text-sm text-slate-600 mt-1">
            Structured Life-Course Preventive Plans linking goals, clinical teams, and recommendations
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="rounded-lg border-slate-300 text-sm py-2 px-3"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.full_name} ({p.population_classification})
              </option>
            ))}
          </select>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 shadow-sm"
          >
            + New Care Plan
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Plans List */}
      {isLoading ? (
        <div className="text-center py-12 text-slate-500">Loading care plans...</div>
      ) : carePlans.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-xl border border-slate-200 text-slate-500">
          No preventive care plans found for this patient. Click "+ New Care Plan" to establish one.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {carePlans.map((plan) => (
            <div key={plan.id} className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
              <div className="flex justify-between items-start border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-lg font-bold text-slate-900">{plan.plan_name}</h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Pathway: <span className="font-mono text-slate-700">{plan.pathway_id}</span> | Review Schedule:{' '}
                    <span className="font-medium text-slate-700">{plan.review_schedule || 'EVERY_6_MONTHS'}</span>
                  </p>
                </div>
                <span className="px-3 py-1 bg-emerald-100 text-emerald-800 text-xs font-semibold rounded-full">
                  {plan.status}
                </span>
              </div>

              {/* Goals */}
              <div>
                <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">Preventive Goals</h4>
                <div className="space-y-1">
                  {plan.goals.map((g, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm text-slate-800 bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                      <span className="text-emerald-500">🎯</span>
                      <span>{g.goal}</span>
                      {g.target && <span className="text-xs text-slate-500">({g.target})</span>}
                    </div>
                  ))}
                </div>
              </div>

              {/* Linked Items */}
              <div>
                <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                  Linked Preventive Recommendations ({plan.items.length})
                </h4>
                <div className="divide-y divide-slate-100 border border-slate-100 rounded-lg overflow-hidden">
                  {plan.items.map((item) => (
                    <div key={item.id} className="p-3 text-sm flex items-center justify-between bg-white hover:bg-slate-50">
                      <div>
                        <p className="font-medium text-slate-900">
                          {item.recommendation?.recommended_action || `Recommendation ID: ${item.recommendation_id}`}
                        </p>
                        <p className="text-xs text-slate-500">
                          Status: <span className="font-medium text-slate-700">{item.status}</span>
                        </p>
                      </div>
                      <span className="text-xs font-medium text-slate-500">
                        Target: {item.target_date ? new Date(item.target_date).toLocaleDateString() : 'Active'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-slate-900/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-900">Create Preventive Care Plan</h3>
            <div className="space-y-3 text-sm">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Plan Title</label>
                <input
                  type="text"
                  value={planName}
                  onChange={(e) => setPlanName(e.target.value)}
                  placeholder="e.g. Well-Child Comprehensive Plan 2026"
                  className="w-full rounded-lg border-slate-300 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Pathway</label>
                <select
                  value={pathwayId}
                  onChange={(e) => setPathwayId(e.target.value)}
                  className="w-full rounded-lg border-slate-300 text-sm"
                >
                  <option value="pediatric_well_child">ADPHC Pediatric Well-Child Standard</option>
                  <option value="adphc_ifhas">ADPHC IFHAS Adult Comprehensive Screening</option>
                  <option value="genomic_secondary_findings">Emirati Genome Program Genomic Findings</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Primary Goal</label>
                <input
                  type="text"
                  value={goalText}
                  onChange={(e) => setGoalText(e.target.value)}
                  placeholder="e.g. Complete developmental surveillance and vaccination schedule"
                  className="w-full rounded-lg border-slate-300 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Link Active Recommendations</label>
                <div className="max-h-36 overflow-y-auto space-y-1.5 border border-slate-200 p-2 rounded-lg">
                  {availableRecs.map((r) => (
                    <label key={r.id} className="flex items-center gap-2 text-xs text-slate-800">
                      <input
                        type="checkbox"
                        checked={selectedRecIds.includes(r.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedRecIds([...selectedRecIds, r.id]);
                          } else {
                            setSelectedRecIds(selectedRecIds.filter((id) => id !== r.id));
                          }
                        }}
                        className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                      />
                      <span>{r.recommended_action}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateCarePlan}
                disabled={isSubmitting}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {isSubmitting ? 'Creating...' : 'Create Care Plan'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function CarePlansPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Care Plans...</div>}>
      <CarePlansContent />
    </Suspense>
  );
}
