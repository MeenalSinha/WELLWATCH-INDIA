'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { apiClient, ApiError } from '@/lib/api';
import { CarePlanResponse, PersonSummary, PreventiveRecommendationResponse } from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { StatusBadge } from '@/components/StatusBadge';
import { MetricCard } from '@/components/MetricCard';
import { EmptyState } from '@/components/EmptyState';
import {
  HeartHandshake,
  Target,
  CheckCircle2,
  Calendar,
  Plus,
  Users,
  ShieldCheck,
  X,
  Layers,
  AlertCircle,
} from 'lucide-react';

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
    <AppShell activePage="/care-plans">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Patient Care', 'Care Plans']}
          title="Preventive Care Plans"
          subtitle="Structured life-course preventive plans linking clinical goals, multidisciplinary care teams, and evidence-backed orders."
          action={{
            label: 'New Care Plan',
            icon: Plus,
            onClick: () => setShowCreateModal(true),
          }}
        />

        {/* Resident Selector */}
        <div className="card-healthcare p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex-shrink-0">
              Resident Target:
            </span>
            <select
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none w-full sm:w-96"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.full_name} ({p.population_classification} &bull; {p.age_years} yrs)
                </option>
              ))}
            </select>
          </div>
          <span className="text-xs text-slate-500">
            Active Plans: <strong className="text-forest-900">{carePlans.length}</strong>
          </span>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Plans List */}
        {isLoading ? (
          <div className="card-healthcare p-12 text-center text-slate-400">
            <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <span>Loading authorized care plans...</span>
          </div>
        ) : carePlans.length === 0 ? (
          <div className="card-healthcare p-8">
            <EmptyState
              title="No Active Care Plans"
              description="No preventive care plans have been established for this resident yet. Establish one to structure goals and care team assignments."
              action={{
                label: 'Create Preventive Care Plan',
                onClick: () => setShowCreateModal(true),
              }}
            />
          </div>
        ) : (
          <div className="space-y-4">
            {carePlans.map((plan) => (
              <div key={plan.id} className="card-healthcare p-6 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-4 gap-2">
                  <div>
                    <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                      <HeartHandshake className="w-4 h-4 text-forest-700" />
                      {plan.plan_name}
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Pathway: <span className="font-mono text-slate-800 font-medium">{plan.pathway_id}</span> &bull; Cadence:{' '}
                      <span className="font-semibold text-slate-700">{plan.review_schedule || 'EVERY_6_MONTHS'}</span>
                    </p>
                  </div>
                  <span className="self-start sm:self-center px-3 py-1 bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold rounded-full">
                    {plan.status}
                  </span>
                </div>

                {/* Goals */}
                <div>
                  <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5 text-forest-700" />
                    Preventive Wellness Goals
                  </h4>
                  <div className="space-y-2">
                    {plan.goals.map((g, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-2.5 text-xs text-slate-800 bg-slate-50 p-3 rounded-xl border border-slate-200"
                      >
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                        <span className="font-medium">{g.goal}</span>
                        {g.target && <span className="text-slate-500 font-mono">({g.target})</span>}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Linked Items */}
                <div>
                  <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-forest-700" />
                    Linked Preventive Recommendations ({plan.items.length})
                  </h4>
                  <div className="divide-y divide-slate-200 border border-slate-200 rounded-xl overflow-hidden bg-white">
                    {plan.items.map((item) => (
                      <div
                        key={item.id}
                        className="p-3 text-xs flex items-center justify-between hover:bg-slate-50 transition"
                      >
                        <div>
                          <p className="font-semibold text-slate-900">
                            {item.recommendation?.recommended_action || `Recommendation ID: ${item.recommendation_id}`}
                          </p>
                          <p className="text-[11px] text-slate-500 mt-0.5">
                            Status: <span className="font-medium text-forest-800">{item.status}</span>
                          </p>
                        </div>
                        <span className="font-mono text-slate-500 text-[11px]">
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
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-2xl max-w-xl w-full p-6 space-y-4 shadow-2xl border border-slate-200">
              <div className="flex justify-between items-center border-b border-slate-200 pb-3">
                <h3 className="text-base font-bold text-slate-900">Establish Preventive Care Plan</h3>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="p-1 rounded text-slate-400 hover:text-slate-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Care Plan Title *</label>
                  <input
                    type="text"
                    value={planName}
                    onChange={(e) => setPlanName(e.target.value)}
                    placeholder="e.g. Well-Child Comprehensive Plan 2026"
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Clinical Protocol</label>
                  <select
                    value={pathwayId}
                    onChange={(e) => setPathwayId(e.target.value)}
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  >
                    <option value="pediatric_well_child">ADPHC Pediatric Well-Child Standard</option>
                    <option value="adphc_ifhas">ADPHC IFHAS Adult Comprehensive Screening</option>
                    <option value="genomic_secondary_findings">Emirati Genome Program Genomic Findings</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Primary Clinical Goal</label>
                  <input
                    type="text"
                    value={goalText}
                    onChange={(e) => setGoalText(e.target.value)}
                    placeholder="e.g. Complete developmental surveillance and vaccination schedule"
                    className="w-full rounded-lg border-slate-300 text-xs px-3 py-2 focus:ring-2 focus:ring-forest-600 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">
                    Link Active Recommendations
                  </label>
                  <div className="max-h-36 overflow-y-auto space-y-1.5 border border-slate-200 p-2.5 rounded-xl bg-slate-50">
                    {availableRecs.length === 0 ? (
                      <p className="text-slate-400 italic">No recommendations available for this resident.</p>
                    ) : (
                      availableRecs.map((r) => (
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
                            className="rounded text-forest-700 focus:ring-forest-600"
                          />
                          <span>{r.recommended_action}</span>
                        </label>
                      ))
                    )}
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-200">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn-healthcare-secondary text-xs px-4 py-2"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleCreateCarePlan}
                  disabled={isSubmitting}
                  className="btn-healthcare-primary text-xs px-4 py-2 disabled:opacity-50"
                >
                  {isSubmitting ? 'Creating...' : 'Establish Care Plan'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

export default function CarePlansPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading care plans...</span>
        </div>
      }
    >
      <CarePlansContent />
    </Suspense>
  );
}
