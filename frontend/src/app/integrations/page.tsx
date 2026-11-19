'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { apiClient, ApiError } from '@/lib/api';
import {
  IntegrationProviderResponse,
  IntegrationMonitoringResponse,
  FhirExportResponse,
  FhirIngestResponse,
  TerminologyLookupResponse,
  PersonSummary,
} from '@/types';
import { AppShell } from '@/components/AppShell';
import { PageHeader } from '@/components/PageHeader';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { EmptyState } from '@/components/EmptyState';
import {
  Network,
  Share2,
  FileCode,
  Download,
  Upload,
  BookOpen,
  ShieldCheck,
  ShieldAlert,
  Activity,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Search,
} from 'lucide-react';

function IntegrationsContent() {
  const [activeTab, setActiveTab] = useState<'adapters' | 'ingest' | 'export' | 'terminology' | 'monitoring'>('adapters');

  // Provider adapters & monitoring
  const [providers, setProviders] = useState<IntegrationProviderResponse[]>([]);
  const [monitoring, setMonitoring] = useState<IntegrationMonitoringResponse | null>(null);

  // Ingestion state
  const [sourceSystem, setSourceSystem] = useState<string>('MALAFFI');
  const [fhirPayload, setFhirPayload] = useState<string>(
    JSON.stringify(
      {
        resourceType: 'Bundle',
        type: 'transaction',
        entry: [
          {
            resource: {
              resourceType: 'Patient',
              identifier: [
                {
                  system: 'http://fhir.doh.gov.ae/identifier/emirates-id',
                  value: '784-1988-1234567-1',
                },
              ],
            },
          },
          {
            resource: {
              resourceType: 'Observation',
              status: 'final',
              code: {
                coding: [
                  {
                    system: 'http://loinc.org',
                    code: '15074-8',
                    display: 'Glucose [Mass/volume] in Blood',
                  },
                ],
              },
              valueQuantity: {
                value: 94.0,
                unit: 'mg/dL',
                system: 'http://unitsofmeasure.org',
                code: 'mg/dL',
              },
            },
          },
        ],
      },
      null,
      2
    )
  );
  const [ingestResult, setIngestResult] = useState<FhirIngestResponse | null>(null);

  // Export state
  const [patients, setPatients] = useState<PersonSummary[]>([]);
  const [selectedExportPatientId, setSelectedExportPatientId] = useState<string>('');
  const [exportResult, setExportResult] = useState<FhirExportResponse | null>(null);

  // Terminology lookup
  const [lookupSystem, setLookupSystem] = useState<string>('LOINC');
  const [lookupCode, setLookupCode] = useState<string>('15074-8');
  const [termResult, setTermResult] = useState<TerminologyLookupResponse | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    loadProvidersAndMonitoring();
    loadPatients();
  }, []);

  const loadProvidersAndMonitoring = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [provList, monData] = await Promise.all([
        apiClient.getIntegrationProviders(),
        apiClient.getIntegrationMonitoring(),
      ]);
      setProviders(provList);
      setMonitoring(monData);
    } catch (err: any) {
      console.error('Failed to load integration providers:', err);
      setError(err instanceof ApiError ? err.message : 'Unable to connect to Interoperability service.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadPatients = async () => {
    try {
      const pList = await apiClient.listPatients(undefined, undefined, 0, 50);
      setPatients(pList.items || []);
      if (pList.items && pList.items.length > 0) {
        setSelectedExportPatientId(pList.items[0].id);
      }
    } catch (err) {
      console.warn('Patients load note:', err);
    }
  };

  const handleIngestBundle = async () => {
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    setIngestResult(null);

    try {
      const parsedBundle = JSON.parse(fhirPayload);
      const res = await apiClient.ingestFhirBundle({
        bundle: parsedBundle,
        source_system: sourceSystem,
      });
      setIngestResult(res);
      setSuccessMessage(
        `FHIR Bundle ingested: ${res.resources_ingested} resources processed. Cryptographic provenance hash recorded.`
      );
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Invalid JSON or FHIR ingestion rejected.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportPatient = async () => {
    if (!selectedExportPatientId) {
      setError('Please select a resident to export.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setExportResult(null);

    try {
      const res = await apiClient.exportFhirPatient(selectedExportPatientId);
      setExportResult(res);
      setSuccessMessage('Resident FHIR R4 Bundle exported successfully.');
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to export patient bundle.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLookupTerminology = async () => {
    setError(null);
    try {
      const res = await apiClient.lookupTerminology(lookupSystem, lookupCode);
      setTermResult(res);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Code not found in governed terminology dictionary.');
    }
  };

  return (
    <AppShell activePage="/integrations">
      <div className="space-y-6">
        <PageHeader
          breadcrumbs={['Platform & Governance', 'Interoperability']}
          title="Healthcare Interoperability Hub"
          subtitle="Standardized health data exchange with Malaffi HIE, SEHA EMR, diagnostic laboratories, and UAE Pass under HL7 FHIR R4 specifications."
          action={{
            label: 'Refresh Adapters',
            icon: RefreshCw,
            onClick: loadProvidersAndMonitoring,
          }}
        />

        {/* Honest Boundary Notice */}
        <div className="p-4 rounded-xl bg-forest-50/70 border border-forest-200 text-xs text-forest-950 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-forest-700 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold block text-forest-900">
              Honest External Integration Boundary &bull; DoH / ADHICS Compliance
            </span>
            <p className="text-slate-700 leading-relaxed">
              HL7 FHIR R4 serialization, identity reconciliation, LOINC/SNOMED CT mapping, and cryptographic provenance are{' '}
              <strong>fully functional</strong>. Live external network connections to Malaffi, SEHA, or UAE Pass production gateways are strictly reported as{' '}
              <strong>NOT_CONFIGURED</strong> until valid DoH-issued mTLS certificates and credentials are provisioned. Zero synthetic connection claims are permitted.
            </p>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="text-xs font-semibold underline">
              Dismiss
            </button>
          </div>
        )}

        {successMessage && (
          <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <span>{successMessage}</span>
            </div>
            <button onClick={() => setSuccessMessage(null)} className="text-xs font-semibold underline">
              Dismiss
            </button>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-200 space-x-6 text-xs font-semibold overflow-x-auto pb-1">
          {[
            { id: 'adapters', label: `Integration Adapters (${providers.length})` },
            { id: 'ingest', label: 'FHIR R4 Ingestion Sandbox' },
            { id: 'export', label: 'FHIR R4 Patient Export' },
            { id: 'terminology', label: 'Governed Medical Terminology' },
            { id: 'monitoring', label: 'Job Telemetry & Provenance' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`pb-3 font-semibold border-b-2 transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'border-forest-800 text-forest-900'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: ADAPTERS */}
        {activeTab === 'adapters' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {providers.map((p) => {
              const isConfigured = p.status === 'CONFIGURED' || p.status === 'ACTIVE';
              return (
                <div key={p.provider_name} className="card-healthcare p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-slate-900 text-sm">{p.display_name}</h3>
                      <span className="text-[11px] font-mono text-slate-500">{p.provider_type}</span>
                    </div>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded text-[11px] font-bold border ${
                        isConfigured
                          ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                          : 'bg-amber-50 text-amber-800 border-amber-300'
                      }`}
                    >
                      {p.status}
                    </span>
                  </div>

                  <div className="text-xs text-slate-600 space-y-1 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                    <div>
                      Endpoint:{' '}
                      <span className="font-mono text-slate-800">{p.endpoint_url || 'Not provisioned'}</span>
                    </div>
                    <div>
                      Enabled in Configuration:{' '}
                      <span className="font-semibold text-slate-800">{p.is_enabled ? 'Yes' : 'No'}</span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-100">
                    <span className="text-[11px] font-semibold text-slate-700 block mb-1">
                      Required Production Credentials:
                    </span>
                    <ul className="list-disc list-inside text-[11px] text-amber-900 space-y-0.5 font-mono">
                      {p.prerequisites.map((prereq, idx) => (
                        <li key={idx}>{prereq}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* TAB 2: INGESTION SANDBOX */}
        {activeTab === 'ingest' && (
          <div className="card-healthcare p-6 space-y-4 max-w-4xl">
            <div>
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Upload className="w-4 h-4 text-forest-700" />
                HL7 FHIR R4 Bundle Ingestion Sandbox
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Process an incoming FHIR R4 transaction bundle from external hospital laboratories or Malaffi reciprocal transfers.
                Resolves patient identity, maps LOINC codes, and generates a SHA-256 cryptographic provenance audit record.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Source System Gateway
              </label>
              <select
                value={sourceSystem}
                onChange={(e) => setSourceSystem(e.target.value)}
                className="w-full md:w-64 rounded-lg border border-slate-300 p-2 text-xs bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
              >
                <option value="MALAFFI">Malaffi HIE</option>
                <option value="SEHA_EMR">SEHA Clinic / Hospital EMR</option>
                <option value="LAB_NETWORK">National Diagnostic Lab Network</option>
                <option value="EXTERNAL_CLINIC">External Primary Care Clinic</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                HL7 FHIR R4 Bundle Payload (JSON)
              </label>
              <textarea
                rows={12}
                value={fhirPayload}
                onChange={(e) => setFhirPayload(e.target.value)}
                className="w-full font-mono text-xs rounded-xl border border-slate-300 p-3 bg-slate-900 text-slate-100 focus:outline-none"
              />
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleIngestBundle}
                disabled={isLoading}
                className="btn-healthcare-primary text-xs py-2.5 px-5 flex items-center gap-1.5 disabled:opacity-50"
              >
                <Upload className="w-4 h-4" />
                <span>{isLoading ? 'Ingesting & Validating...' : 'Ingest FHIR Bundle with Provenance'}</span>
              </button>
            </div>

            {ingestResult && (
              <div className="mt-4 border-t border-slate-200 pt-4">
                <h4 className="text-xs font-bold text-forest-900 uppercase tracking-wider mb-2">
                  Ingestion Pipeline Telemetry Result
                </h4>
                <pre className="bg-slate-900 text-emerald-300 p-4 rounded-xl text-xs overflow-x-auto font-mono">
                  {JSON.stringify(ingestResult, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: EXPORT */}
        {activeTab === 'export' && (
          <div className="card-healthcare p-6 space-y-4 max-w-4xl">
            <div>
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Download className="w-4 h-4 text-forest-700" />
                Export Resident as HL7 FHIR R4 Bundle
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Exports patient profile, clinical observations, encounters, and preventive screening records as a standard FHIR R4 Bundle for external care transitions.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Target Resident
                </label>
                <select
                  value={selectedExportPatientId}
                  onChange={(e) => setSelectedExportPatientId(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 p-2 text-xs bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                >
                  {patients.map((pat) => (
                    <option key={pat.id} value={pat.id}>
                      {pat.full_name} ({pat.emirate || 'Abu Dhabi'})
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-end">
                <button
                  onClick={handleExportPatient}
                  disabled={isLoading}
                  className="w-full btn-healthcare-primary text-xs py-2 px-4 flex items-center justify-center gap-1.5 disabled:opacity-50"
                >
                  <Download className="w-4 h-4" />
                  <span>{isLoading ? 'Serializing Bundle...' : 'Export Standard FHIR R4 Bundle'}</span>
                </button>
              </div>
            </div>

            {exportResult && (
              <div className="mt-4 border-t border-slate-200 pt-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Serialized FHIR R4 Bundle Payload
                  </h4>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Exported: {new Date(exportResult.exported_at).toLocaleString()}
                  </span>
                </div>
                <pre className="bg-slate-900 text-slate-100 p-4 rounded-xl text-xs overflow-x-auto max-h-96 font-mono">
                  {JSON.stringify(exportResult.bundle, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: TERMINOLOGY LOOKUP */}
        {activeTab === 'terminology' && (
          <div className="card-healthcare p-6 space-y-4 max-w-2xl">
            <div>
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-forest-700" />
                Governed Medical Terminology Mapper
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Inspect canonical mappings between WELLWATCH internal clinical concepts and standard vocabularies (LOINC, SNOMED CT, ICD-10, UCUM).
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Coding Standard</label>
                <select
                  value={lookupSystem}
                  onChange={(e) => setLookupSystem(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 p-2 text-xs bg-white font-medium focus:ring-2 focus:ring-forest-600 focus:outline-none"
                >
                  <option value="LOINC">LOINC (Observations &amp; Labs)</option>
                  <option value="SNOMED_CT">SNOMED CT (Clinical Encounters)</option>
                  <option value="ICD_10">ICD-10 (Conditions &amp; Diagnoses)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Standard Code</label>
                <input
                  type="text"
                  value={lookupCode}
                  onChange={(e) => setLookupCode(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 p-2 text-xs font-mono focus:ring-2 focus:ring-forest-600 focus:outline-none"
                />
              </div>
            </div>

            <div className="flex justify-end pt-1">
              <button
                onClick={handleLookupTerminology}
                className="btn-healthcare-primary text-xs py-2 px-4 flex items-center gap-1.5"
              >
                <Search className="w-3.5 h-3.5" />
                <span>Query Terminology Directory</span>
              </button>
            </div>

            {termResult && (
              <div className="mt-4 border-t border-slate-200 pt-4">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
                  Canonical Vocabulary Mapping
                </h4>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs space-y-2">
                  <div>
                    Internal Code:{' '}
                    <strong className="font-mono text-slate-800">{termResult.internal_code}</strong>
                  </div>
                  <div>
                    Display Name: <strong className="text-slate-800">{termResult.display_name}</strong>
                  </div>
                  <div>
                    Target Standard:{' '}
                    <span className="font-mono bg-white px-2 py-0.5 rounded border border-slate-200 text-forest-800">
                      {termResult.target_system}: {termResult.target_code}
                    </span>
                  </div>
                  <div>
                    Target Display: <span>{termResult.target_display}</span>
                  </div>
                  <div>
                    Canonical UCUM Unit:{' '}
                    <span className="font-mono text-slate-700">{termResult.ucum_unit || 'None'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 5: MONITORING & PROVENANCE */}
        {activeTab === 'monitoring' && monitoring && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard
                title="Total Integration Adapters"
                value={monitoring.total_providers}
                icon={Network}
                subtitle="Configured gateway endpoints"
                variant="forest"
              />
              <MetricCard
                title="Active in Production"
                value={monitoring.configured_providers}
                icon={CheckCircle2}
                subtitle="Live mTLS verified"
                variant="emerald"
              />
              <MetricCard
                title="Pending Credentials"
                value={monitoring.not_configured_providers}
                icon={ShieldAlert}
                subtitle="Honest boundary enforced"
                variant="amber"
              />
            </div>

            <div className="card-healthcare p-6 space-y-3">
              <h4 className="text-sm font-bold text-slate-900">Recent Integration Execution Logs</h4>
              {monitoring.recent_jobs && monitoring.recent_jobs.length > 0 ? (
                <div className="space-y-2">
                  {monitoring.recent_jobs.map((job, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 flex justify-between items-center text-xs"
                    >
                      <div>
                        <span className="font-semibold text-slate-900">{job.provider_name}</span> &bull;{' '}
                        {job.job_type}
                        <div className="text-[11px] text-slate-400 font-mono mt-0.5">{job.status}</div>
                      </div>
                      <span className="text-slate-400 font-mono">
                        {new Date(job.executed_at).toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center text-slate-400 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                  No historical integration jobs recorded in this partition session.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

export default function IntegrationsPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-slate-400">
          <div className="w-6 h-6 border-2 border-forest-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <span>Loading interoperability center...</span>
        </div>
      }
    >
      <IntegrationsContent />
    </Suspense>
  );
}
