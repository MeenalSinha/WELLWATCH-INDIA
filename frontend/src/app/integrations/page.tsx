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
        `FHIR Bundle ingested: ${res.resources_ingested} resources processed. Provenance hash recorded.`
      );
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Invalid JSON or FHIR ingestion rejected.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportPatient = async () => {
    if (!selectedExportPatientId) {
      setError('Please select a patient to export.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setExportResult(null);

    try {
      const res = await apiClient.exportFhirPatient(selectedExportPatientId);
      setExportResult(res);
      setSuccessMessage(`Patient FHIR R4 Bundle exported successfully.`);
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b pb-4 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-gray-900">Healthcare Interoperability Center</h1>
            <span className="inline-flex items-center rounded-md bg-teal-50 px-2.5 py-0.5 text-xs font-semibold text-teal-800 border border-teal-200">
              Phase 14 Production
            </span>
            <span className="inline-flex items-center rounded-md bg-blue-50 px-2.5 py-0.5 text-xs font-semibold text-blue-800 border border-blue-200">
              HL7 FHIR R4
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Standardized health data exchange with Malaffi HIE, SEHA EMR, diagnostic laboratories and UAE Pass identity.
          </p>
        </div>

        <button
          onClick={loadProvidersAndMonitoring}
          className="self-start md:self-auto bg-white border border-gray-300 text-gray-700 px-3 py-1.5 rounded-md text-sm font-medium hover:bg-gray-50 shadow-sm"
        >
          Refresh Adapters
        </button>
      </div>

      {/* Honest Boundary Notice */}
      <div className="rounded-md bg-blue-50 p-4 border border-blue-200">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-semibold text-blue-900">Honest External Integration Boundary</h3>
            <p className="mt-1 text-xs text-blue-800">
              HL7 FHIR R4 serialization, identity reconciliation, and terminology mapping are <strong>fully functional</strong>.
              However, external live connections to Malaffi, SEHA, or UAE Pass production gateways are strictly reported as{' '}
              <strong>NOT_CONFIGURED</strong> until valid DOH-issued mTLS certificates and security credentials are provided.
              Zero simulated or synthetic connection claims are permitted.
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-4 text-sm text-red-700 border border-red-200 flex justify-between items-center">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700 text-xs">Dismiss</button>
        </div>
      )}

      {successMessage && (
        <div className="rounded-md bg-emerald-50 p-4 text-sm text-emerald-700 border border-emerald-200 flex justify-between items-center">
          <span>{successMessage}</span>
          <button onClick={() => setSuccessMessage(null)} className="text-emerald-500 hover:text-emerald-700 text-xs">Dismiss</button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          {[
            { id: 'adapters', name: `Integration Adapters (${providers.length})` },
            { id: 'ingest', name: 'FHIR R4 Ingestion Sandbox' },
            { id: 'export', name: 'FHIR R4 Patient Export' },
            { id: 'terminology', name: 'Governed Terminology Lookup' },
            { id: 'monitoring', name: 'Job Logs & Provenance' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-teal-600 text-teal-700 font-semibold'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* TAB 1: ADAPTERS */}
      {activeTab === 'adapters' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {providers.map((p) => {
              const isConfigured = p.status === 'CONFIGURED' || p.status === 'ACTIVE';
              return (
                <div key={p.provider_name} className="bg-white p-5 rounded-lg border shadow-sm space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-gray-900">{p.display_name}</h3>
                      <span className="text-xs font-mono text-gray-500">{p.provider_type}</span>
                    </div>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold ${
                        isConfigured
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800 border border-amber-300'
                      }`}
                    >
                      {p.status}
                    </span>
                  </div>

                  <div className="text-xs text-gray-600 space-y-1">
                    <div>Endpoint: <span className="font-mono text-gray-800">{p.endpoint_url || 'Not provisioned'}</span></div>
                    <div>Enabled: <span className="font-semibold">{p.is_enabled ? 'Yes' : 'No'}</span></div>
                  </div>

                  <div className="border-t pt-3">
                    <span className="text-xs font-semibold text-gray-700 block mb-1">Missing Production Credentials:</span>
                    <ul className="list-disc list-inside text-xs text-amber-800 space-y-0.5">
                      {p.prerequisites.map((prereq, idx) => (
                        <li key={idx} className="font-mono text-[11px]">{prereq}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 2: FHIR R4 INGESTION SANDBOX */}
      {activeTab === 'ingest' && (
        <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4 max-w-4xl">
          <h3 className="text-base font-semibold text-gray-900">HL7 FHIR R4 Bundle Ingestion</h3>
          <p className="text-xs text-gray-500">
            Simulate or process an incoming FHIR R4 Bundle from external hospital laboratories or Malaffi reciprocal transfers.
            The pipeline resolves patient identity, translates LOINC codes, and persists a cryptographic provenance audit record.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Source System</label>
              <select
                value={sourceSystem}
                onChange={(e) => setSourceSystem(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              >
                <option value="MALAFFI">Malaffi HIE</option>
                <option value="SEHA_EMR">SEHA Clinic / Hospital EMR</option>
                <option value="LAB_NETWORK">National Diagnostic Lab Network</option>
                <option value="EXTERNAL_CLINIC">External Primary Care Clinic</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
              HL7 FHIR R4 Bundle (JSON)
            </label>
            <textarea
              rows={12}
              value={fhirPayload}
              onChange={(e) => setFhirPayload(e.target.value)}
              className="w-full font-mono text-xs rounded-md border border-gray-300 p-3 bg-gray-50 focus:border-teal-500 focus:outline-none"
            />
          </div>

          <div className="flex justify-end gap-3 pt-2">
            <button
              onClick={handleIngestBundle}
              disabled={isLoading}
              className="bg-teal-600 text-white px-5 py-2 rounded-md text-sm font-semibold hover:bg-teal-700 shadow-sm disabled:opacity-50"
            >
              {isLoading ? 'Ingesting & Validating...' : 'Ingest FHIR Bundle with Provenance'}
            </button>
          </div>

          {ingestResult && (
            <div className="mt-4 border-t pt-4">
              <h4 className="text-sm font-semibold text-emerald-800 mb-2">Ingestion Pipeline Result</h4>
              <pre className="bg-gray-900 text-emerald-300 p-4 rounded-md text-xs overflow-x-auto">
                {JSON.stringify(ingestResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: FHIR R4 PATIENT EXPORT */}
      {activeTab === 'export' && (
        <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4 max-w-4xl">
          <h3 className="text-base font-semibold text-gray-900">Export Patient as HL7 FHIR R4 Bundle</h3>
          <p className="text-xs text-gray-500">
            Export a patient profile, clinical observations, encounters, and preventive screening records as a standard FHIR R4 Bundle for external care transitions.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Select Patient</label>
              <select
                value={selectedExportPatientId}
                onChange={(e) => setSelectedExportPatientId(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
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
                className="w-full bg-teal-600 text-white py-2 px-4 rounded-md text-sm font-semibold hover:bg-teal-700 shadow-sm disabled:opacity-50"
              >
                {isLoading ? 'Serializing...' : 'Export Standard FHIR R4 Bundle'}
              </button>
            </div>
          </div>

          {exportResult && (
            <div className="mt-4 border-t pt-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-900">FHIR R4 Bundle Payload</h4>
                <span className="text-xs text-gray-400">
                  Exported at: {new Date(exportResult.exported_at).toLocaleString()}
                </span>
              </div>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-md text-xs overflow-x-auto max-h-96">
                {JSON.stringify(exportResult.bundle, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: GOVERNED TERMINOLOGY LOOKUP */}
      {activeTab === 'terminology' && (
        <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4 max-w-2xl">
          <h3 className="text-base font-semibold text-gray-900">Governed Medical Terminology Mapper</h3>
          <p className="text-xs text-gray-500">
            Inspect canonical mappings between WELLWATCH internal clinical codes and standard medical vocabularies (LOINC, SNOMED CT, ICD-10, UCUM).
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Coding System</label>
              <select
                value={lookupSystem}
                onChange={(e) => setLookupSystem(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              >
                <option value="LOINC">LOINC (Observations & Labs)</option>
                <option value="SNOMED_CT">SNOMED CT (Clinical Encounters)</option>
                <option value="ICD_10">ICD-10 (Conditions & Diagnoses)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Standard Code</label>
              <input
                type="text"
                value={lookupCode}
                onChange={(e) => setLookupCode(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2 text-sm font-mono focus:border-teal-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={handleLookupTerminology}
              className="bg-teal-600 text-white px-5 py-2 rounded-md text-sm font-semibold hover:bg-teal-700 shadow-sm"
            >
              Query Terminology Directory
            </button>
          </div>

          {termResult && (
            <div className="mt-4 border-t pt-4">
              <h4 className="text-sm font-semibold text-gray-900 mb-2">Mapping Result</h4>
              <div className="bg-gray-50 p-4 rounded-md border text-xs space-y-2">
                <div>Internal Code: <strong className="font-mono text-gray-800">{termResult.internal_code}</strong></div>
                <div>Display Name: <strong className="text-gray-800">{termResult.display_name}</strong></div>
                <div>Standard Code: <span className="font-mono bg-white px-2 py-0.5 rounded border">{termResult.target_system}: {termResult.target_code}</span></div>
                <div>Standard Concept: <span>{termResult.target_display}</span></div>
                <div>Canonical UCUM Unit: <span className="font-mono">{termResult.ucum_unit || 'None'}</span></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: MONITORING & PROVENANCE */}
      {activeTab === 'monitoring' && monitoring && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-5 rounded-lg border shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Total Adapters</div>
              <div className="mt-2 text-3xl font-bold text-gray-900">{monitoring.total_providers}</div>
            </div>

            <div className="bg-white p-5 rounded-lg border shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Configured in Prod</div>
              <div className="mt-2 text-3xl font-bold text-emerald-600">{monitoring.configured_providers}</div>
            </div>

            <div className="bg-white p-5 rounded-lg border shadow-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Pending Certificates</div>
              <div className="mt-2 text-3xl font-bold text-amber-600">{monitoring.not_configured_providers}</div>
              <div className="mt-1 text-xs text-amber-700">Honest boundary enforced</div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border shadow-sm space-y-3">
            <h4 className="text-sm font-semibold text-gray-900">Recent Integration Execution Logs</h4>
            <div className="text-xs text-gray-500">
              {monitoring.recent_jobs && monitoring.recent_jobs.length > 0 ? (
                <div className="space-y-2">
                  {monitoring.recent_jobs.map((job, idx) => (
                    <div key={idx} className="bg-gray-50 p-3 rounded-md border flex justify-between items-center">
                      <div>
                        <span className="font-semibold text-gray-800">{job.provider_name}</span> - {job.job_type}
                        <div className="text-[11px] text-gray-400 font-mono">{job.status}</div>
                      </div>
                      <span className="text-xs text-gray-400">{new Date(job.executed_at).toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 bg-gray-50 rounded-md border text-center text-gray-400">
                  No historical integration jobs recorded in the active session.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function IntegrationsPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-sm text-gray-500">Loading Interoperability Center...</div>}>
      <IntegrationsContent />
    </Suspense>
  );
}
