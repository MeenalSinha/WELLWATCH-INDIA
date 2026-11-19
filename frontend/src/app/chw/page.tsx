'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { apiClient, ApiError } from '@/lib/api';
import {
  saveOutboxOperation,
  getOutboxOperations,
  clearOutbox,
  removeOutboxOperation,
  cacheAssignmentPackage,
  getCachedAssignmentPackage,
  isAssignmentPackageExpired,
  saveDeviceBinding,
  getSavedDeviceBinding,
} from '@/lib/offlineStorage';
import {
  OfflinePatientSummary,
  OfflineAssignmentPackageResponse,
  OfflineSyncBatchRequest,
  OfflineSyncOperationRequest,
  OfflineSyncBatchResponse,
  SyncConflictResponse,
} from '@/types';

function ChwWorkspaceContent() {
  const [activeTab, setActiveTab] = useState<'assigned' | 'screening' | 'outbox' | 'conflicts'>('assigned');
  const [deviceId, setDeviceId] = useState<string>('CHW-DEV-001');
  const [isDeviceRegistered, setIsDeviceRegistered] = useState<boolean>(true);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [isExpiredCache, setIsExpiredCache] = useState<boolean>(false);

  // Assignment data
  const [assignmentPkg, setAssignmentPkg] = useState<OfflineAssignmentPackageResponse | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<OfflinePatientSummary | null>(null);

  // Field screening form state
  const [systolic, setSystolic] = useState<number>(120);
  const [diastolic, setDiastolic] = useState<number>(80);
  const [glucose, setGlucose] = useState<number>(95);
  const [bmi, setBmi] = useState<number>(23.5);
  const [notes, setNotes] = useState<string>('Routine community field screening.');

  // Outbox queue (durable IndexedDB persistence for offline field use)
  const [outbox, setOutbox] = useState<OfflineSyncOperationRequest[]>([]);
  const [syncResults, setSyncResults] = useState<OfflineSyncBatchResponse | null>(null);
  const [conflicts, setConflicts] = useState<SyncConflictResponse[]>([]);

  // Conflict resolution state
  const [selectedConflict, setSelectedConflict] = useState<SyncConflictResponse | null>(null);
  const [resolutionChoice, setResolutionChoice] = useState<'ACCEPT_CLIENT' | 'KEEP_SERVER'>('KEEP_SERVER');
  const [resolutionNotes, setResolutionNotes] = useState<string>('');

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    // Network status listener
    if (typeof window !== 'undefined') {
      setIsOnline(navigator.onLine);
      const onOnline = () => setIsOnline(true);
      const onOffline = () => setIsOnline(false);
      window.addEventListener('online', onOnline);
      window.addEventListener('offline', onOffline);

      // Load durable offline state
      initDurableState();

      return () => {
        window.removeEventListener('online', onOnline);
        window.removeEventListener('offline', onOffline);
      };
    } else {
      initDurableState();
    }
  }, []);

  const initDurableState = async () => {
    // 1. Restore device binding
    try {
      const dev = await getSavedDeviceBinding();
      if (dev) {
        setDeviceId(dev.deviceId);
        setIsDeviceRegistered(dev.isRegistered);
      }
    } catch (e) {
      console.warn('Device binding restore note:', e);
    }

    // 2. Restore pending offline outbox operations from IndexedDB
    try {
      const storedOps = await getOutboxOperations();
      setOutbox(storedOps);
    } catch (e) {
      console.warn('Outbox restore note:', e);
    }

    // 3. Load assignments (try cached first, then live if online)
    await loadAssignments();
    await loadConflicts();
  };

  const loadAssignments = async () => {
    setIsLoading(true);
    setError(null);

    // Check cached package first
    try {
      const cached = await getCachedAssignmentPackage();
      if (cached) {
        setAssignmentPkg(cached);
        const expired = isAssignmentPackageExpired(cached, 72);
        setIsExpiredCache(expired);
        if (cached.assigned_patients.length > 0 && !selectedPatient) {
          setSelectedPatient(cached.assigned_patients[0]);
        }
      }
    } catch (e) {
      console.warn('Cached package check note:', e);
    }

    // Attempt live fetch if online
    try {
      const pkg = await apiClient.getOfflineAssignments(72);
      setAssignmentPkg(pkg);
      setIsExpiredCache(false);
      await cacheAssignmentPackage(pkg);
      if (pkg.assigned_patients.length > 0 && !selectedPatient) {
        setSelectedPatient(pkg.assigned_patients[0]);
      }
    } catch (err: any) {
      console.warn('Live assignment load note:', err);
      // If offline and have cached package, stay on cached
      if (assignmentPkg) {
        setError('Working with locally cached offline cohort (Network unavailable).');
      } else {
        setError(err instanceof ApiError ? err.message : 'Working in offline mode or unable to fetch server assignments.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const loadConflicts = async () => {
    try {
      const conflictList = await apiClient.getSyncConflicts('PENDING');
      setConflicts(conflictList);
    } catch (err) {
      console.warn('Conflicts load note:', err);
    }
  };

  const handleRegisterDevice = async () => {
    setError(null);
    try {
      await apiClient.registerOfflineDevice({
        device_id: deviceId,
        device_model: 'Field Tablet Pro 12',
        os_platform: 'WebPWA',
        app_version: '1.0.0',
      });
      setIsDeviceRegistered(true);
      await saveDeviceBinding(deviceId, true);
      setSuccessMessage(`Device ${deviceId} registered successfully.`);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to register field device.');
    }
  };

  const handleAddToOutbox = async () => {
    if (!selectedPatient) {
      setError('Please select an assigned patient.');
      return;
    }

    // Cryptographically secure UUID generation for operation identity
    const uuidPart = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID().slice(0, 8)
      : Date.now().toString(36);
    const opId = `OP-${Date.now()}-${uuidPart}`;
    const idempKey = `IDEMP-${selectedPatient.patient_id}-${Date.now()}`;

    const newOp: OfflineSyncOperationRequest = {
      operation_id: opId,
      idempotency_key: idempKey,
      resource_type: 'SCREENING',
      operation_type: 'CREATE',
      client_timestamp: new Date().toISOString(),
      payload: {
        patient_id: selectedPatient.patient_id,
        encounter_type: 'COMMUNITY_VITALS_CHECK',
        blood_pressure_systolic: systolic,
        blood_pressure_diastolic: diastolic,
        glucose_value: glucose,
        bmi_value: bmi,
        clinical_notes: notes,
      },
    };

    // Durable persistence in IndexedDB outbox
    await saveOutboxOperation(newOp);
    setOutbox((prev) => [...prev, newOp]);
    setSuccessMessage(`Screening for ${selectedPatient.full_name} saved to durable offline outbox.`);
    setActiveTab('outbox');
  };

  const handleSyncOutbox = async () => {
    if (outbox.length === 0) {
      setError('Outbox is empty. No operations to synchronize.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);

    const batchReq: OfflineSyncBatchRequest = {
      device_id: deviceId,
      operations: outbox,
    };

    try {
      const res = await apiClient.syncOfflineBatch(batchReq);
      setSyncResults(res);

      // Clean up successfully synced operations from IndexedDB
      for (const result of res.results) {
        if (result.status === 'SYNCED') {
          await removeOutboxOperation(result.operation_id);
        }
      }

      // Reload remaining operations from IndexedDB
      const remaining = await getOutboxOperations();
      setOutbox(remaining);

      setSuccessMessage(
        `Sync completed: ${res.synced_count} synced, ${res.conflict_count} conflicts detected, ${res.rejected_count} rejected.`
      );
      await loadConflicts();
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Sync failed. Server unreachable or rejected batch.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResolveConflict = async () => {
    if (!selectedConflict) return;
    if (!resolutionNotes.trim()) {
      setError('A mandatory clinical rationale must be entered to resolve clinical data conflicts.');
      return;
    }

    try {
      await apiClient.resolveSyncConflict(selectedConflict.conflict_id, {
        resolution: resolutionChoice,
        notes: resolutionNotes,
      });
      setSuccessMessage(`Conflict ${selectedConflict.conflict_id} resolved with ${resolutionChoice}.`);
      setSelectedConflict(null);
      setResolutionNotes('');
      await loadConflicts();
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : 'Failed to resolve conflict.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Device Status */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b pb-4 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-gray-900">Community Health Worker (CHW) Workspace</h1>
            <span className="inline-flex items-center rounded-md bg-teal-50 px-2.5 py-0.5 text-xs font-semibold text-teal-800 border border-teal-200">
              Phase 13 Offline-First
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Resilient community field screening, local encrypted assignment caching, and clinical conflict safety.
          </p>
        </div>

        {/* Connectivity & Device Control */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 bg-gray-50 border rounded-md px-3 py-1.5 text-xs">
            <span className="font-semibold text-gray-600">Storage:</span>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold bg-teal-100 text-teal-800">
              IndexedDB Persistent
            </span>
          </div>

          <div className="flex items-center gap-2 bg-gray-50 border rounded-md px-3 py-1.5 text-xs">
            <span className="font-semibold text-gray-600">Network:</span>
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-bold ${
                isOnline ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
              }`}
            >
              {isOnline ? 'ONLINE' : 'OFFLINE (FIELD)'}
            </span>
            <button
              onClick={() => setIsOnline(!isOnline)}
              className="text-gray-400 hover:text-gray-600 underline ml-1"
            >
              Toggle
            </button>
          </div>

          <div className="flex items-center gap-2 bg-gray-50 border rounded-md px-3 py-1.5 text-xs">
            <span className="font-semibold text-gray-600">Device:</span>
            <span className="font-mono text-gray-800">{deviceId}</span>
          </div>
        </div>
      </div>

      {isExpiredCache && (
        <div className="rounded-md bg-amber-50 p-4 text-sm text-amber-800 border border-amber-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-bold">POLICY WARNING:</span>
            <span>Offline assignment cache duration has exceeded 72 hours. Reconnection and fresh cohort download are required per DoH Offline Operations Policy.</span>
          </div>
        </div>
      )}

      {/* Safety Notice: No Silent Overwrites */}
      <div className="rounded-md bg-emerald-50 p-4 border border-emerald-200">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-semibold text-emerald-900">Clinical Record Preservation & Conflict Isolation</h3>
            <p className="mt-1 text-xs text-emerald-800">
              Offline synchronizations strictly enforce clinical safety: newer verified medical observations in hospital records are <strong>never overwritten silently</strong> with last-write-wins.
              Discrepant observations are quarantined in the clinical conflict queue for licensed provider review.
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

      {/* Workspace Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          {[
            { id: 'assigned', name: `Assigned Patients (${assignmentPkg?.assigned_patients.length || 0})` },
            { id: 'screening', name: 'Field Screening Form' },
            { id: 'outbox', name: `Sync Outbox (${outbox.length})` },
            { id: 'conflicts', name: `Conflict Queue (${conflicts.length})` },
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

      {/* TAB 1: ASSIGNED PATIENTS */}
      {activeTab === 'assigned' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700 uppercase">Assigned Community Cohort</h3>
            <span className="text-xs text-gray-500">
              Package expires: {assignmentPkg?.expires_at ? new Date(assignmentPkg.expires_at).toLocaleString() : '72 hours'}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {assignmentPkg?.assigned_patients.map((pat) => (
              <div
                key={pat.patient_id}
                onClick={() => {
                  setSelectedPatient(pat);
                  setActiveTab('screening');
                }}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedPatient?.patient_id === pat.patient_id
                    ? 'border-teal-500 bg-teal-50/50 shadow-sm'
                    : 'bg-white hover:border-gray-300 shadow-sm'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-gray-900">{pat.full_name}</span>
                  <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded font-mono">
                    {pat.gender}
                  </span>
                </div>
                <div className="mt-2 text-xs text-gray-500 space-y-1">
                  <div>Emirates ID: {pat.emirates_id || 'Not on file'}</div>
                  <div>DOB: {pat.date_of_birth || 'N/A'}</div>
                  <div className="flex justify-between items-center pt-2 border-t mt-2">
                    <span className="text-teal-700 font-medium">Care Gaps: {pat.active_care_gaps_count}</span>
                    <span className="text-xs text-teal-600 hover:underline">Select for Screening &rarr;</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {(!assignmentPkg || assignmentPkg.assigned_patients.length === 0) && (
            <div className="bg-white p-8 text-center rounded-lg border text-gray-500 text-sm">
              No assigned patients found for this CHW. Click below to load demo assignment.
              <div className="mt-3">
                <button
                  onClick={loadAssignments}
                  className="bg-teal-600 text-white px-4 py-2 rounded-md text-xs font-medium hover:bg-teal-700"
                >
                  Reload Current Assignment Package
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: FIELD SCREENING FORM */}
      {activeTab === 'screening' && (
        <div className="bg-white p-6 rounded-lg border shadow-sm max-w-2xl space-y-4">
          <div className="flex items-center justify-between border-b pb-3">
            <h3 className="text-base font-semibold text-gray-900">Conduct Field Health Check</h3>
            {selectedPatient ? (
              <span className="text-xs bg-teal-100 text-teal-800 px-2.5 py-1 rounded font-medium">
                Patient: {selectedPatient.full_name}
              </span>
            ) : (
              <span className="text-xs text-amber-700">Please select an assigned patient</span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                Systolic BP (mmHg)
              </label>
              <input
                type="number"
                value={systolic}
                onChange={(e) => setSystolic(Number(e.target.value))}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                Diastolic BP (mmHg)
              </label>
              <input
                type="number"
                value={diastolic}
                onChange={(e) => setDiastolic(Number(e.target.value))}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                Fasting Glucose (mg/dL)
              </label>
              <input
                type="number"
                value={glucose}
                onChange={(e) => setGlucose(Number(e.target.value))}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                Calculated BMI (kg/m²)
              </label>
              <input
                type="number"
                step="0.1"
                value={bmi}
                onChange={(e) => setBmi(Number(e.target.value))}
                className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
              Field Clinical Observation Notes
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
            />
          </div>

          <div className="pt-2 flex justify-end gap-3">
            <button
              onClick={handleAddToOutbox}
              className="bg-teal-600 text-white px-5 py-2 rounded-md text-sm font-semibold hover:bg-teal-700 shadow-sm"
            >
              Save to Offline Outbox
            </button>
          </div>
        </div>
      )}

      {/* TAB 3: SYNC OUTBOX */}
      {activeTab === 'outbox' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700 uppercase">
              Pending Field Operations ({outbox.length})
            </h3>
            <button
              onClick={handleSyncOutbox}
              disabled={isLoading || outbox.length === 0}
              className="bg-emerald-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-emerald-700 shadow-sm disabled:opacity-50"
            >
              {isLoading ? 'Synchronizing Batch...' : 'Synchronize Outbox to Server'}
            </button>
          </div>

          <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 text-xs font-semibold text-gray-500 uppercase">
                <tr>
                  <th className="px-6 py-3 text-left">Operation ID</th>
                  <th className="px-6 py-3 text-left">Resource</th>
                  <th className="px-6 py-3 text-left">Idempotency Key</th>
                  <th className="px-6 py-3 text-left">Client Timestamp</th>
                  <th className="px-6 py-3 text-left">Payload Summary</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 text-xs">
                {outbox.map((op) => (
                  <tr key={op.operation_id}>
                    <td className="px-6 py-4 font-mono font-medium text-gray-900">{op.operation_id}</td>
                    <td className="px-6 py-4">{op.resource_type}</td>
                    <td className="px-6 py-4 font-mono text-gray-500">{op.idempotency_key}</td>
                    <td className="px-6 py-4 text-gray-500">{new Date(op.client_timestamp).toLocaleTimeString()}</td>
                    <td className="px-6 py-4 text-gray-600">
                      BP: {op.payload.blood_pressure_systolic}/{op.payload.blood_pressure_diastolic} | Glucose: {op.payload.glucose_value}
                    </td>
                  </tr>
                ))}
                {outbox.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-gray-400">
                      All offline records are synchronized. Outbox is currently empty.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {syncResults && (
            <div className="bg-gray-900 text-gray-100 p-4 rounded-md text-xs space-y-2">
              <div className="font-semibold text-emerald-400">Last Batch Synchronization Result:</div>
              <pre>{JSON.stringify(syncResults, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: CLINICAL CONFLICT REVIEW */}
      {activeTab === 'conflicts' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700 uppercase">
              Clinical Conflicts Requiring Provider Resolution ({conflicts.length})
            </h3>
            <button
              onClick={loadConflicts}
              className="bg-white border text-gray-700 px-3 py-1.5 rounded-md text-xs font-medium hover:bg-gray-50 shadow-sm"
            >
              Refresh Queue
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {conflicts.map((conf) => (
              <div
                key={conf.conflict_id}
                onClick={() => setSelectedConflict(conf)}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedConflict?.conflict_id === conf.conflict_id
                    ? 'border-red-500 bg-red-50/30 shadow-md'
                    : 'bg-white hover:border-gray-300 shadow-sm'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-red-700">{conf.conflict_id}</span>
                  <span className="text-xs bg-red-100 text-red-800 px-2 py-0.5 rounded font-semibold">
                    {conf.resolution_status}
                  </span>
                </div>
                <div className="mt-2 text-xs text-gray-800 font-medium">{conf.conflict_reason}</div>
                <div className="mt-2 pt-2 border-t grid grid-cols-2 gap-2 text-xs text-gray-600">
                  <div>
                    <span className="font-semibold block text-gray-700">Client Field Data:</span>
                    <pre className="bg-gray-50 p-2 rounded text-[10px] mt-1 overflow-x-auto">
                      {JSON.stringify(conf.client_payload, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <span className="font-semibold block text-gray-700">Server Hospital Data:</span>
                    <pre className="bg-gray-50 p-2 rounded text-[10px] mt-1 overflow-x-auto">
                      {JSON.stringify(conf.server_payload, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            ))}

            {conflicts.length === 0 && (
              <div className="col-span-2 bg-white p-8 text-center rounded-lg border text-gray-500 text-sm">
                No outstanding clinical conflicts. All field and facility data records are concordant.
              </div>
            )}
          </div>

          {/* Conflict Resolution Modal / Action Card */}
          {selectedConflict && (
            <div className="bg-white p-6 rounded-lg border-2 border-red-300 shadow-lg space-y-4">
              <h4 className="text-base font-semibold text-gray-900">
                Resolve Conflict: {selectedConflict.conflict_id}
              </h4>
              <p className="text-xs text-gray-600">
                As a licensed healthcare provider, choose whether to accept the offline CHW observation or preserve the server clinical record.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                    Resolution Action
                  </label>
                  <select
                    value={resolutionChoice}
                    onChange={(e) => setResolutionChoice(e.target.value as any)}
                    className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
                  >
                    <option value="KEEP_SERVER">KEEP_SERVER (Preserve verified hospital observation)</option>
                    <option value="ACCEPT_CLIENT">ACCEPT_CLIENT (Overwrite with CHW field measurement)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">
                    Mandatory Clinical Rationale Notes
                  </label>
                  <input
                    type="text"
                    placeholder="Document clinical reason for resolution decision..."
                    value={resolutionNotes}
                    onChange={(e) => setResolutionNotes(e.target.value)}
                    className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-teal-500 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  onClick={() => setSelectedConflict(null)}
                  className="bg-white border text-gray-700 px-4 py-2 rounded-md text-xs font-medium hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleResolveConflict}
                  className="bg-red-600 text-white px-5 py-2 rounded-md text-xs font-semibold hover:bg-red-700 shadow-sm"
                >
                  Confirm Resolution & Audit Log
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ChwWorkspacePage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-sm text-gray-500">Loading CHW Workspace...</div>}>
      <ChwWorkspaceContent />
    </Suspense>
  );
}
