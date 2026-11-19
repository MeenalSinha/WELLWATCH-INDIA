'use client';

import React, { useState, useEffect, Suspense } from 'react';
import {
  Smartphone,
  Wifi,
  WifiOff,
  Database,
  Users,
  Stethoscope,
  Send,
  AlertTriangle,
  CheckCircle2,
  Clock,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  HardDrive,
} from 'lucide-react';
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
import { AppShell } from '@/components/AppShell';
import { StatusBadge } from '@/components/StatusBadge';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { EmptyState } from '@/components/EmptyState';

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

      for (const result of res.results) {
        if (result.status === 'SYNCED') {
          await removeOutboxOperation(result.operation_id);
        }
      }

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
    <AppShell
      pageTitle="Community Health Worker Workspace"
      breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'CHW Field' }]}
    >
      <div className="space-y-6">
        {/* Field Device & Offline Status Header */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2.5 flex-wrap">
              <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                CHW Field Workspace
              </h1>
              <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-forest-50 text-forest-800 border border-forest-200">
                Phase 13 Offline-First Engine
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Resilient community field screening, durable IndexedDB assignment caching, and clinical conflict safety.
            </p>
          </div>

          {/* Connectivity & Device Indicators */}
          <div className="flex flex-wrap items-center gap-2.5 shrink-0 text-xs">
            {/* Storage Badge */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F8FAF9] border border-slate-200 text-slate-700">
              <HardDrive className="w-3.5 h-3.5 text-forest-700" />
              <span>Storage:</span>
              <strong className="text-slate-900">IndexedDB</strong>
            </div>

            {/* Network Status Toggle */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F8FAF9] border border-slate-200">
              {isOnline ? (
                <Wifi className="w-3.5 h-3.5 text-emerald-600" />
              ) : (
                <WifiOff className="w-3.5 h-3.5 text-amber-600" />
              )}
              <span className={`font-bold ${isOnline ? 'text-emerald-700' : 'text-amber-700'}`}>
                {isOnline ? 'ONLINE' : 'OFFLINE (FIELD)'}
              </span>
              <button
                onClick={() => setIsOnline(!isOnline)}
                className="text-[10px] text-slate-400 hover:text-slate-700 underline ml-1"
                title="Toggle simulated connection"
              >
                Toggle
              </button>
            </div>

            {/* Device ID */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F8FAF9] border border-slate-200 font-mono text-[11px] text-slate-600">
              <Smartphone className="w-3.5 h-3.5 text-slate-500" />
              <span>{deviceId}</span>
            </div>
          </div>
        </div>

        {/* Safety Notice: No Silent Overwrites */}
        <div className="p-4 rounded-xl bg-forest-50/70 border border-forest-100 flex items-start gap-3">
          <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0 mt-0.5" />
          <div className="space-y-0.5 text-xs">
            <div className="font-bold text-forest-900">
              Clinical Record Preservation & Conflict Isolation
            </div>
            <p className="text-forest-800/90 leading-relaxed text-[11px]">
              Offline synchronizations strictly enforce clinical safety: newer verified medical observations in hospital records are <strong>never overwritten silently</strong> with last-write-wins.
              Discrepant observations are quarantined in the clinical conflict queue for licensed provider review.
            </p>
          </div>
        </div>

        {isExpiredCache && (
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 text-xs text-amber-900 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
              <span>
                <strong>Policy Warning:</strong> Offline assignment package has exceeded 72 hours. Reconnection and fresh cohort download are mandated under DoH Offline Operations Policy.
              </span>
            </div>
          </div>
        )}

        {/* Alerts */}
        {error && (
          <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 font-medium flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-rose-600 text-xs font-bold">✕</button>
          </div>
        )}
        {successMessage && (
          <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 font-medium flex items-center justify-between">
            <span>{successMessage}</span>
            <button onClick={() => setSuccessMessage(null)} className="text-emerald-600 text-xs font-bold">✕</button>
          </div>
        )}

        {/* Workspace Navigation Tabs */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
          <div className="border-b border-slate-200/80 px-6 pt-3 flex gap-6 overflow-x-auto text-xs font-medium">
            {[
              { id: 'assigned', name: `Assigned Patients (${assignmentPkg?.assigned_patients.length || 0})`, icon: Users },
              { id: 'screening', name: 'Field Screening Form', icon: Stethoscope },
              { id: 'outbox', name: `Sync Outbox (${outbox.length})`, icon: Send },
              { id: 'conflicts', name: `Conflict Queue (${conflicts.length})`, icon: AlertTriangle },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`pb-3.5 border-b-2 font-semibold flex items-center gap-2 transition whitespace-nowrap ${
                    isActive
                      ? 'border-forest-700 text-forest-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>

          <div className="p-6">
            {/* TAB 1: ASSIGNED PATIENTS */}
            {activeTab === 'assigned' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Assigned Community Cohort
                  </h3>
                  <span className="text-xs text-slate-500">
                    Package expires: {assignmentPkg?.expires_at ? new Date(assignmentPkg.expires_at).toLocaleString() : '72 hours'}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {assignmentPkg?.assigned_patients.map((pat) => {
                    const isSelected = selectedPatient?.patient_id === pat.patient_id;
                    return (
                      <div
                        key={pat.patient_id}
                        onClick={() => {
                          setSelectedPatient(pat);
                          setActiveTab('screening');
                        }}
                        className={`p-4 rounded-xl border cursor-pointer transition-all ${
                          isSelected
                            ? 'border-forest-700 bg-forest-50/50 shadow-xs'
                            : 'border-slate-200/90 bg-white hover:border-slate-300 shadow-2xs'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-900 text-sm">{pat.full_name}</span>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">
                            {pat.gender}
                          </span>
                        </div>
                        <div className="mt-2.5 text-xs text-slate-500 space-y-1">
                          <div>Emirates ID: <strong className="text-slate-700 font-mono">{pat.emirates_id || 'Not on file'}</strong></div>
                          <div>DOB: <strong className="text-slate-700">{pat.date_of_birth || 'N/A'}</strong></div>
                          <div className="flex justify-between items-center pt-2.5 border-t border-slate-100 mt-2">
                            <span className="text-forest-700 font-semibold text-xs">Care Gaps: {pat.active_care_gaps_count}</span>
                            <span className="text-xs text-forest-800 font-medium flex items-center gap-1 hover:underline">
                              Start Screening &rarr;
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {(!assignmentPkg || assignmentPkg.assigned_patients.length === 0) && (
                  <EmptyState
                    title="No Assigned Patients Found"
                    description="No assigned community members found in the current cache package."
                    actionText="Reload Current Assignment Package"
                    onAction={loadAssignments}
                  />
                )}
              </div>
            )}

            {/* TAB 2: FIELD SCREENING FORM */}
            {activeTab === 'screening' && (
              <div className="max-w-2xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Conduct Community Health Screening
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Log biometric vitals in the field. Data is saved immediately to IndexedDB.
                    </p>
                  </div>
                  {selectedPatient ? (
                    <span className="text-xs px-2.5 py-1 rounded-full bg-forest-50 text-forest-800 border border-forest-200 font-semibold">
                      Target: {selectedPatient.full_name}
                    </span>
                  ) : (
                    <span className="text-xs text-amber-700">Please select an assigned patient</span>
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Systolic BP (mmHg)
                    </label>
                    <input
                      type="number"
                      value={systolic}
                      onChange={(e) => setSystolic(Number(e.target.value))}
                      className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Diastolic BP (mmHg)
                    </label>
                    <input
                      type="number"
                      value={diastolic}
                      onChange={(e) => setDiastolic(Number(e.target.value))}
                      className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Fasting Glucose (mg/dL)
                    </label>
                    <input
                      type="number"
                      value={glucose}
                      onChange={(e) => setGlucose(Number(e.target.value))}
                      className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Calculated BMI (kg/m²)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={bmi}
                      onChange={(e) => setBmi(Number(e.target.value))}
                      className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Field Clinical Observation Notes
                  </label>
                  <textarea
                    rows={3}
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                  />
                </div>

                <div className="pt-2 flex justify-end">
                  <button
                    onClick={handleAddToOutbox}
                    className="px-5 py-2.5 rounded-lg bg-forest-800 hover:bg-forest-900 text-white font-semibold text-xs shadow-xs transition flex items-center gap-1.5"
                  >
                    <HardDrive className="w-3.5 h-3.5" />
                    Save to Offline Outbox (IndexedDB)
                  </button>
                </div>
              </div>
            )}

            {/* TAB 3: SYNC OUTBOX */}
            {activeTab === 'outbox' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Pending Field Operations ({outbox.length})
                  </h3>
                  <button
                    onClick={handleSyncOutbox}
                    disabled={isLoading || outbox.length === 0}
                    className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs shadow-xs transition disabled:opacity-50 flex items-center gap-1.5"
                  >
                    <Send className="w-3.5 h-3.5" />
                    {isLoading ? 'Synchronizing Batch...' : 'Synchronize Outbox to Server'}
                  </button>
                </div>

                <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden">
                  <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead className="bg-[#F8FAF9] text-slate-600 uppercase tracking-wider font-semibold text-[11px]">
                      <tr>
                        <th className="px-5 py-3 text-left">Operation ID</th>
                        <th className="px-5 py-3 text-left">Resource</th>
                        <th className="px-5 py-3 text-left">Idempotency Key</th>
                        <th className="px-5 py-3 text-left">Client Timestamp</th>
                        <th className="px-5 py-3 text-left">Payload Summary</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {outbox.map((op) => (
                        <tr key={op.operation_id} className="hover:bg-slate-50/60">
                          <td className="px-5 py-3 font-mono font-medium text-slate-900">{op.operation_id}</td>
                          <td className="px-5 py-3">{op.resource_type}</td>
                          <td className="px-5 py-3 font-mono text-slate-500">{op.idempotency_key}</td>
                          <td className="px-5 py-3 text-slate-500">{new Date(op.client_timestamp).toLocaleTimeString()}</td>
                          <td className="px-5 py-3 text-slate-700 font-medium">
                            BP: {op.payload.blood_pressure_systolic}/{op.payload.blood_pressure_diastolic} | Glucose: {op.payload.glucose_value}
                          </td>
                        </tr>
                      ))}
                      {outbox.length === 0 && (
                        <tr>
                          <td colSpan={5} className="px-5 py-8 text-center text-slate-400">
                            All offline operations are synchronized. Outbox is currently empty.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {syncResults && (
                  <div className="bg-slate-950 text-slate-200 p-4 rounded-xl text-xs space-y-2 font-mono">
                    <span className="font-semibold text-emerald-400">Last Batch Synchronization Ack:</span>
                    <pre className="overflow-x-auto">{JSON.stringify(syncResults, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}

            {/* TAB 4: CLINICAL CONFLICT REVIEW */}
            {activeTab === 'conflicts' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Clinical Conflicts Requiring Provider Resolution ({conflicts.length})
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Quarantined discrepant field observations requiring human clinical review.
                    </p>
                  </div>
                  <button
                    onClick={loadConflicts}
                    className="px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-slate-700 font-semibold text-xs hover:bg-slate-50 transition"
                  >
                    Refresh Queue
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {conflicts.map((conf) => {
                    const isSelected = selectedConflict?.conflict_id === conf.conflict_id;
                    return (
                      <div
                        key={conf.conflict_id}
                        onClick={() => setSelectedConflict(conf)}
                        className={`p-4 rounded-xl border cursor-pointer transition-all space-y-2 ${
                          isSelected
                            ? 'border-rose-500 bg-rose-50/30 shadow-xs'
                            : 'border-slate-200/90 bg-white hover:border-slate-300 shadow-2xs'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-xs font-bold text-rose-700">{conf.conflict_id}</span>
                          <StatusBadge status={conf.resolution_status} size="sm" />
                        </div>
                        <div className="text-xs text-slate-800 font-medium">{conf.conflict_reason}</div>
                        <div className="pt-2 border-t border-slate-100 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="font-semibold block text-slate-700">Client Field Observation:</span>
                            <pre className="bg-[#F8FAF9] p-2 rounded text-[10px] font-mono mt-1 overflow-x-auto">
                              {JSON.stringify(conf.client_payload, null, 2)}
                            </pre>
                          </div>
                          <div>
                            <span className="font-semibold block text-slate-700">Hospital Verified Observation:</span>
                            <pre className="bg-[#F8FAF9] p-2 rounded text-[10px] font-mono mt-1 overflow-x-auto">
                              {JSON.stringify(conf.server_payload, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {conflicts.length === 0 && (
                    <div className="col-span-2">
                      <EmptyState
                        title="No Outstanding Clinical Conflicts"
                        description="All field observations and hospital clinical records are concordant."
                      />
                    </div>
                  )}
                </div>

                {/* Conflict Resolution Action Card */}
                {selectedConflict && (
                  <div className="bg-white p-6 rounded-2xl border-2 border-rose-200 shadow-sm space-y-4">
                    <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Resolve Clinical Conflict: {selectedConflict.conflict_id}
                    </h4>
                    <p className="text-xs text-slate-600">
                      As a licensed healthcare provider, specify whether to accept the field measurement or preserve the hospital observation.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-semibold text-slate-700 mb-1">
                          Resolution Action
                        </label>
                        <select
                          value={resolutionChoice}
                          onChange={(e) => setResolutionChoice(e.target.value as any)}
                          className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                        >
                          <option value="KEEP_SERVER">KEEP_SERVER (Preserve verified hospital observation)</option>
                          <option value="ACCEPT_CLIENT">ACCEPT_CLIENT (Accept CHW field measurement)</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-xs font-semibold text-slate-700 mb-1">
                          Mandatory Clinical Rationale Notes
                        </label>
                        <input
                          type="text"
                          placeholder="Document clinical reason for resolution decision..."
                          value={resolutionNotes}
                          onChange={(e) => setResolutionNotes(e.target.value)}
                          className="w-full text-xs p-2.5 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-forest-600"
                        />
                      </div>
                    </div>

                    <div className="flex justify-end gap-2.5 pt-2 border-t border-slate-100">
                      <button
                        onClick={() => setSelectedConflict(null)}
                        className="px-4 py-2 border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 hover:bg-slate-50 transition"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleResolveConflict}
                        className="px-5 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-lg text-xs font-semibold shadow-xs transition"
                      >
                        Confirm Resolution & Sign Audit Log
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

export default function ChwWorkspacePage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs text-slate-500">Loading CHW Workspace...</div>}>
      <ChwWorkspaceContent />
    </Suspense>
  );
}
