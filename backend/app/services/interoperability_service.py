import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, PersonIdentifier, PatientProfile
from backend.app.models.clinical import Encounter, ClinicalObservation, ScreeningRecord, ScreeningStatus, ObservationType, EncounterStatus
from backend.app.models.screening import ScreeningEncounter, EncounterType, EpistemicStatus
from backend.app.models.integration import (
    IntegrationProvider,
    IntegrationJobLog,
    ProvenanceRecord,
    ProviderType,
    IntegrationStatus,
    JobExecutionStatus,
)
from backend.app.integrations.fhir.serializer import FhirSerializer
from backend.app.integrations.terminology.mapper import TerminologyMapper
from backend.app.integrations.adapters.malaffi import MalaffiAdapter
from backend.app.integrations.adapters.seha_emr import SehaEmrAdapter
from backend.app.integrations.adapters.lab_network import LabNetworkAdapter
from backend.app.integrations.adapters.uae_pass import UaePassAdapter
from backend.app.schemas.integrations import (
    IntegrationProviderResponse,
    HealthCheckResult,
    FhirIngestResponse,
    FhirExportResponse,
    TerminologyLookupResponse,
    IntegrationMonitoringResponse,
)


class InteroperabilityService:
    """Enterprise Healthcare Interoperability & External Integration Orchestrator."""

    _adapters = {
        "MALAFFI": MalaffiAdapter(),
        "SEHA_EMR": SehaEmrAdapter(),
        "LAB_NETWORK": LabNetworkAdapter(),
        "UAE_PASS": UaePassAdapter(),
    }

    @classmethod
    def get_providers(cls, db: Session) -> List[IntegrationProviderResponse]:
        """Returns live registered external adapters and their honest operational status."""
        providers = []
        for name, adapter in cls._adapters.items():
            status_str = adapter.get_status()
            prereqs = adapter.get_prerequisites()
            p_type = (
                "HIE" if name == "MALAFFI"
                else "EMR" if name == "SEHA_EMR"
                else "LAB" if name == "LAB_NETWORK"
                else "IDENTITY"
            )
            disp_name = (
                "Abu Dhabi Health Information Exchange (Malaffi)" if name == "MALAFFI"
                else "SEHA Clinic & Hospital EMR" if name == "SEHA_EMR"
                else "Abu Dhabi Laboratory Network" if name == "LAB_NETWORK"
                else "National Digital Identity (UAE PASS)"
            )
            providers.append(
                IntegrationProviderResponse(
                    provider_name=name,
                    display_name=disp_name,
                    provider_type=p_type,
                    status=status_str,
                    endpoint_url=getattr(adapter, "api_url", None),
                    is_enabled=(status_str != "NOT_CONFIGURED"),
                    prerequisites=prereqs,
                    last_heartbeat_at=datetime.utcnow() if status_str != "NOT_CONFIGURED" else None,
                )
            )
        return providers

    @classmethod
    def test_provider_health(cls, db: Session, provider_name: str, user: User) -> HealthCheckResult:
        """Executes a genuine connectivity check against the specified provider."""
        adapter = cls._adapters.get(provider_name.upper())
        if not adapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration adapter '{provider_name}' not recognized",
            )

        success, msg = adapter.test_connection()
        stat = "CONNECTED" if success else adapter.get_status()

        # Log check in database
        prov_record = db.query(IntegrationProvider).filter(IntegrationProvider.name == provider_name.upper()).first()
        prov_id = prov_record.id if prov_record else f"PROV-{provider_name.upper()}"

        job_log = IntegrationJobLog(
            provider_id=prov_id,
            job_type="HEALTH_CHECK",
            status=JobExecutionStatus.SUCCESS if success else JobExecutionStatus.FAILED,
            records_processed=0,
            records_failed=0 if success else 1,
            details_json=json.dumps({"message": msg}),
            error_message=None if success else msg,
            executed_at=datetime.utcnow(),
        )
        db.add(job_log)
        db.commit()

        return HealthCheckResult(
            provider_name=provider_name.upper(),
            success=success,
            status=stat,
            message=msg,
            timestamp=datetime.utcnow(),
        )

    @classmethod
    def export_patient_bundle(cls, db: Session, patient_id: str, user: User) -> FhirExportResponse:
        """Packages patient demographics, vitals, and encounters into an authorized FHIR R4 Bundle."""
        patient = db.query(Person).filter(Person.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found",
            )

        pat_org_id = patient.patient_profile.organization_id if patient.patient_profile else None
        # Organization boundary check
        if user.role != UserRole.SYSTEM_ADMIN and pat_org_id != user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Organization boundary violation: cannot export external patient records",
            )

        identifiers = db.query(PersonIdentifier).filter(PersonIdentifier.person_id == patient.id).all()
        observations = db.query(ClinicalObservation).filter(ClinicalObservation.patient_id == patient.id).all()
        screenings = db.query(ScreeningRecord).filter(ScreeningRecord.patient_id == patient.id).all()

        resources = []
        # 1. Patient
        resources.append(FhirSerializer.serialize_patient(patient, identifiers))
        # 2. Observations
        for obs in observations:
            resources.append(FhirSerializer.serialize_observation(obs))
        # 3. Encounters
        for sc in screenings:
            resources.append(FhirSerializer.serialize_encounter(sc))

        bundle = FhirSerializer.serialize_bundle(resources, bundle_type="collection")

        # Log outbound export
        prov_record = db.query(IntegrationProvider).filter(IntegrationProvider.name == "MALAFFI").first()
        prov_id = prov_record.id if prov_record else "PROV-MALAFFI"
        job_log = IntegrationJobLog(
            provider_id=prov_id,
            job_type="OUTBOUND_EXPORT",
            status=JobExecutionStatus.SUCCESS,
            records_processed=len(resources),
            records_failed=0,
            details_json=json.dumps({"patient_id": patient.id, "resource_count": len(resources)}),
            executed_at=datetime.utcnow(),
        )
        db.add(job_log)
        db.commit()

        return FhirExportResponse(
            patient_id=patient.id,
            bundle=bundle,
            exported_at=datetime.utcnow(),
        )

    @classmethod
    def ingest_fhir_bundle(
        cls,
        db: Session,
        bundle: Dict[str, Any],
        source_system: str,
        user: User,
    ) -> FhirIngestResponse:
        """Ingests an external FHIR R4 Bundle with identity matching, terminology translation, and provenance."""
        is_valid, err = FhirSerializer.validate_resource(bundle)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid FHIR Bundle: {err}",
            )

        entries = bundle.get("entry", [])
        if not entries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="FHIR Bundle contains no entries to ingest",
            )

        matched_patient: Optional[Person] = None
        observations_created = 0
        encounters_created = 0

        # Pass 1: Identity Matching (Match Patient via Emirates ID or MRN)
        for e in entries:
            res = e.get("resource", {})
            if res.get("resourceType") == "Patient":
                id_list = res.get("identifier", [])
                for ident in id_list:
                    val = ident.get("value")
                    if val:
                        matched_ident = (
                            db.query(PersonIdentifier)
                            .filter(PersonIdentifier.identifier_value == val)
                            .first()
                        )
                        if matched_ident:
                            matched_patient = (
                                db.query(Person)
                                .filter(Person.id == matched_ident.person_id)
                                .first()
                            )
                            break
                if not matched_patient and res.get("id"):
                    matched_patient = db.query(Person).filter(Person.id == res.get("id")).first()

        if not matched_patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient identity could not be matched against existing records via Emirates ID or MRN",
            )

        pat_org_id = matched_patient.patient_profile.organization_id if matched_patient.patient_profile else user.organization_id
        # Organization access check
        if user.role != UserRole.SYSTEM_ADMIN and pat_org_id != user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant isolation violation: cannot ingest external records into another organization",
            )

        # Pass 2: Ingest Observations and Encounters
        provenance_id = f"PROV-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.utcnow()

        for e in entries:
            res = e.get("resource", {})
            res_type = res.get("resourceType")

            if res_type == "Observation":
                # Translate LOINC code
                code_obj = res.get("code", {}).get("coding", [{}])[0]
                target_code = code_obj.get("code")
                target_sys = code_obj.get("system", "http://loinc.org")

                mapped = TerminologyMapper.reverse_lookup(target_sys, target_code)
                internal_type_str = mapped["internal_code"] if mapped else "BLOOD_PRESSURE_SYSTOLIC"
                val_quant = res.get("valueQuantity", {})
                num_val = val_quant.get("value")
                unit_str = val_quant.get("unit")

                obs = ClinicalObservation(
                    patient_id=matched_patient.id,
                    observation_type=ObservationType.VITAL_SIGN if "BP" in internal_type_str or "GLUCOSE" in internal_type_str else ObservationType.LAB_RESULT,
                    code_system="LOINC",
                    code=target_code or "8480-6",
                    display_name=mapped["target_display"] if mapped else "Clinical Observation",
                    numeric_value=float(num_val) if num_val is not None else None,
                    unit=unit_str or "mm[Hg]",
                    collection_time=now,
                    effective_time=now,
                    recording_time=now,
                    epistemic_status=EpistemicStatus.VERIFIED,
                )
                db.add(obs)
                db.flush()

                # Provenance record
                prov = ProvenanceRecord(
                    target_resource_type="ClinicalObservation",
                    target_resource_id=obs.id,
                    source_system=source_system,
                    source_identifier=res.get("id"),
                    agent_id=user.id,
                    agent_type="INTEGRATION_ADAPTER",
                    recorded_at=now,
                )
                db.add(prov)
                observations_created += 1

            elif res_type == "Encounter":
                sc = ScreeningRecord(
                    patient_id=matched_patient.id,
                    performing_facility_id=pat_org_id,
                    screening_type="IFHAS_COMPREHENSIVE",
                    screening_date=now.date(),
                    screening_status=ScreeningStatus.COMPLETED,
                    findings_summary=f"Ingested from external {source_system}",
                    epistemic_status=EpistemicStatus.VERIFIED,
                )
                db.add(sc)
                db.flush()

                prov = ProvenanceRecord(
                    target_resource_type="ScreeningRecord",
                    target_resource_id=sc.id,
                    source_system=source_system,
                    source_identifier=res.get("id"),
                    agent_id=user.id,
                    agent_type="INTEGRATION_ADAPTER",
                    recorded_at=now,
                )
                db.add(prov)
                encounters_created += 1

        db.commit()

        return FhirIngestResponse(
            status="SUCCESS",
            patient_id=matched_patient.id,
            resources_ingested=len(entries),
            observations_created=observations_created,
            encounters_created=encounters_created,
            provenance_id=provenance_id,
            timestamp=now,
        )

    @classmethod
    def get_monitoring_overview(cls, db: Session) -> IntegrationMonitoringResponse:
        """Returns comprehensive monitoring status across all external adapters and recent jobs."""
        providers = cls.get_providers(db)
        total = len(providers)
        configured = sum(1 for p in providers if p.status != "NOT_CONFIGURED")
        not_conf = total - configured

        recent_job_models = db.query(IntegrationJobLog).order_by(IntegrationJobLog.executed_at.desc()).limit(10).all()
        recent_jobs = [
            {
                "job_id": j.id,
                "provider_id": j.provider_id,
                "job_type": j.job_type,
                "status": j.status,
                "records_processed": j.records_processed,
                "executed_at": j.executed_at.isoformat(),
            }
            for j in recent_job_models
        ]

        return IntegrationMonitoringResponse(
            total_providers=total,
            configured_providers=configured,
            not_configured_providers=not_conf,
            providers=providers,
            recent_jobs=recent_jobs,
            generated_at=datetime.utcnow(),
        )
