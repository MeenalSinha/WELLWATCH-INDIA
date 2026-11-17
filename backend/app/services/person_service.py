import hashlib
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc

from backend.app.models.person import (
    Person,
    PersonIdentifier,
    PatientProfile,
    Gender,
    RecordStatus,
    PopulationClassification,
    CompletenessStatus,
    IdentifierNamespace,
    IdentifierVerificationStatus,
)
from backend.app.models.clinical import (
    Encounter,
    ClinicalObservation,
    ClinicalCondition,
    FamilyHistory,
    ScreeningRecord,
    ObservationType,
    EncounterType,
    EncounterStatus,
    FamilyRelationship,
    ScreeningStatus,
)
from backend.app.models.provenance import (
    DataSource,
    DataProvenance,
    SourceType,
    TrustLevel,
    ProvenanceVerificationStatus,
)
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.screening import EpistemicStatus
from backend.app.models.user import User, UserRole
from backend.app.schemas.person import TimelineEvent
from backend.app.services.clinical_validation_service import ClinicalValidationService
from backend.app.services.data_quality_service import DataQualityService


class PersonService:
    @staticmethod
    def hash_identifier(value: str) -> str:
        """Computes SHA-256 hash for secure zero-knowledge identifier querying."""
        return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()

    @classmethod
    def register_person(
        cls,
        db: Session,
        first_name: str,
        last_name: str,
        date_of_birth: date,
        gender: Gender,
        organization_id: str,
        emirate: str = "Abu Dhabi",
        preferred_language: str = "ar",
        nationality: str = "ARE",
        district: Optional[str] = None,
        user_id: Optional[str] = None,
        primary_identifier: Optional[Tuple[IdentifierNamespace, str]] = None,
        guardian_user_id: Optional[str] = None,
        relationship_to_guardian: Optional[str] = None,
        data_source_id: Optional[str] = None,
    ) -> Person:
        """
        Registers a new healthcare subject (Person) and initializes their longitudinal PatientProfile.
        Enforces deduplication on national/clinical identifiers.
        """
        # Deduplication check
        if primary_identifier:
            ns, val = primary_identifier
            existing_id = db.query(PersonIdentifier).filter(
                PersonIdentifier.namespace == ns,
                PersonIdentifier.identifier_value == val.strip()
            ).first()
            if existing_id:
                raise ValueError(f"Duplicate registration rejected: Identifier {val} is already registered under namespace {ns.value}.")

        # Determine population classification based on age
        today = date.today()
        age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
        if age < 18:
            classification = PopulationClassification.PEDIATRIC
        elif age >= 65:
            classification = PopulationClassification.GERIATRIC
        else:
            classification = PopulationClassification.ADULT

        # Create Person
        person = Person(
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            date_of_birth=date_of_birth,
            gender=gender,
            nationality=nationality,
            emirate=emirate,
            district=district,
            preferred_language=preferred_language,
            user_id=user_id,
            record_status=RecordStatus.ACTIVE,
        )
        db.add(person)
        db.flush()

        # Add primary identifier if supplied
        if primary_identifier:
            ns, val = primary_identifier
            ident = PersonIdentifier(
                person_id=person.id,
                namespace=ns,
                issuing_authority="ICP UAE" if ns == IdentifierNamespace.EMIRATES_ID else "DOH Abu Dhabi",
                identifier_value=val.strip(),
                identifier_hash=cls.hash_identifier(val),
                verification_status=IdentifierVerificationStatus.VERIFIED if ns == IdentifierNamespace.INTERNAL_UUID else IdentifierVerificationStatus.UNVERIFIED,
                valid_from=date.today(),
            )
            db.add(ident)

        # Create PatientProfile
        profile = PatientProfile(
            person_id=person.id,
            organization_id=organization_id,
            population_classification=classification,
            completeness_score=0.0,
            completeness_status=CompletenessStatus.MINIMAL,
        )
        db.add(profile)

        # Link guardian for minors if provided
        if classification == PopulationClassification.PEDIATRIC and guardian_user_id:
            guardianship = Guardianship(
                guardian_user_id=guardian_user_id,
                dependent_user_id=guardian_user_id,  # Fallback user ID
                guardian_person_id=None,
                dependent_person_id=person.id,
                relationship_type=relationship_to_guardian or RelationshipType.LEGAL_GUARDIAN.value,
                verification_status=GuardianshipStatus.VERIFIED.value,
            )
            db.add(guardianship)

        db.commit()
        db.refresh(person)
        return person

    @classmethod
    def list_persons(
        cls,
        db: Session,
        requesting_role: str,
        requesting_org_id: Optional[str],
        classification_filter: Optional[str] = None,
        search_query: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Person], int]:
        """
        Lists persons with organization boundary enforcement, role-based visibility, and search.
        """
        if requesting_role == UserRole.SYSTEM_ADMIN.value:
            raise PermissionError("System Administrators do not have access to patient directories.")

        query = db.query(Person).join(PatientProfile)

        # Facility tenant isolation
        if requesting_role in [UserRole.HEALTHCARE_PROVIDER.value, UserRole.COMMUNITY_HEALTH_WORKER.value, UserRole.CARE_COORDINATOR.value]:
            if requesting_org_id:
                query = query.filter(PatientProfile.organization_id == requesting_org_id)

        # Classification filter
        if classification_filter:
            query = query.filter(PatientProfile.population_classification == classification_filter)

        # Search filter
        if search_query:
            q = f"%{search_query.strip().lower()}%"
            query = query.filter(
                or_(
                    Person.first_name.ilike(q),
                    Person.last_name.ilike(q),
                    Person.district.ilike(q)
                )
            )

        total = query.count()
        items = query.order_by(desc(Person.created_at)).offset(skip).limit(limit).all()
        return items, total

    @classmethod
    def get_person_by_id(
        cls,
        db: Session,
        person_id: str,
        requesting_user_id: str,
        requesting_role: str,
        requesting_org_id: Optional[str]
    ) -> Optional[Person]:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            return None

        if requesting_role == UserRole.SYSTEM_ADMIN.value:
            raise PermissionError("System Administrators do not have access to individual patient clinical records.")

        if requesting_role == UserRole.INDIVIDUAL.value:
            if person.user_id != requesting_user_id:
                raise PermissionError("Access denied: You can only access your own health record.")
            return person

        if requesting_role == UserRole.PARENT_GUARDIAN.value:
            is_guardian = db.query(Guardianship).filter(
                Guardianship.guardian_user_id == requesting_user_id,
                or_(
                    Guardianship.dependent_person_id == person.id,
                    and_(
                        Guardianship.dependent_user_id == person.user_id,
                        person.user_id != None
                    )
                ),
                Guardianship.verification_status == GuardianshipStatus.VERIFIED.value
            ).first()
            if not is_guardian:
                raise PermissionError("Access denied: You are not a verified guardian for this dependent.")
            return person

        if requesting_role in [UserRole.HEALTHCARE_PROVIDER.value, UserRole.COMMUNITY_HEALTH_WORKER.value, UserRole.CARE_COORDINATOR.value]:
            if person.patient_profile and person.patient_profile.organization_id != requesting_org_id:
                raise PermissionError("Cross-organization boundary violation: Record belongs to different healthcare facility.")

        return person

    @classmethod
    def get_pediatric_profile(cls, db: Session, person: Person) -> Dict[str, Any]:
        """Gathers developmental milestones, Well-Child visits, and family context for pediatric subject."""
        # 1. Guardianships
        guardians = db.query(Guardianship).filter(
            Guardianship.dependent_person_id == person.id,
            Guardianship.verification_status == GuardianshipStatus.VERIFIED.value
        ).all()
        guardian_list = [
            {"id": g.id, "relationship": g.relationship_type, "scope": g.authorization_scope, "verified": True}
            for g in guardians
        ]

        # 2. Well-Child Screenings
        screenings = db.query(ScreeningRecord).filter(
            ScreeningRecord.patient_id == person.id
        ).order_by(desc(ScreeningRecord.screening_date)).all()
        screening_list = [
            {
                "screening_type": s.screening_type,
                "date": s.screening_date.isoformat(),
                "status": s.screening_status.value if hasattr(s.screening_status, 'value') else s.screening_status,
                "epistemic_status": s.epistemic_status.value if hasattr(s.epistemic_status, 'value') else s.epistemic_status,
                "findings": s.findings_summary
            }
            for s in screenings
        ]

        # 3. Developmental Milestones & Growth Observations
        dev_obs = db.query(ClinicalObservation).filter(
            ClinicalObservation.patient_id == person.id,
            ClinicalObservation.observation_type.in_([ObservationType.DEVELOPMENTAL, ObservationType.ANTHROPOMETRIC])
        ).order_by(desc(ClinicalObservation.collection_time)).all()
        dev_list = [
            {
                "code": o.code,
                "name": o.display_name,
                "value": o.numeric_value or o.string_value,
                "unit": o.unit,
                "date": o.collection_time.isoformat(),
                "status": o.epistemic_status.value if hasattr(o.epistemic_status, 'value') else o.epistemic_status
            }
            for o in dev_obs
        ]

        # 4. Family History
        fam = db.query(FamilyHistory).filter(FamilyHistory.patient_id == person.id).all()
        fam_list = [
            {"relationship": f.relationship_to_patient.value if hasattr(f.relationship_to_patient, 'value') else f.relationship_to_patient,
             "condition": f.condition_name, "onset_age": f.approximate_age_of_onset}
            for f in fam
        ]

        return {
            "guardians": guardian_list,
            "well_child_visits": screening_list,
            "developmental_milestones": dev_list,
            "family_history": fam_list,
            "completeness": {
                "score": person.patient_profile.completeness_score if person.patient_profile else 0.0,
                "status": person.patient_profile.completeness_status.value if person.patient_profile else "MINIMAL"
            }
        }

    @classmethod
    def get_adult_profile(cls, db: Session, person: Person) -> Dict[str, Any]:
        """Gathers latest vitals, chronic conditions, lifestyle, and IFHAS screening history."""
        # 1. Latest Vitals
        vitals_obs = db.query(ClinicalObservation).filter(
            ClinicalObservation.patient_id == person.id,
            ClinicalObservation.observation_type.in_([ObservationType.VITAL_SIGN, ObservationType.ANTHROPOMETRIC, ObservationType.LAB_RESULT])
        ).order_by(desc(ClinicalObservation.collection_time)).all()

        latest_vitals = {}
        for o in vitals_obs:
            if o.code not in latest_vitals:
                latest_vitals[o.code] = {
                    "name": o.display_name,
                    "value": o.canonical_value or o.numeric_value or o.string_value,
                    "unit": o.canonical_unit or o.unit,
                    "date": o.collection_time.isoformat(),
                    "epistemic_status": o.epistemic_status.value if hasattr(o.epistemic_status, 'value') else o.epistemic_status
                }

        # 2. Chronic Conditions
        conditions = db.query(ClinicalCondition).filter(
            ClinicalCondition.patient_id == person.id
        ).all()
        cond_list = [
            {
                "code": c.code,
                "description": c.clinical_description,
                "status": c.clinical_status.value if hasattr(c.clinical_status, 'value') else c.clinical_status,
                "verification": c.verification_status.value if hasattr(c.verification_status, 'value') else c.verification_status
            }
            for c in conditions
        ]

        # 3. Lifestyle
        lifestyle_obs = db.query(ClinicalObservation).filter(
            ClinicalObservation.patient_id == person.id,
            ClinicalObservation.observation_type == ObservationType.LIFESTYLE
        ).all()
        lifestyle = {o.code: {"name": o.display_name, "value": o.string_value or o.numeric_value} for o in lifestyle_obs}

        # 4. IFHAS Screening History
        screenings = db.query(ScreeningRecord).filter(
            ScreeningRecord.patient_id == person.id
        ).order_by(desc(ScreeningRecord.screening_date)).all()
        scr_list = [
            {
                "screening_type": s.screening_type,
                "date": s.screening_date.isoformat(),
                "status": s.screening_status.value if hasattr(s.screening_status, 'value') else s.screening_status,
                "findings": s.findings_summary
            }
            for s in screenings
        ]

        # 5. Family History
        fam = db.query(FamilyHistory).filter(FamilyHistory.patient_id == person.id).all()
        fam_list = [
            {"relationship": f.relationship_to_patient.value if hasattr(f.relationship_to_patient, 'value') else f.relationship_to_patient,
             "condition": f.condition_name, "onset_age": f.approximate_age_of_onset}
            for f in fam
        ]

        return {
            "latest_vitals": latest_vitals,
            "chronic_conditions": cond_list,
            "lifestyle_indicators": lifestyle,
            "screening_history": scr_list,
            "family_history": fam_list,
            "completeness": {
                "score": person.patient_profile.completeness_score if person.patient_profile else 0.0,
                "status": person.patient_profile.completeness_status.value if person.patient_profile else "MINIMAL"
            }
        }

    @classmethod
    def get_longitudinal_timeline(cls, db: Session, person_id: str) -> List[TimelineEvent]:
        """Aggregates all clinical events into a unified chronological timeline."""
        events: List[TimelineEvent] = []

        # Encounters
        encounters = db.query(Encounter).filter(Encounter.patient_id == person_id).all()
        for e in encounters:
            events.append(TimelineEvent(
                event_id=e.id,
                event_type="ENCOUNTER",
                title=f"Healthcare Encounter: {e.encounter_type.value if hasattr(e.encounter_type, 'value') else e.encounter_type}",
                description=e.clinical_notes or "Clinical preventive consultation",
                timestamp=e.encounter_date,
                source_name=e.facility.name if e.facility else "Healthcare Facility",
                epistemic_status=EpistemicStatus.VERIFIED,
                value_display=e.status.value if hasattr(e.status, 'value') else e.status
            ))

        # Observations
        observations = db.query(ClinicalObservation).filter(ClinicalObservation.patient_id == person_id).all()
        for o in observations:
            val_str = f"{o.canonical_value or o.numeric_value} {o.canonical_unit or o.unit}" if o.numeric_value is not None else (o.string_value or "")
            events.append(TimelineEvent(
                event_id=o.id,
                event_type="OBSERVATION",
                title=f"Clinical Measurement: {o.display_name}",
                description=f"LOINC {o.code} recorded",
                timestamp=o.collection_time,
                source_name=o.provenance.data_source.system_name if o.provenance and o.provenance.data_source else "Clinical System",
                epistemic_status=o.epistemic_status,
                value_display=val_str
            ))

        # Screenings
        screenings = db.query(ScreeningRecord).filter(ScreeningRecord.patient_id == person_id).all()
        for s in screenings:
            ts = datetime.combine(s.screening_date, datetime.min.time(), tzinfo=timezone.utc)
            events.append(TimelineEvent(
                event_id=s.id,
                event_type="SCREENING",
                title=f"Preventive Screening: {s.screening_type}",
                description=s.findings_summary or "Preventive screening package",
                timestamp=ts,
                source_name=s.facility.name if s.facility else "Screening Facility",
                epistemic_status=s.epistemic_status,
                value_display=s.screening_status.value if hasattr(s.screening_status, 'value') else s.screening_status
            ))

        # Conditions
        conditions = db.query(ClinicalCondition).filter(ClinicalCondition.patient_id == person_id).all()
        for c in conditions:
            ts = datetime.combine(c.onset_date or date.today(), datetime.min.time(), tzinfo=timezone.utc)
            events.append(TimelineEvent(
                event_id=c.id,
                event_type="CONDITION",
                title=f"Clinical Condition: {c.clinical_description}",
                description=f"ICD-10 {c.code}",
                timestamp=ts,
                source_name="Diagnosing Provider",
                epistemic_status=EpistemicStatus.VERIFIED,
                value_display=c.clinical_status.value if hasattr(c.clinical_status, 'value') else c.clinical_status
            ))

        # Sort descending by timestamp
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events

    @classmethod
    def record_clinical_observation(
        cls,
        db: Session,
        person_id: str,
        observation_type: ObservationType,
        code: str,
        display_name: str,
        numeric_value: Optional[float],
        unit: Optional[str],
        string_value: Optional[str] = None,
        encounter_id: Optional[str] = None,
        collection_time: Optional[datetime] = None,
        epistemic_status: EpistemicStatus = EpistemicStatus.VERIFIED,
        data_source_id: Optional[str] = None,
        processor_user_id: Optional[str] = None,
    ) -> ClinicalObservation:
        coll_time = collection_time or datetime.now(timezone.utc)
        
        canonical_val, canonical_unit = ClinicalValidationService.normalize_and_validate_observation(
            code=code,
            numeric_value=numeric_value,
            unit=unit
        )

        provenance = None
        if data_source_id:
            provenance = DataProvenance(
                data_source_id=data_source_id,
                collection_method="STRUCTURED_CLINICAL_ENTRY",
                verification_status=ProvenanceVerificationStatus.VERIFIED if epistemic_status == EpistemicStatus.VERIFIED else ProvenanceVerificationStatus.UNVERIFIED,
                transformation_history=f"Normalized {numeric_value} {unit} -> {canonical_val} {canonical_unit}" if unit != canonical_unit else None,
                responsible_processor_id=processor_user_id,
            )
            db.add(provenance)
            db.flush()

        obs = ClinicalObservation(
            patient_id=person_id,
            encounter_id=encounter_id,
            observation_type=observation_type,
            code_system="LOINC",
            code=code,
            display_name=display_name,
            numeric_value=numeric_value,
            string_value=string_value,
            unit=unit,
            canonical_value=canonical_val,
            canonical_unit=canonical_unit,
            collection_time=coll_time,
            epistemic_status=epistemic_status,
            provenance_id=provenance.id if provenance else None,
        )
        db.add(obs)
        
        person = db.query(Person).filter(Person.id == person_id).first()
        if person and person.patient_profile:
            all_obs = db.query(ClinicalObservation.code).filter(ClinicalObservation.patient_id == person_id).all()
            codes = [row[0] for row in all_obs] + [code]
            has_fam = db.query(FamilyHistory).filter(FamilyHistory.patient_id == person_id).count() > 0
            has_scr = db.query(ScreeningRecord).filter(ScreeningRecord.patient_id == person_id).count() > 0
            
            score, status = DataQualityService.calculate_profile_completeness(
                classification=person.patient_profile.population_classification,
                available_codes=codes,
                has_family_history=has_fam,
                has_screening_record=has_scr
            )
            person.patient_profile.completeness_score = score
            person.patient_profile.completeness_status = status

        db.commit()
        db.refresh(obs)
        return obs
