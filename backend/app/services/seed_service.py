"""
Comprehensive development and demonstration database seeding service for WELLWATCH PREVENT.
Provides verified personas matching the frontend quick-select login, realistic demographic patients
spanning the life course (pediatric well-child, adult IFHAS screening, geriatric), clinical vitals,
risk assessments, preventive recommendations, care plans, appointments, and outreach notifications.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Any
from sqlalchemy.orm import Session

from backend.app.core.security import get_password_hash
from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
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
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.provenance import (
    DataSource,
    DataProvenance,
    SourceType,
    TrustLevel,
    ConnectionStatus,
    ProvenanceVerificationStatus,
)
from backend.app.models.clinical import (
    Encounter,
    EncounterType,
    EncounterStatus,
    ClinicalObservation,
    ObservationType,
    ScreeningRecord,
    ScreeningStatus,
)
from backend.app.models.screening import EpistemicStatus
from backend.app.models.assessment import (
    PreventiveAssessment,
    AssessmentStatus,
    ClinicalRiskTier,
    HumanReviewStatus,
)
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
    PreventiveCarePlan,
    CarePlanItem,
)
from backend.app.models.navigation import (
    Appointment,
    AppointmentStatus,
    Referral,
    ReferralStatus,
    RecallTask,
    RecallStatus,
)
from backend.app.models.outreach import (
    CommunicationPreference,
    CommunicationChannel,
    OutreachAuthorizationStatus,
    OutreachTask,
    OutreachDeliveryStatus,
    InAppNotification,
)
from backend.app.models.offline import (
    DeviceRegistration,
    DeviceStatus,
    OfflineAssignment,
)

PRECOMPUTED_DEV_PASSWORD_HASH = get_password_hash("ValidPass123!")


def seed_initial_development_data(db: Session) -> Dict[str, Any]:
    """
    Seeds verified development database entities if not already present.
    Safe to run repeatedly; checks uniqueness before inserting.
    """
    now = datetime.now(timezone.utc)
    today = date.today()

    # -------------------------------------------------------------------------
    # 1. ORGANIZATIONS (Abu Dhabi Healthcare Facilities)
    # -------------------------------------------------------------------------
    orgs_data = [
        {
            "name": "SEHA Al Bateen Healthcare Center",
            "license_number": "DOH-ORG-00123",
            "organization_type": "PRIMARY_CARE_CLINIC",
            "region": "ABU_DHABI_CITY",
            "is_active": True,
        },
        {
            "name": "SEHA Al Ain Regional Hospital",
            "license_number": "DOH-ORG-00456",
            "organization_type": "HOSPITAL",
            "region": "AL_AIN",
            "is_active": True,
        },
        {
            "name": "Cleveland Clinic Abu Dhabi",
            "license_number": "DOH-ORG-00789",
            "organization_type": "HOSPITAL",
            "region": "AL_MARYAH_ISLAND",
            "is_active": True,
        },
        {
            "name": "Al Dhafra Family Medicine Center",
            "license_number": "DOH-ORG-00321",
            "organization_type": "PRIMARY_CARE_CLINIC",
            "region": "AL_DHAFRA",
            "is_active": True,
        },
    ]

    org_map = {}
    for o_data in orgs_data:
        existing = db.query(Organization).filter(Organization.license_number == o_data["license_number"]).first()
        if not existing:
            org = Organization(**o_data)
            db.add(org)
            db.commit()
            db.refresh(org)
            org_map[o_data["license_number"]] = org
        else:
            org_map[o_data["license_number"]] = existing

    primary_org = org_map["DOH-ORG-00123"]
    hospital_org = org_map["DOH-ORG-00456"]

    # -------------------------------------------------------------------------
    # 2. USERS (Test Personas with password 'ValidPass123!')
    # -------------------------------------------------------------------------
    persona_users = [
        {
            "email": "healthcare_provider@wellwatch.ae",
            "full_name": "Dr. Mariam Al Mansoori (Physician)",
            "role": UserRole.HEALTHCARE_PROVIDER.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "community_health_worker@wellwatch.ae",
            "full_name": "Fatima Al Zaabi (CHW Lead)",
            "role": UserRole.COMMUNITY_HEALTH_WORKER.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "individual@wellwatch.ae",
            "full_name": "Zayed Al Hashimi (Patient)",
            "role": UserRole.INDIVIDUAL.value,
            "organization_id": None,
        },
        {
            "email": "parent_guardian@wellwatch.ae",
            "full_name": "Amina Al Mazrouei (Mother & Guardian)",
            "role": UserRole.PARENT_GUARDIAN.value,
            "organization_id": None,
        },
        {
            "email": "care_coordinator@wellwatch.ae",
            "full_name": "Rashid Al Nuaimi (Care Coordinator)",
            "role": UserRole.CARE_COORDINATOR.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "population_health_admin@wellwatch.ae",
            "full_name": "Dr. Tariq Al Qasimi (Population Health Admin)",
            "role": UserRole.POPULATION_HEALTH_ADMIN.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "system_admin@wellwatch.ae",
            "full_name": "Sultan Al Suwaidi (System Admin)",
            "role": UserRole.SYSTEM_ADMIN.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "data_governance_officer@wellwatch.ae",
            "full_name": "Hessa Al Dhaheri (Data Governance Officer)",
            "role": UserRole.DATA_GOVERNANCE_OFFICER.value,
            "organization_id": primary_org.id,
        },
        {
            "email": "child.dependent@wellwatch.ae",
            "full_name": "Baby Zayed Al Mazrouei (Dependent)",
            "role": UserRole.INDIVIDUAL.value,
            "organization_id": None,
        },
    ]

    user_map = {}
    for u_data in persona_users:
        user = db.query(User).filter(User.email == u_data["email"]).first()
        if not user:
            user = User(
                email=u_data["email"],
                full_name=u_data["full_name"],
                password_hash=PRECOMPUTED_DEV_PASSWORD_HASH,
                role=u_data["role"],
                organization_id=u_data["organization_id"],
                is_active=True,
                is_verified=True,
                failed_login_count=0,
                locked_until=None,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            # Ensure password hash is fresh and any lockout is cleared
            user.password_hash = PRECOMPUTED_DEV_PASSWORD_HASH
            user.failed_login_count = 0
            user.locked_until = None
            user.is_active = True
            db.commit()
            db.refresh(user)
        user_map[u_data["email"]] = user

    # -------------------------------------------------------------------------
    # 3. DATA SOURCE & PROVENANCE
    # -------------------------------------------------------------------------
    data_source = db.query(DataSource).filter(DataSource.system_name == "SEHA EMR Gateway").first()
    if not data_source:
        data_source = DataSource(
            source_type=SourceType.CLINIC_EMR,
            organization_id=primary_org.id,
            system_name="SEHA EMR Gateway",
            system_version="2.4",
            trust_level=TrustLevel.ACCREDITED_CLINICAL,
            connection_status=ConnectionStatus.OPERATIONAL,
            last_sync_at=now,
        )
        db.add(data_source)
        db.commit()
        db.refresh(data_source)

    provenance = db.query(DataProvenance).first()
    if not provenance:
        provenance = DataProvenance(
            data_source_id=data_source.id,
            collection_method="DIRECT_CLINICAL_ENTRY",
            import_timestamp=now,
            verification_status=ProvenanceVerificationStatus.VERIFIED,
            responsible_processor_id=user_map["healthcare_provider@wellwatch.ae"].id,
        )
        db.add(provenance)
        db.commit()
        db.refresh(provenance)

    # -------------------------------------------------------------------------
    # 4. PATIENT PERSONS (Life-Course Cohort for Abu Dhabi)
    # -------------------------------------------------------------------------
    patients_data = [
        {
            "key": "zayed_adult",
            "first_name": "Zayed",
            "last_name": "Al Hashimi",
            "dob": today - timedelta(days=42 * 365 + 120),  # 42 years old
            "gender": Gender.MALE,
            "user_id": user_map["individual@wellwatch.ae"].id,
            "eid": "784-1984-1234567-1",
            "classification": PopulationClassification.ADULT,
            "district": "Al Bateen",
            "completeness": 88.5,
            "is_high_risk": 0,
        },
        {
            "key": "baby_zayed",
            "first_name": "Baby Zayed",
            "last_name": "Al Mazrouei",
            "dob": today - timedelta(days=18 * 30),  # 18 months old
            "gender": Gender.MALE,
            "user_id": user_map["child.dependent@wellwatch.ae"].id,
            "eid": "784-2025-9876543-2",
            "classification": PopulationClassification.PEDIATRIC,
            "district": "Al Mushrif",
            "completeness": 92.0,
            "is_high_risk": 0,
        },
        {
            "key": "amina_mother",
            "first_name": "Amina",
            "last_name": "Al Mazrouei",
            "dob": today - timedelta(days=35 * 365 + 40),  # 35 years old
            "gender": Gender.FEMALE,
            "user_id": user_map["parent_guardian@wellwatch.ae"].id,
            "eid": "784-1991-5554321-8",
            "classification": PopulationClassification.ADULT,
            "district": "Al Mushrif",
            "completeness": 85.0,
            "is_high_risk": 0,
        },
        {
            "key": "mariam_diabetic",
            "first_name": "Mariam",
            "last_name": "Al Dhaheri",
            "dob": today - timedelta(days=54 * 365 + 80),  # 54 years old
            "gender": Gender.FEMALE,
            "user_id": None,
            "eid": "784-1972-8889991-3",
            "classification": PopulationClassification.ADULT,
            "district": "Al Jimi (Al Ain)",
            "completeness": 95.0,
            "is_high_risk": 1,
        },
        {
            "key": "ahmed_elderly",
            "first_name": "Ahmed",
            "last_name": "Al Suwaidi",
            "dob": today - timedelta(days=68 * 365 + 200),  # 68 years old
            "gender": Gender.MALE,
            "user_id": None,
            "eid": "784-1958-3332211-4",
            "classification": PopulationClassification.GERIATRIC,
            "district": "Madinat Zayed (Al Dhafra)",
            "completeness": 90.0,
            "is_high_risk": 1,
        },
        {
            "key": "fatima_young_adult",
            "first_name": "Fatima",
            "last_name": "Al Mansoori",
            "dob": today - timedelta(days=28 * 365 + 10),  # 28 years old
            "gender": Gender.FEMALE,
            "user_id": None,
            "eid": "784-1998-4445566-7",
            "classification": PopulationClassification.ADULT,
            "district": "Khalifa City",
            "completeness": 80.0,
            "is_high_risk": 0,
        },
    ]

    person_map = {}
    for p_info in patients_data:
        person = None
        # Check by Emirates ID first
        existing_ident = db.query(PersonIdentifier).filter(
            PersonIdentifier.namespace == IdentifierNamespace.EMIRATES_ID,
            PersonIdentifier.identifier_value == p_info["eid"],
        ).first()

        if existing_ident:
            person = existing_ident.person
        elif p_info["user_id"]:
            person = db.query(Person).filter(Person.user_id == p_info["user_id"]).first()

        if not person:
            person = Person(
                first_name=p_info["first_name"],
                last_name=p_info["last_name"],
                date_of_birth=p_info["dob"],
                gender=p_info["gender"],
                preferred_language="ar",
                nationality="ARE",
                emirate="Abu Dhabi",
                district=p_info["district"],
                record_status=RecordStatus.ACTIVE,
                user_id=p_info["user_id"],
            )
            db.add(person)
            db.commit()
            db.refresh(person)

            # Emirates ID Identifier
            identifier = PersonIdentifier(
                person_id=person.id,
                namespace=IdentifierNamespace.EMIRATES_ID,
                issuing_authority="ICP UAE",
                identifier_value=p_info["eid"],
                verification_status=IdentifierVerificationStatus.VERIFIED,
            )
            db.add(identifier)

            # Patient Profile
            profile = PatientProfile(
                person_id=person.id,
                organization_id=primary_org.id,
                population_classification=p_info["classification"],
                completeness_score=p_info["completeness"],
                completeness_status=CompletenessStatus.COMPLETE if p_info["completeness"] > 90 else CompletenessStatus.PARTIALLY_COMPLETE,
                last_assessment_date=now - timedelta(days=7),
                primary_facility_name=primary_org.name,
                is_high_risk=p_info["is_high_risk"],
            )
            db.add(profile)

            # Communication Preference
            comm_pref = CommunicationPreference(
                person_id=person.id,
                preferred_language="ar",
                preferred_channel=CommunicationChannel.IN_APP,
                consent_opt_in=True,
                authorization_status=OutreachAuthorizationStatus.AUTHORIZED,
                guardian_proxy_required=True if p_info["classification"] == PopulationClassification.PEDIATRIC else False,
            )
            db.add(comm_pref)
            db.commit()

        person_map[p_info["key"]] = person

    # -------------------------------------------------------------------------
    # 5. GUARDIANSHIP (Mother Amina -> Baby Zayed)
    # -------------------------------------------------------------------------
    existing_guardianship = db.query(Guardianship).filter(
        Guardianship.guardian_user_id == user_map["parent_guardian@wellwatch.ae"].id,
        Guardianship.dependent_user_id == user_map["child.dependent@wellwatch.ae"].id,
    ).first()

    if not existing_guardianship:
        guardianship = Guardianship(
            guardian_user_id=user_map["parent_guardian@wellwatch.ae"].id,
            dependent_user_id=user_map["child.dependent@wellwatch.ae"].id,
            guardian_person_id=person_map["amina_mother"].id,
            dependent_person_id=person_map["baby_zayed"].id,
            relationship_type=RelationshipType.MOTHER.value,
            verification_status=GuardianshipStatus.VERIFIED.value,
            verification_evidence_ref="ICP-FAMILY-BOOK-00912",
            verified_by_user_id=user_map["healthcare_provider@wellwatch.ae"].id,
            authorization_scope="FULL_PREVENTIVE_HEALTHCARE_PROXY",
            valid_from=now - timedelta(days=365),
        )
        db.add(guardianship)
        db.commit()

    # -------------------------------------------------------------------------
    # 6. ENCOUNTERS & CLINICAL OBSERVATIONS
    # -------------------------------------------------------------------------
    # Encounter for Zayed (Adult IFHAS)
    zayed = person_map["zayed_adult"]
    enc_zayed = db.query(Encounter).filter(Encounter.patient_id == zayed.id).first()
    if not enc_zayed:
        enc_zayed = Encounter(
            patient_id=zayed.id,
            facility_id=primary_org.id,
            provider_user_id=user_map["healthcare_provider@wellwatch.ae"].id,
            encounter_type=EncounterType.IFHAS_COMPREHENSIVE_SCREENING,
            encounter_date=now - timedelta(days=14),
            status=EncounterStatus.COMPLETED,
            clinical_notes="Annual IFHAS comprehensive screening encounter. Patient in good overall health; mild borderline blood pressure.",
            provenance_id=provenance.id,
        )
        db.add(enc_zayed)
        db.commit()
        db.refresh(enc_zayed)

        obs_zayed = [
            # Systolic BP
            ClinicalObservation(
                patient_id=zayed.id,
                encounter_id=enc_zayed.id,
                observation_type=ObservationType.VITAL_SIGN,
                code="8480-6",
                display_name="Systolic Blood Pressure",
                numeric_value=128.0,
                unit="mmHg",
                canonical_value=128.0,
                canonical_unit="mmHg",
                collection_time=now - timedelta(days=14),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
            # Diastolic BP
            ClinicalObservation(
                patient_id=zayed.id,
                encounter_id=enc_zayed.id,
                observation_type=ObservationType.VITAL_SIGN,
                code="8462-4",
                display_name="Diastolic Blood Pressure",
                numeric_value=82.0,
                unit="mmHg",
                canonical_value=82.0,
                canonical_unit="mmHg",
                collection_time=now - timedelta(days=14),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
            # Fasting Blood Glucose
            ClinicalObservation(
                patient_id=zayed.id,
                encounter_id=enc_zayed.id,
                observation_type=ObservationType.LAB_RESULT,
                code="1558-6",
                display_name="Fasting Blood Glucose",
                numeric_value=104.0,
                unit="mg/dL",
                canonical_value=5.77,
                canonical_unit="mmol/L",
                collection_time=now - timedelta(days=14),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
            # BMI
            ClinicalObservation(
                patient_id=zayed.id,
                encounter_id=enc_zayed.id,
                observation_type=ObservationType.ANTHROPOMETRIC,
                code="39156-5",
                display_name="Body Mass Index (BMI)",
                numeric_value=27.6,
                unit="kg/m2",
                canonical_value=27.6,
                canonical_unit="kg/m2",
                collection_time=now - timedelta(days=14),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
        ]
        db.add_all(obs_zayed)
        db.commit()

    # Encounter for Baby Zayed (Well-Child 18M)
    baby = person_map["baby_zayed"]
    enc_baby = db.query(Encounter).filter(Encounter.patient_id == baby.id).first()
    if not enc_baby:
        enc_baby = Encounter(
            patient_id=baby.id,
            facility_id=primary_org.id,
            provider_user_id=user_map["healthcare_provider@wellwatch.ae"].id,
            encounter_type=EncounterType.WELL_CHILD_VISIT,
            encounter_date=now - timedelta(days=7),
            status=EncounterStatus.COMPLETED,
            clinical_notes="18-Month Well-Child visit. Normal developmental progress; gross motor and speech milestones met. MMR booster recommended.",
            provenance_id=provenance.id,
        )
        db.add(enc_baby)
        db.commit()
        db.refresh(enc_baby)

        obs_baby = [
            ClinicalObservation(
                patient_id=baby.id,
                encounter_id=enc_baby.id,
                observation_type=ObservationType.ANTHROPOMETRIC,
                code="8302-2",
                display_name="Body Height",
                numeric_value=83.5,
                unit="cm",
                canonical_value=83.5,
                canonical_unit="cm",
                collection_time=now - timedelta(days=7),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
            ClinicalObservation(
                patient_id=baby.id,
                encounter_id=enc_baby.id,
                observation_type=ObservationType.ANTHROPOMETRIC,
                code="29463-7",
                display_name="Body Weight",
                numeric_value=11.4,
                unit="kg",
                canonical_value=11.4,
                canonical_unit="kg",
                collection_time=now - timedelta(days=7),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
            ClinicalObservation(
                patient_id=baby.id,
                encounter_id=enc_baby.id,
                observation_type=ObservationType.DEVELOPMENTAL,
                code="72172-0",
                display_name="Developmental Milestones Screen (18M)",
                string_value="MILESTONES_MET",
                collection_time=now - timedelta(days=7),
                epistemic_status=EpistemicStatus.VERIFIED,
                provenance_id=provenance.id,
            ),
        ]
        db.add_all(obs_baby)
        db.commit()

    # -------------------------------------------------------------------------
    # 7. SCREENING RECORDS
    # -------------------------------------------------------------------------
    existing_screening = db.query(ScreeningRecord).filter(ScreeningRecord.patient_id == zayed.id).first()
    if not existing_screening:
        screening = ScreeningRecord(
            patient_id=zayed.id,
            encounter_id=enc_zayed.id,
            screening_type="IFHAS_ANNUAL_CARDIOVASCULAR",
            screening_date=today - timedelta(days=14),
            screening_status=ScreeningStatus.COMPLETED,
            completion_evidence_ref="LAB-IFHAS-2026-9921",
            performing_facility_id=primary_org.id,
            findings_summary="Mild pre-hypertension; borderline fasting glucose. Lifestyle modification advised.",
            epistemic_status=EpistemicStatus.VERIFIED,
            provenance_id=provenance.id,
        )
        db.add(screening)
        db.commit()

    # -------------------------------------------------------------------------
    # 8. PREVENTIVE ASSESSMENTS & RECOMMENDATIONS
    # -------------------------------------------------------------------------
    existing_assessment = db.query(PreventiveAssessment).filter(PreventiveAssessment.patient_id == zayed.id).first()
    if not existing_assessment:
        assessment = PreventiveAssessment(
            patient_id=zayed.id,
            assessment_type="IFHAS_PERIODIC_COMPREHENSIVE_ASSESSMENT",
            population_category="ADULT",
            pathway_id="DOH_IFHAS_ADULT_PREVENTION_V1",
            pathway_version="1.0.0",
            assessment_status=AssessmentStatus.COMPLETED,
            clinical_risk_tier=ClinicalRiskTier.MODERATE_RISK,
            eligibility_summary='{"is_eligible": true, "program": "DoH IFHAS Adult Comprehensive Screening"}',
            care_gaps_summary='{"unmet_gaps": ["LIPID_PANEL_FASTING", "ANNUAL_DENTAL_EXAM"]}',
            contributing_factors='{"systolic_bp": 128, "fasting_glucose": 104, "bmi": 27.6}',
            data_quality_limitations='{"missing_fields": [], "completeness_pct": 92.5}',
            human_review_status=HumanReviewStatus.APPROVED,
            reviewed_by_user_id=user_map["healthcare_provider@wellwatch.ae"].id,
            reviewed_at=now - timedelta(days=10),
            recommended_action="Initiate lifestyle modification and schedule follow-up lipid screening within 90 days.",
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
    else:
        assessment = existing_assessment

    # Decoupled Preventive Recommendations for Zayed
    existing_recs = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.person_id == zayed.id).all()
    if not existing_recs:
        rec1 = PreventiveRecommendation(
            organization_id=primary_org.id,
            person_id=zayed.id,
            assessment_id=assessment.id,
            pathway_id="DOH_IFHAS_ADULT_PREVENTION_V1",
            care_gap_id="LIPID_PANEL_FASTING",
            category=RecommendationCategory.PREVENTIVE_LABORATORY_SCREENING,
            priority=RecommendationPriority.MEDIUM,
            lifecycle_status=RecommendationLifecycleStatus.APPROVED,
            recommended_action="Comprehensive Lipid Panel & Fasting Glucose Screening",
            clinical_rationale="Recommended annual lipid profile and glucose re-check to monitor cardiometabolic trajectory.",
            supporting_evidence={"systolic_bp": 128, "fasting_glucose": 104},
            source_guideline="DoH Abu Dhabi Adult Preventive Screening Standards (IFHAS)",
            uncertainty_status="CONFIRMED",
            rule_id="DOH-REC-LIPID-001",
            rule_version="1.0.0",
            requires_human_review=False,
            target_date=now + timedelta(days=76),
        )

        rec2 = PreventiveRecommendation(
            organization_id=primary_org.id,
            person_id=zayed.id,
            assessment_id=assessment.id,
            pathway_id="DOH_IFHAS_ADULT_PREVENTION_V1",
            care_gap_id="LIFESTYLE_NUTRITION_EXERCISE",
            category=RecommendationCategory.LIFESTYLE_WELLNESS,
            priority=RecommendationPriority.ROUTINE,
            lifecycle_status=RecommendationLifecycleStatus.OFFERED,
            recommended_action="Cardiometabolic Lifestyle & Dietary Optimization Consultation",
            clinical_rationale="Guidance on Mediterranean diet patterns and 150 min/week aerobic activity to lower borderline blood pressure.",
            supporting_evidence={"systolic_bp": 128, "bmi": 27.6},
            source_guideline="DoH Primary Care Prevention Guidelines 2026",
            uncertainty_status="CONFIRMED",
            rule_id="DOH-REC-LIFESTYLE-002",
            rule_version="1.0.0",
            requires_human_review=False,
            target_date=now + timedelta(days=46),
        )

        db.add_all([rec1, rec2])
        db.commit()
        db.refresh(rec1)
        db.refresh(rec2)
    else:
        rec1 = existing_recs[0]
        rec2 = existing_recs[1] if len(existing_recs) > 1 else existing_recs[0]

    # Additional Recommendations for Baby Zayed (Pediatric Well-Child & Developmental)
    existing_baby_recs = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.person_id == baby.id).all()
    if not existing_baby_recs:
        rec_baby1 = PreventiveRecommendation(
            organization_id=primary_org.id,
            person_id=baby.id,
            pathway_id="DOH_PEDIATRIC_WELL_CHILD_V1",
            care_gap_id="WELL_CHILD_18M_MILESTONE",
            category=RecommendationCategory.DEVELOPMENTAL_SCREENING,
            priority=RecommendationPriority.HIGH,
            lifecycle_status=RecommendationLifecycleStatus.APPROVED,
            recommended_action="18-Month ASQ-3 & M-CHAT-R Developmental Screening",
            clinical_rationale="Routine developmental and autism spectrum screening at 18 months per DoH Child Health Standards.",
            supporting_evidence={"milestone_screening_at_12m": "COMPLETED", "age_months": 18},
            source_guideline="DoH Abu Dhabi Child Health & Developmental Guidelines 2026",
            uncertainty_status="CONFIRMED",
            rule_id="DOH-REC-PEDIATRIC-18M",
            rule_version="1.0.0",
            requires_human_review=False,
            target_date=now + timedelta(days=30),
        )
        rec_baby2 = PreventiveRecommendation(
            organization_id=primary_org.id,
            person_id=baby.id,
            pathway_id="DOH_PEDIATRIC_VACCINATION_V1",
            care_gap_id="VACCINE_MMR_VARICELLA_DOSE2",
            category=RecommendationCategory.WELL_CHILD_VISIT,
            priority=RecommendationPriority.CRITICAL,
            lifecycle_status=RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW,
            recommended_action="Administer MMR and Varicella Dose 2 Pediatric Immunizations",
            clinical_rationale="National Immunization Schedule mandate for toddlers aged 18 months.",
            supporting_evidence={"previous_vaccine_dose1": "COMPLETED"},
            source_guideline="UAE National Immunization Program / DoH Schedule",
            uncertainty_status="CONFIRMED",
            rule_id="DOH-REC-VACCINE-18M",
            rule_version="1.0.0",
            requires_human_review=True,
            target_date=now + timedelta(days=14),
        )
        db.add_all([rec_baby1, rec_baby2])
        db.commit()

    # -------------------------------------------------------------------------
    # 9. PREVENTIVE CARE PLAN & ITEMS
    # -------------------------------------------------------------------------
    existing_care_plan = db.query(PreventiveCarePlan).filter(PreventiveCarePlan.person_id == zayed.id).first()
    if not existing_care_plan:
        care_plan = PreventiveCarePlan(
            organization_id=primary_org.id,
            person_id=zayed.id,
            plan_name="Zayed Al Hashimi — 2026 IFHAS Precision Prevention Plan",
            pathway_id="DOH_IFHAS_ADULT_PREVENTION_V1",
            status="ACTIVE",
            goals=[
                {"goal": "Maintain blood pressure below 120/80 mmHg", "target_date": (today + timedelta(days=180)).isoformat()},
                {"goal": "Target BMI under 25.0 kg/m² through sustainable activity", "target_date": (today + timedelta(days=365)).isoformat()},
            ],
            responsible_org_id=primary_org.id,
            assigned_care_team=[
                {"user_id": user_map["healthcare_provider@wellwatch.ae"].id, "role": "LEAD_PHYSICIAN"},
                {"user_id": user_map["care_coordinator@wellwatch.ae"].id, "role": "CARE_COORDINATOR"},
            ],
            target_date=now + timedelta(days=180),
            review_schedule="EVERY_6_MONTHS",
        )
        db.add(care_plan)
        db.commit()
        db.refresh(care_plan)

        cp_item = CarePlanItem(
            care_plan_id=care_plan.id,
            recommendation_id=rec1.id,
            status="PLANNED",
            target_date=now + timedelta(days=76),
        )
        db.add(cp_item)
        db.commit()
    else:
        care_plan = existing_care_plan

    # -------------------------------------------------------------------------
    # 10. APPOINTMENT & REFERRAL
    # -------------------------------------------------------------------------
    existing_appt = db.query(Appointment).filter(Appointment.person_id == zayed.id).first()
    if not existing_appt:
        appt = Appointment(
            organization_id=primary_org.id,
            person_id=zayed.id,
            recommendation_id=rec1.id,
            care_plan_id=care_plan.id,
            service_type="PREVENTIVE_LAB_DRAW",
            scheduled_time=now + timedelta(days=14, hours=2),
            location="SEHA Al Bateen Healthcare Center — Phlebotomy Suite B",
            provider_id=user_map["healthcare_provider@wellwatch.ae"].id,
            status=AppointmentStatus.CONFIRMED,
        )
        db.add(appt)

        referral = Referral(
            organization_id=primary_org.id,
            person_id=zayed.id,
            recommendation_id=rec2.id,
            referring_provider_id=user_map["healthcare_provider@wellwatch.ae"].id,
            receiving_org_id=primary_org.id,
            specialty="CLINICAL_NUTRITION",
            clinical_indication="Borderline blood pressure and BMI 27.6. Early preventive dietary coaching.",
            priority="ROUTINE",
            status=ReferralStatus.SUBMITTED,
        )
        db.add(referral)

        recall = RecallTask(
            organization_id=primary_org.id,
            person_id=zayed.id,
            pathway_id="DOH_IFHAS_ADULT_PREVENTION_V1",
            screening_type="IFHAS_ANNUAL_CARDIOVASCULAR",
            due_date=now + timedelta(days=350),
            status=RecallStatus.ACTIVE,
            reminder_count=0,
        )
        db.add(recall)
        db.commit()

    # -------------------------------------------------------------------------
    # 11. NOTIFICATIONS & OUTREACH (Arabic & English)
    # -------------------------------------------------------------------------
    existing_notif = db.query(InAppNotification).filter(
        InAppNotification.recipient_user_id == user_map["individual@wellwatch.ae"].id
    ).first()

    if not existing_notif:
        notif_ar = InAppNotification(
            organization_id=primary_org.id,
            person_id=zayed.id,
            recipient_user_id=user_map["individual@wellwatch.ae"].id,
            title="مرحبًا بك في منصة ويل ووتش للرعاية الوقائية",
            message_body="عزيزي زايد، تم توثيق تقرير الفحص الوقائي الشامل لبرنامج 'افحص' (IFHAS) بنجاح. خطتك الوقائية المخصصة متوفرة الآن في ملفك الشخصي.",
            is_read=False,
        )
        notif_en = InAppNotification(
            organization_id=primary_org.id,
            person_id=zayed.id,
            recipient_user_id=user_map["individual@wellwatch.ae"].id,
            title="Welcome to WELLWATCH PREVENT",
            message_body="Dear Zayed, your comprehensive IFHAS preventive screening assessment is ready. Your personalized care plan has been reviewed and approved by Dr. Mariam Al Mansoori.",
            is_read=False,
        )
        db.add_all([notif_ar, notif_en])
        db.commit()

    # -------------------------------------------------------------------------
    # 12. CHW DEVICE REGISTRATION & OFFLINE ASSIGNMENT
    # -------------------------------------------------------------------------
    chw_user = user_map["community_health_worker@wellwatch.ae"]
    chw_dev = db.query(DeviceRegistration).filter(DeviceRegistration.user_id == chw_user.id).first()
    if not chw_dev:
        chw_dev = DeviceRegistration(
            device_id="CHW-TAB-AD-00892",
            user_id=chw_user.id,
            organization_id=primary_org.id,
            device_model="Samsung Galaxy Tab Active4 Pro (Rugged)",
            os_platform="Android 14 (Knox Secure)",
            app_version="1.0.0",
            status=DeviceStatus.ACTIVE,
            last_sync_at=now,
            registered_at=now - timedelta(days=30),
        )
        db.add(chw_dev)
        db.commit()

        # Offline assignment for community visits
        assignment = OfflineAssignment(
            user_id=chw_user.id,
            patient_id=zayed.id,
            organization_id=primary_org.id,
            assignment_reason="COMMUNITY_WELLNESS_FOLLOWUP",
            assigned_at=now - timedelta(days=2),
            expires_at=now + timedelta(days=5),
        )
        db.add(assignment)
        db.commit()

    return {
        "status": "success",
        "organizations_seeded": len(org_map),
        "users_seeded": len(user_map),
        "patients_seeded": len(person_map),
    }
