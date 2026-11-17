from typing import List, Optional
from sqlalchemy.orm import Session
from backend.app.models.audit import AuditOutcome
from backend.app.models.screening import ScreeningEncounter
from backend.app.models.user import User
from backend.app.schemas.screening import ScreeningCreate
from backend.app.services.audit_service import AuditService


class ScreeningService:
    @staticmethod
    def create_screening_encounter(
        db: Session,
        provider: User,
        data: ScreeningCreate,
        correlation_id: str
    ) -> ScreeningEncounter:
        # Calculate BMI if height and weight are provided
        calculated_bmi = None
        if data.height_cm and data.weight_kg and data.height_cm > 0:
            height_m = data.height_cm / 100.0
            calculated_bmi = round(data.weight_kg / (height_m * height_m), 1)

        encounter = ScreeningEncounter(
            patient_id=data.patient_id,
            provider_id=provider.id,
            facility_id=data.facility_id,
            encounter_type=data.encounter_type.value,
            systolic_bp=data.systolic_bp,
            diastolic_bp=data.diastolic_bp,
            pulse_rate=data.pulse_rate,
            fasting_glucose=data.fasting_glucose,
            height_cm=data.height_cm,
            weight_kg=data.weight_kg,
            bmi=calculated_bmi,
            epistemic_status=data.epistemic_status.value,
            notes=data.notes
        )
        db.add(encounter)
        db.commit()
        db.refresh(encounter)

        AuditService.log_event(
            db=db,
            action="SCREENING_ENCOUNTER_CREATED",
            resource_type="ENCOUNTER",
            resource_id=encounter.id,
            actor_user_id=provider.id,
            organization_id=data.facility_id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            details={"type": data.encounter_type.value, "epistemic_status": data.epistemic_status.value}
        )
        return encounter

    @staticmethod
    def get_scoped_screenings(
        db: Session,
        user: User,
        facility_id: Optional[str] = None
    ) -> List[ScreeningEncounter]:
        query = db.query(ScreeningEncounter)
        if user.role in ["COMMUNITY_HEALTH_WORKER", "HEALTHCARE_PROVIDER", "CARE_COORDINATOR"]:
            if user.organization_id:
                query = query.filter(ScreeningEncounter.facility_id == user.organization_id)
        elif facility_id:
            query = query.filter(ScreeningEncounter.facility_id == facility_id)
            
        return query.order_by(ScreeningEncounter.created_at.desc()).limit(100).all()
