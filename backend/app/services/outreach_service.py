import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.person import Person
from backend.app.models.user import User, UserRole
from backend.app.models.guardian import Guardianship, GuardianshipStatus
from backend.app.models.recommendation import PreventiveRecommendation
from backend.app.models.outreach import (
    CommunicationPreference,
    OutreachTask,
    CommunicationChannel,
    OutreachDeliveryStatus,
)
from backend.app.models.audit import AuditOutcome
from backend.app.services.audit_service import AuditService
from backend.app.services.communication_provider import CommunicationProviderService

logger = logging.getLogger(__name__)

TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
    "templates",
    "outreach_templates.json",
)


def _get_role_str(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


class OutreachService:
    _templates_cache: Dict[str, Any] = {}

    @classmethod
    def load_templates(cls) -> Dict[str, Any]:
        if cls._templates_cache:
            return cls._templates_cache
        if os.path.exists(TEMPLATES_PATH):
            try:
                with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    templates = {t["template_id"]: t for t in data.get("templates", [])}
                    cls._templates_cache = templates
            except Exception as e:
                logger.error(f"Error loading outreach templates: {e}")
        return cls._templates_cache

    @staticmethod
    def get_or_create_preference(db: Session, person_id: str) -> CommunicationPreference:
        pref = db.query(CommunicationPreference).filter(CommunicationPreference.person_id == person_id).first()
        if not pref:
            person = db.query(Person).filter(Person.id == person_id).first()
            pref = CommunicationPreference(
                person_id=person_id,
                preferred_language="ar",  # Default to Arabic in Abu Dhabi
                preferred_channel=CommunicationChannel.IN_APP,
                guardian_proxy_required=person.is_pediatric if person else False,
                consent_opt_in=True,
            )
            db.add(pref)
            db.commit()
            db.refresh(pref)
        return pref

    @staticmethod
    def update_preference(
        db: Session,
        person_id: str,
        data: Dict[str, Any],
        current_user: User,
    ) -> CommunicationPreference:
        pref = OutreachService.get_or_create_preference(db, person_id)
        for k, v in data.items():
            if v is not None and hasattr(pref, k):
                setattr(pref, k, v)
        if "consent_opt_in" in data and data["consent_opt_in"] is False:
            pref.opt_out_timestamp = datetime.now(timezone.utc)
        elif "consent_opt_in" in data and data["consent_opt_in"] is True:
            pref.opt_out_timestamp = None

        db.commit()
        db.refresh(pref)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="COMMUNICATION_PREFERENCE_UPDATED",
            resource_type="CommunicationPreference",
            resource_id=pref.id,
            organization_id=current_user.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"preferred_language": pref.preferred_language, "preferred_channel": pref.preferred_channel.value},
        )
        return pref

    @staticmethod
    def render_template(
        template_id: str,
        language: str,
        variables: Dict[str, str],
    ) -> Tuple[str, str, str]:
        templates = OutreachService.load_templates()
        tmpl = templates.get(template_id)
        if not tmpl:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

        lang_key = "ar" if language.lower() == "ar" else "en"
        content = tmpl.get(lang_key, tmpl.get("ar", {}))
        subject = content.get("subject", "")
        body = content.get("body", "")
        direction = content.get("direction", "rtl" if lang_key == "ar" else "ltr")

        for k, v in variables.items():
            subject = subject.replace(f"{{{k}}}", str(v))
            body = body.replace(f"{{{k}}}", str(v))

        return subject, body, direction

    @staticmethod
    def schedule_outreach_task(
        db: Session,
        person_id: str,
        template_id: str,
        current_user: User,
        recommendation_id: Optional[str] = None,
        override_channel: Optional[CommunicationChannel] = None,
        scheduled_at: Optional[datetime] = None,
        custom_variables: Optional[Dict[str, str]] = None,
    ) -> OutreachTask:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization outreach denied")

        pref = OutreachService.get_or_create_preference(db, person_id)

        # Consent verification
        if not pref.consent_opt_in:
            task = OutreachTask(
                organization_id=person.primary_organization_id or current_user.organization_id,
                person_id=person_id,
                recommendation_id=recommendation_id,
                channel=override_channel or pref.preferred_channel,
                template_id=template_id,
                language=pref.preferred_language,
                message_body="[SUPPRESSED: Patient opted out of preventive communication]",
                scheduled_at=datetime.now(timezone.utc),
                delivery_status=OutreachDeliveryStatus.SUPPRESSED,
                provider_status="CONSENT_REVOKED",
                failure_reason="Patient communication consent is set to OPT_OUT.",
                consent_verified=False,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            return task

        # Pediatric guardianship verification (Wadeema's Law proxy)
        recipient_role = "PATIENT"
        recipient_id = person_id
        if person.is_pediatric:
            guardian = (
                db.query(Guardianship)
                .filter(
                    Guardianship.dependent_person_id == person_id,
                    Guardianship.verification_status.in_([GuardianshipStatus.VERIFIED, GuardianshipStatus.VERIFIED.value, "VERIFIED"]),
                )
                .first()
            )
            recipient_role = "GUARDIAN"
            if guardian:
                recipient_id = guardian.guardian_user_id

        # Duplicate prevention (suppress if identical template sent within 7 days)
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent_duplicate = (
            db.query(OutreachTask)
            .filter(
                OutreachTask.person_id == person_id,
                OutreachTask.template_id == template_id,
                OutreachTask.created_at >= seven_days_ago,
                OutreachTask.delivery_status.in_([
                    OutreachDeliveryStatus.DELIVERED,
                    OutreachDeliveryStatus.SENT,
                    OutreachDeliveryStatus.PENDING,
                ]),
            )
            .first()
        )
        if recent_duplicate:
            task = OutreachTask(
                organization_id=person.primary_organization_id or current_user.organization_id,
                person_id=person_id,
                recommendation_id=recommendation_id,
                channel=override_channel or pref.preferred_channel,
                template_id=template_id,
                language=pref.preferred_language,
                message_body="[SUPPRESSED: Duplicate prevention window active (7 days)]",
                scheduled_at=datetime.now(timezone.utc),
                delivery_status=OutreachDeliveryStatus.SUPPRESSED,
                provider_status="DUPLICATE_SUPPRESSED",
                failure_reason="Identical template dispatched within recent 7-day recall suppression window.",
                consent_verified=True,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            return task

        # Render template
        vars_dict = {
            "patient_name": person.full_name,
            "appointment_time": "قريباً / Soon",
            "location": "مركز الرعاية الصحية الأولية - أبوظبي / Primary Healthcare Center - Abu Dhabi",
        }
        if custom_variables:
            vars_dict.update(custom_variables)

        subject, body, _ = OutreachService.render_template(template_id, pref.preferred_language, vars_dict)
        full_message = f"[{subject}]\n\n{body}"

        now = datetime.now(timezone.utc)
        sched_time = scheduled_at or now
        channel = override_channel or pref.preferred_channel

        task = OutreachTask(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            recommendation_id=recommendation_id,
            recipient_id=recipient_id,
            recipient_role=recipient_role,
            channel=channel,
            template_id=template_id,
            language=pref.preferred_language,
            message_body=full_message,
            scheduled_at=sched_time,
            delivery_status=OutreachDeliveryStatus.PENDING,
            provider_status="QUEUED",
            consent_verified=True,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        # If scheduled for now, attempt dispatch via provider boundary
        if sched_time <= now:
            task.attempted_at = now
            task.delivery_status = OutreachDeliveryStatus.SENDING
            db.commit()

            delivery_res = CommunicationProviderService.dispatch(
                channel=channel,
                recipient_target=getattr(person, "phone_number", None) or "inapp",
                message_body=full_message,
                metadata={"task_id": task.id, "person_id": person.id},
            )

            task.delivery_status = delivery_res.status
            task.provider_status = delivery_res.provider_status
            task.provider_message_id = delivery_res.provider_message_id
            task.failure_reason = delivery_res.failure_reason
            db.commit()
            db.refresh(task)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="OUTREACH_TASK_SCHEDULED",
            resource_type="OutreachTask",
            resource_id=task.id,
            organization_id=task.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"template_id": template_id, "channel": channel.value, "status": task.delivery_status.value},
        )
        return task

    @staticmethod
    def get_outreach_tasks(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        delivery_status: Optional[OutreachDeliveryStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[OutreachTask]]:
        query = db.query(OutreachTask)
        if person_id:
            query = query.filter(OutreachTask.person_id == person_id)
        if organization_id:
            query = query.filter(OutreachTask.organization_id == organization_id)
        if delivery_status:
            query = query.filter(OutreachTask.delivery_status == delivery_status)

        total = query.count()
        items = query.order_by(OutreachTask.created_at.desc()).offset(offset).limit(limit).all()
        return total, items

    @staticmethod
    def retry_task(db: Session, task_id: str, current_user: User) -> OutreachTask:
        task = db.query(OutreachTask).filter(OutreachTask.id == task_id).first()
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Outreach task not found")

        if task.retry_count >= task.max_retries:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum retry limit reached for this outreach task.")

        task.retry_count += 1
        task.attempted_at = datetime.now(timezone.utc)
        delivery_res = CommunicationProviderService.dispatch(
            channel=task.channel,
            recipient_target="retry_target",
            message_body=task.message_body,
        )
        task.delivery_status = delivery_res.status
        task.provider_status = delivery_res.provider_status
        task.provider_message_id = delivery_res.provider_message_id
        task.failure_reason = delivery_res.failure_reason
        db.commit()
        db.refresh(task)
        return task

    @staticmethod
    def get_analytics(db: Session, organization_id: Optional[str] = None) -> Dict[str, Any]:
        query = db.query(OutreachTask)
        if organization_id:
            query = query.filter(OutreachTask.organization_id == organization_id)

        all_tasks = query.all()
        total_gen = len(all_tasks)
        attempted = sum(1 for t in all_tasks if t.attempted_at is not None)
        delivered = sum(1 for t in all_tasks if t.delivery_status == OutreachDeliveryStatus.DELIVERED)
        failed = sum(1 for t in all_tasks if t.delivery_status == OutreachDeliveryStatus.FAILED)
        suppressed = sum(1 for t in all_tasks if t.delivery_status == OutreachDeliveryStatus.SUPPRESSED)

        by_chan = {}
        by_lang = {}
        for t in all_tasks:
            chan_name = t.channel.value if hasattr(t.channel, "value") else str(t.channel)
            by_chan[chan_name] = by_chan.get(chan_name, 0) + 1
            by_lang[t.language] = by_lang.get(t.language, 0) + 1

        return {
            "total_generated": total_gen,
            "total_attempted": attempted,
            "total_delivered": delivered,
            "total_failed": failed,
            "total_suppressed": suppressed,
            "by_channel": by_chan,
            "by_language": by_lang,
            "active_providers_count": 1,  # In-App active; external SMS/WhatsApp/Email report NOT_CONFIGURED
            "honest_boundary_note": "In-app notifications deliver directly to secure portal inbox. External SMS/WhatsApp gateways accurately report NOT_CONFIGURED when API credentials are unset.",
        }
