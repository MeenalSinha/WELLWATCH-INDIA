from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Gender
from backend.app.models.outreach import (
    CommunicationPreference,
    OutreachTask,
    CommunicationChannel,
    OutreachDeliveryStatus,
    OutreachAuthorizationStatus,
)
from backend.app.services.person_service import PersonService
from backend.app.services.outreach_service import OutreachService


def test_template_rendering_arabic_and_english():
    # Render Arabic
    subj_ar, body_ar, dir_ar = OutreachService.render_template(
        template_id="TMPL-PED-WELL-CHILD-INVITE",
        language="ar",
        variables={"patient_name": "زايد المنصوري"},
    )
    assert dir_ar == "rtl"
    assert "زايد المنصوري" in body_ar
    assert "مركز أبوظبي للصحة العامة" in subj_ar

    # Render English
    subj_en, body_en, dir_en = OutreachService.render_template(
        template_id="TMPL-PED-WELL-CHILD-INVITE",
        language="en",
        variables={"patient_name": "Zayed Al-Mansoori"},
    )
    assert dir_en == "ltr"
    assert "Zayed Al-Mansoori" in body_en
    assert "ADPHC" in subj_en


def test_default_preference_is_arabic_and_in_app(db_session: Session):
    org = Organization(name="SEHA Al Mushrif Clinic", license_number="DOH-HOSP-O01", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Salem",
        last_name="Al Marzouqi",
        date_of_birth=date.today() - timedelta(days=500),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    pref = OutreachService.get_or_create_preference(db_session, child.id)
    assert pref.preferred_language == "ar"
    assert pref.preferred_channel == CommunicationChannel.IN_APP
    assert pref.consent_opt_in is False
    assert pref.authorization_status == OutreachAuthorizationStatus.NOT_PROVIDED
    assert pref.guardian_proxy_required is True  # minor requires guardian


def test_opt_out_consent_suppresses_outreach(db_session: Session):
    org = Organization(name="SEHA Al Zaab Clinic", license_number="DOH-HOSP-O02", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.outreach.opt@seha.ae",
        password_hash="pw",
        full_name="Dr. Buthaina",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Ahmed",
        last_name="Al Suwaidi",
        date_of_birth=date.today() - timedelta(days=35 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Opt out of communication
    OutreachService.update_preference(
        db=db_session,
        person_id=adult.id,
        data={"consent_opt_in": False},
        current_user=provider,
    )

    # Schedule outreach task -> MUST BE SUPPRESSED
    task = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-IFHAS-INVITE",
        current_user=provider,
    )
    assert task.delivery_status == OutreachDeliveryStatus.SUPPRESSED
    assert task.provider_status == "CONSENT_REVOKED"
    assert task.consent_verified is False


def test_pediatric_outreach_targets_guardian_under_wadeemas_law(db_session: Session):
    org = Organization(name="SEHA Pediatric Outreach", license_number="DOH-HOSP-O03", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.ped.outreach@seha.ae",
        password_hash="pw",
        full_name="Dr. Shamma",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    parent = User(
        email="parent.guardian@wellwatch.ae",
        password_hash="pw",
        full_name="Aisha Al Ketbi",
        role=UserRole.PARENT_GUARDIAN,
        is_active=True,
    )
    db_session.add_all([provider, parent])
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Hamdan",
        last_name="Al Ketbi",
        date_of_birth=date.today() - timedelta(days=400),
        gender=Gender.MALE,
        organization_id=org.id,
        guardian_user_id=parent.id,
        relationship_to_guardian="Mother",
    )

    # Guardian provides explicit affirmative consent
    OutreachService.update_preference(
        db=db_session,
        person_id=child.id,
        data={"consent_opt_in": True, "authorization_status": OutreachAuthorizationStatus.AUTHORIZED},
        current_user=provider,
    )

    task = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=child.id,
        template_id="TMPL-PED-WELL-CHILD-INVITE",
        current_user=provider,
    )
    assert task.recipient_role == "GUARDIAN"
    assert task.recipient_id == parent.id


def test_duplicate_outreach_suppressed_within_7_days(db_session: Session):
    org = Organization(name="SEHA Deduplication Center", license_number="DOH-HOSP-O04", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.dedup@seha.ae",
        password_hash="pw",
        full_name="Dr. Saeed",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Rashed",
        last_name="Al Falasi",
        date_of_birth=date.today() - timedelta(days=30 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Explicit affirmative consent granted
    OutreachService.update_preference(
        db=db_session,
        person_id=adult.id,
        data={"consent_opt_in": True, "authorization_status": OutreachAuthorizationStatus.AUTHORIZED},
        current_user=provider,
    )

    # 1st outreach task
    t1 = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-IFHAS-INVITE",
        current_user=provider,
        override_channel=CommunicationChannel.IN_APP,
    )
    assert t1.delivery_status == OutreachDeliveryStatus.DELIVERED

    # 2nd identical outreach task immediately -> MUST BE SUPPRESSED
    t2 = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-IFHAS-INVITE",
        current_user=provider,
        override_channel=CommunicationChannel.IN_APP,
    )
    assert t2.delivery_status == OutreachDeliveryStatus.SUPPRESSED
    assert t2.provider_status == "DUPLICATE_SUPPRESSED"


def test_honest_provider_boundary_reports_not_configured_for_sms(db_session: Session):
    org = Organization(name="SEHA SMS Honest Boundary", license_number="DOH-HOSP-O05", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.sms.boundary@seha.ae",
        password_hash="pw",
        full_name="Dr. Reem",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Nasser",
        last_name="Al Mazrouei",
        date_of_birth=date.today() - timedelta(days=50 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Affirmative consent provided
    OutreachService.update_preference(
        db=db_session,
        person_id=adult.id,
        data={"consent_opt_in": True, "authorization_status": OutreachAuthorizationStatus.AUTHORIZED},
        current_user=provider,
    )

    task = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-BP-RECALL",
        current_user=provider,
        override_channel=CommunicationChannel.SMS,
    )
    assert task.delivery_status == OutreachDeliveryStatus.FAILED
    assert task.provider_status == "NOT_CONFIGURED"
    assert "SMS Gateway API key not configured" in task.failure_reason


def test_outreach_analytics_metrics(db_session: Session):
    org = Organization(name="SEHA Analytics Facility", license_number="DOH-HOSP-O06", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    stats = OutreachService.get_analytics(db=db_session, organization_id=org.id)
    assert "total_generated" in stats
    assert "total_delivered" in stats
    assert "total_suppressed" in stats


def test_in_app_notification_inbox_and_read_lifecycle(db_session: Session):
    org = Organization(name="SEHA Inbox Facility", license_number="DOH-HOSP-O07", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.inbox@seha.ae",
        password_hash="pw",
        full_name="Dr. Laila",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    patient_user = User(
        email="patient.inbox@wellwatch.ae",
        password_hash="pw",
        full_name="Maryam Al Suwaidi",
        role=UserRole.INDIVIDUAL,
        is_active=True,
    )
    db_session.add_all([provider, patient_user])
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Maryam",
        last_name="Al Suwaidi",
        date_of_birth=date.today() - timedelta(days=28 * 365),
        gender=Gender.FEMALE,
        organization_id=org.id,
        user_id=patient_user.id,
    )

    OutreachService.update_preference(
        db=db_session,
        person_id=adult.id,
        data={"consent_opt_in": True, "authorization_status": OutreachAuthorizationStatus.AUTHORIZED},
        current_user=provider,
    )

    task = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-IFHAS-INVITE",
        current_user=provider,
        override_channel=CommunicationChannel.IN_APP,
    )
    assert task.delivery_status == OutreachDeliveryStatus.DELIVERED

    # Patient checks inbox
    notifications = OutreachService.get_inbox(db=db_session, user_id=patient_user.id)
    assert len(notifications) >= 1
    notif = notifications[0]
    assert notif.is_read is False

    # Mark as read
    read_notif = OutreachService.mark_notification_read(db=db_session, notification_id=notif.id, user_id=patient_user.id)
    assert read_notif.is_read is True
