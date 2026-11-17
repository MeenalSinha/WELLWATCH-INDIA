import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.person import Person
from backend.app.models.user import User, UserRole
from backend.app.models.assessment import PreventiveAssessment, ClinicalRiskTier
from backend.app.models.screening import EpistemicStatus
from backend.app.models.genomics import GenomicFinding, VariantClassification
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationReview,
    PreventiveCarePlan,
    CarePlanItem,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)
from backend.app.models.audit import AuditOutcome
from backend.app.services.audit_service import AuditService
from backend.app.services.pathway_engine import PathwayEngine
from backend.app.services.clinical_validation_service import ClinicalValidationService

logger = logging.getLogger(__name__)

RULES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
    "recommendation_rules",
)


def _get_role_str(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


class RecommendationService:
    _rules_cache: List[Dict[str, Any]] = []

    @classmethod
    def load_rules(cls) -> List[Dict[str, Any]]:
        if cls._rules_cache:
            return cls._rules_cache

        all_rules = []
        if os.path.exists(RULES_DIR):
            for fname in os.listdir(RULES_DIR):
                if fname.endswith(".json"):
                    fpath = os.path.join(RULES_DIR, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            rules = data.get("rules", [])
                            all_rules.extend(rules)
                    except Exception as e:
                        logger.error(f"Error loading recommendation rules from {fpath}: {e}")
        cls._rules_cache = all_rules
        return cls._rules_cache

    @classmethod
    def reload_rules(cls):
        cls._rules_cache = []
        return cls.load_rules()

    @staticmethod
    def generate_recommendations_for_patient(
        db: Session,
        person_id: str,
        current_user: User,
    ) -> List[PreventiveRecommendation]:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        role_str = _get_role_str(current_user)
        # Multi-tenant isolation check
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization patient access denied")

        # Fetch latest assessment
        latest_assessment = (
            db.query(PreventiveAssessment)
            .filter(PreventiveAssessment.patient_id == person_id)
            .order_by(PreventiveAssessment.created_at.desc())
            .first()
        )

        # Fetch approved genomic findings
        genomic_findings = (
            db.query(GenomicFinding)
            .filter(GenomicFinding.patient_id == person_id)
            .all()
        )

        rules = RecommendationService.load_rules()
        created_recommendations = []
        now = datetime.now(timezone.utc)

        # Identify applicable care gaps & epistemic uncertainty
        care_gaps_identified = []
        epistemic_status = "CONFIRMED"
        assessment_id = None
        evidence_dict = {}

        if latest_assessment:
            assessment_id = latest_assessment.id
            evidence_dict["assessment_id"] = latest_assessment.id
            evidence_dict["clinical_risk_tier"] = latest_assessment.clinical_risk_tier.value if hasattr(latest_assessment.clinical_risk_tier, "value") else str(latest_assessment.clinical_risk_tier)
            
            try:
                factors = json.loads(latest_assessment.contributing_factors) if isinstance(latest_assessment.contributing_factors, str) else (latest_assessment.contributing_factors or [])
            except Exception:
                factors = []
            evidence_dict["risk_indicators"] = factors
            
            # Extract care gaps from assessment
            for ind in factors:
                if isinstance(ind, dict):
                    ind_name = ind.get("indicator", "") or ind.get("factor", "")
                    if "BLOOD_PRESSURE" in ind_name or "HYPERTENSION" in ind_name:
                        care_gaps_identified.append("GAP-BP-OVERDUE")
                    elif "GLUCOSE" in ind_name or "DIABETES" in ind_name:
                        care_gaps_identified.append("GAP-DM-GLUCOSE-OVERDUE")

        # Evaluate pediatric vs adult pathways
        if person.is_pediatric:
            care_gaps_identified.append("GAP-WELL-CHILD-DUE")
        else:
            care_gaps_identified.append("GAP-IFHAS-COMPREHENSIVE")

        # Check genomic findings
        for gf in genomic_findings:
            if gf.variant_classification in [VariantClassification.PATHOGENIC, VariantClassification.LIKELY_PATHOGENIC]:
                care_gaps_identified.append("GAP-GENOMIC-COUNSELING-DUE")
                if "LDLR" in (gf.gene_symbol or ""):
                    care_gaps_identified.append("GAP-LIPID-SPECIALIST-EVALUATION")

        # Query existing active recommendations for this patient to prevent duplicate active recs
        existing_recs = (
            db.query(PreventiveRecommendation)
            .filter(
                PreventiveRecommendation.person_id == person_id,
                PreventiveRecommendation.lifecycle_status.notin_([
                    RecommendationLifecycleStatus.CANCELLED,
                    RecommendationLifecycleStatus.SUPERSEDED,
                    RecommendationLifecycleStatus.COMPLETED,
                ])
            )
            .all()
        )
        existing_rule_ids = {r.rule_id for r in existing_recs}

        for rule in rules:
            rule_id = rule["rule_id"]
            if rule_id in existing_rule_ids:
                continue

            trigger = rule.get("trigger_condition", {})
            req_care_gap = trigger.get("care_gap_code")
            req_epistemic = trigger.get("epistemic_status")

            matched = False
            if req_care_gap and req_care_gap in care_gaps_identified:
                matched = True
            elif req_epistemic and req_epistemic == epistemic_status:
                matched = True

            if not matched:
                continue

            # Check population eligibility
            app_pop = rule.get("applicable_population", "all")
            if app_pop == "pediatric" and not person.is_pediatric:
                continue
            if app_pop == "adult" and person.is_pediatric:
                continue

            # Calculate priority and target window
            window_days = rule.get("recommended_window_days", 30)
            target_date = now + timedelta(days=window_days)
            priority_str = rule.get("default_priority", "ROUTINE")
            priority = RecommendationPriority[priority_str]

            rec = PreventiveRecommendation(
                organization_id=person.primary_organization_id or current_user.organization_id,
                person_id=person.id,
                assessment_id=assessment_id,
                pathway_id=rule.get("pathway_id", "general"),
                care_gap_id=req_care_gap,
                category=RecommendationCategory[rule["category"]],
                priority=priority,
                lifecycle_status=RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW if rule.get("requires_human_review", True) else RecommendationLifecycleStatus.APPROVED,
                recommended_action=rule["recommended_action"],
                clinical_rationale=rule["clinical_rationale"],
                supporting_evidence={
                    "evidence_dict": evidence_dict,
                    "matched_care_gap": req_care_gap,
                    "epistemic_status": epistemic_status,
                    "patient_age": person.age_years,
                },
                source_guideline=rule["source_guideline"],
                uncertainty_status=epistemic_status,
                rule_id=rule_id,
                rule_version=rule.get("rule_version", "1.0.0"),
                target_date=target_date,
                expiration_date=now + timedelta(days=window_days * 3),
                requires_human_review=rule.get("requires_human_review", True),
            )
            db.add(rec)
            created_recommendations.append(rec)

        db.commit()

        for r in created_recommendations:
            db.refresh(r)
            AuditService.log(
                db=db,
                user_id=current_user.id,
                action="RECOMMENDATION_GENERATED",
                resource_type="PreventiveRecommendation",
                resource_id=r.id,
                organization_id=r.organization_id,
                outcome=AuditOutcome.SUCCESS,
                details={"rule_id": r.rule_id, "category": r.category.value, "priority": r.priority.value},
            )

        return created_recommendations

    @staticmethod
    def get_recommendations(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[RecommendationLifecycleStatus] = None,
        priority: Optional[RecommendationPriority] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[PreventiveRecommendation]]:
        query = db.query(PreventiveRecommendation)
        if person_id:
            query = query.filter(PreventiveRecommendation.person_id == person_id)
        if organization_id:
            query = query.filter(PreventiveRecommendation.organization_id == organization_id)
        if status:
            query = query.filter(PreventiveRecommendation.lifecycle_status == status)
        if priority:
            query = query.filter(PreventiveRecommendation.priority == priority)

        total = query.count()
        items = query.order_by(PreventiveRecommendation.created_at.desc()).offset(offset).limit(limit).all()
        return total, items

    @staticmethod
    def get_recommendation_by_id(
        db: Session,
        recommendation_id: str,
        current_user: User,
    ) -> PreventiveRecommendation:
        rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
        if not rec:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recommendation not found")
        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and rec.organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization recommendation access denied")
        return rec

    @staticmethod
    def review_recommendation(
        db: Session,
        recommendation_id: str,
        reviewer_user: User,
        review_decision: str,
        rationale: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> PreventiveRecommendation:
        role_str = _get_role_str(reviewer_user)
        # Clinical persona authorization check
        if role_str != UserRole.HEALTHCARE_PROVIDER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' is not authorized for clinical recommendation review. Only licensed healthcare providers may review.",
            )

        rec = RecommendationService.get_recommendation_by_id(db, recommendation_id, reviewer_user)

        valid_decisions = ["APPROVED", "MODIFIED", "REJECTED", "INFO_REQUESTED"]
        decision_upper = review_decision.upper()
        if decision_upper not in valid_decisions:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid review decision: {review_decision}")

        review = RecommendationReview(
            recommendation_id=rec.id,
            reviewer_id=reviewer_user.id,
            review_decision=decision_upper,
            rationale=rationale,
            modifications=modifications,
        )
        db.add(review)

        if decision_upper == "APPROVED":
            rec.lifecycle_status = RecommendationLifecycleStatus.APPROVED
        elif decision_upper == "MODIFIED":
            rec.lifecycle_status = RecommendationLifecycleStatus.MODIFIED
            if modifications and "recommended_action" in modifications:
                rec.recommended_action = modifications["recommended_action"]
        elif decision_upper == "REJECTED":
            rec.lifecycle_status = RecommendationLifecycleStatus.CANCELLED
        elif decision_upper == "INFO_REQUESTED":
            rec.lifecycle_status = RecommendationLifecycleStatus.DRAFT

        rec.assigned_provider_id = reviewer_user.id
        rec.reviewer_notes = rationale
        db.commit()
        db.refresh(rec)

        AuditService.log(
            db=db,
            user_id=reviewer_user.id,
            action="RECOMMENDATION_REVIEWED",
            resource_type="PreventiveRecommendation",
            resource_id=rec.id,
            organization_id=rec.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"decision": decision_upper, "rationale": rationale},
        )
        return rec

    @staticmethod
    def update_status(
        db: Session,
        recommendation_id: str,
        current_user: User,
        new_status: RecommendationLifecycleStatus,
        declined_reason: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
    ) -> PreventiveRecommendation:
        rec = RecommendationService.get_recommendation_by_id(db, recommendation_id, current_user)

        # Validate decline reason
        if new_status == RecommendationLifecycleStatus.DECLINED and not declined_reason:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Declined recommendations require a documented reason.")

        rec.lifecycle_status = new_status
        if declined_reason:
            rec.declined_reason = declined_reason
        if reviewer_notes:
            rec.reviewer_notes = reviewer_notes

        db.commit()
        db.refresh(rec)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="RECOMMENDATION_STATUS_UPDATED",
            resource_type="PreventiveRecommendation",
            resource_id=rec.id,
            organization_id=rec.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"new_status": new_status.value, "declined_reason": declined_reason},
        )
        return rec

    @staticmethod
    def create_care_plan(
        db: Session,
        person_id: str,
        plan_name: str,
        pathway_id: str,
        goals: List[Dict[str, Any]],
        assigned_care_team: List[Dict[str, Any]],
        recommendation_ids: List[str],
        current_user: User,
        target_date: Optional[datetime] = None,
        review_schedule: Optional[str] = "EVERY_6_MONTHS",
    ) -> PreventiveCarePlan:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        care_plan = PreventiveCarePlan(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person.id,
            plan_name=plan_name,
            pathway_id=pathway_id,
            status="ACTIVE",
            goals=goals,
            responsible_org_id=person.primary_organization_id or current_user.organization_id,
            assigned_care_team=assigned_care_team,
            target_date=target_date,
            review_schedule=review_schedule,
        )
        db.add(care_plan)
        db.flush()

        for rec_id in recommendation_ids:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == rec_id).first()
            if rec:
                item = CarePlanItem(
                    care_plan_id=care_plan.id,
                    recommendation_id=rec.id,
                    status="PLANNED",
                    target_date=rec.target_date,
                )
                db.add(item)

        db.commit()
        db.refresh(care_plan)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="CARE_PLAN_CREATED",
            resource_type="PreventiveCarePlan",
            resource_id=care_plan.id,
            organization_id=care_plan.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"plan_name": plan_name, "pathway_id": pathway_id, "item_count": len(recommendation_ids)},
        )
        return care_plan

    @staticmethod
    def get_care_plans_for_patient(
        db: Session,
        person_id: str,
        current_user: User,
    ) -> List[PreventiveCarePlan]:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        plans = (
            db.query(PreventiveCarePlan)
            .filter(PreventiveCarePlan.person_id == person_id)
            .order_by(PreventiveCarePlan.created_at.desc())
            .all()
        )
        return plans
