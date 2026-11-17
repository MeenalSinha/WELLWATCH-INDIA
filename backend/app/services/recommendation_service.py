import os
import json
import logging
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.person import Person
from backend.app.models.user import User, UserRole
from backend.app.models.assessment import PreventiveAssessment, ClinicalRiskTier, AssessmentStatus
from backend.app.models.clinical import ScreeningRecord, ClinicalObservation, ScreeningStatus
from backend.app.models.screening import EpistemicStatus
from backend.app.models.genomics import GenomicFinding, VariantClassification, GenomicReviewStatus
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
from backend.app.services.pathway_engine import PathwayEngine, CareGapStatus

logger = logging.getLogger(__name__)

RULES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
    "recommendation_rules",
)

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
    "schemas",
    "recommendation_rules.schema.json",
)


def _get_role_str(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


class RecommendationService:
    _rules_cache: List[Dict[str, Any]] = []

    @classmethod
    def load_rules(cls) -> List[Dict[str, Any]]:
        """
        Loads, validates, and caches governed recommendation rules.
        Enforces schema validation, approval status, and duplicate rule_id detection.
        """
        if cls._rules_cache:
            return cls._rules_cache

        all_rules: List[Dict[str, Any]] = []
        seen_rule_ids: Set[str] = set()

        if not os.path.exists(RULES_DIR):
            logger.warning(f"Recommendation rules directory {RULES_DIR} does not exist.")
            cls._rules_cache = []
            return cls._rules_cache

        rule_files = sorted([f for f in os.listdir(RULES_DIR) if f.endswith(".json")])
        for fname in rule_files:
            fpath = os.path.join(RULES_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                rules = data.get("rules", [])
                for rule in rules:
                    # Validate required schema fields
                    rule_id = rule.get("rule_id")
                    if not rule_id:
                        raise ValueError(f"Rule in {fname} missing required 'rule_id'")

                    # Duplicate detection across files
                    if rule_id in seen_rule_ids:
                        raise ValueError(f"Duplicate recommendation rule_id '{rule_id}' detected across rule files.")
                    seen_rule_ids.add(rule_id)

                    # Required schema attributes
                    required_fields = [
                        "rule_version",
                        "category",
                        "pathway_id",
                        "trigger_condition",
                        "default_priority",
                        "recommended_action",
                        "clinical_rationale",
                        "source_guideline",
                        "requires_human_review",
                    ]
                    for rf in required_fields:
                        if rf not in rule:
                            raise ValueError(f"Rule '{rule_id}' in {fname} missing required field '{rf}'")

                    # Approval governance check
                    approval_status = rule.get("approval_status", "APPROVED")
                    if approval_status != "APPROVED":
                        logger.info(f"Skipping unapproved rule '{rule_id}' with status '{approval_status}'")
                        continue

                    all_rules.append(rule)
            except Exception as e:
                logger.error(f"Error loading and validating recommendation rules from {fpath}: {e}")
                raise e

        cls._rules_cache = all_rules
        return cls._rules_cache

    @classmethod
    def reload_rules(cls) -> List[Dict[str, Any]]:
        cls._rules_cache = []
        return cls.load_rules()

    @staticmethod
    def generate_recommendations_for_patient(
        db: Session,
        person_id: str,
        current_user: User,
    ) -> List[PreventiveRecommendation]:
        """
        Generates evidence-backed preventive recommendations for a patient.
        Strictly decouples eligibility from confirmed care gaps.
        Consumes Phase 6 pathway evaluations and Phase 7 assessments.
        Propagates epistemic uncertainty rather than defaulting to confirmed.
        """
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

        # Fetch reviewed genomic findings
        genomic_findings = (
            db.query(GenomicFinding)
            .filter(
                GenomicFinding.patient_id == person_id,
                GenomicFinding.review_status == GenomicReviewStatus.GENETIC_COUNSELOR_VERIFIED,
                GenomicFinding.variant_classification.in_([
                    VariantClassification.PATHOGENIC,
                    VariantClassification.LIKELY_PATHOGENIC,
                ])
            )
            .all()
        )

        rules = RecommendationService.load_rules()
        created_recommendations = []
        now = datetime.now(timezone.utc)

        care_gaps_identified: List[str] = []
        epistemic_status = "ROBUST"
        assessment_id = None
        evidence_dict: Dict[str, Any] = {}

        # 1. Evaluate longitudinal clinical pathway through PathwayEngine
        pathway_key = "pediatric_well_child" if person.is_pediatric else "adphc_ifhas"
        pathway_def = PathwayEngine.load_pathway(pathway_key)
        
        pathway_care_gaps = []
        if pathway_def:
            pathway_care_gaps = PathwayEngine.evaluate_care_gaps(db=db, pathway=pathway_def, person=person)

        # Check for confirmed DUE/OVERDUE milestones vs NEEDS_DATA
        for gap in pathway_care_gaps:
            gap_status = gap.get("status")
            m_id = gap.get("milestone_id", "")
            
            if gap_status in [CareGapStatus.DUE.value, CareGapStatus.OVERDUE.value]:
                if person.is_pediatric:
                    care_gaps_identified.append("GAP-WELL-CHILD-DUE")
                    components = gap.get("components", [])
                    if any("VACCIN" in str(c).upper() for c in components):
                        care_gaps_identified.append("GAP-VACCINATION-DUE")
                else:
                    if "IFHAS" in m_id:
                        care_gaps_identified.append("GAP-IFHAS-COMPREHENSIVE")
                    if "CARDIOVASCULAR" in m_id or "CARDIOMETABOLIC" in m_id:
                        care_gaps_identified.append("GAP-BP-OVERDUE")
                    if "DIABETES" in m_id:
                        care_gaps_identified.append("GAP-DM-GLUCOSE-OVERDUE")
                    if "COLORECTAL" in m_id:
                        care_gaps_identified.append("GAP-COLORECTAL-FIT-DUE")
                    if "BREAST" in m_id and getattr(person.gender, "value", str(person.gender)) == "FEMALE":
                        care_gaps_identified.append("GAP-BREAST-MAMMO-DUE")
            elif gap_status in [CareGapStatus.NEEDS_DATA.value, CareGapStatus.POTENTIALLY_OVERDUE.value]:
                epistemic_status = "NEEDS_DATA"

        # 2. Consume Phase 7 Assessment if available
        if latest_assessment:
            assessment_id = latest_assessment.id
            evidence_dict["assessment_id"] = latest_assessment.id
            evidence_dict["clinical_risk_tier"] = (
                latest_assessment.clinical_risk_tier.value
                if hasattr(latest_assessment.clinical_risk_tier, "value")
                else str(latest_assessment.clinical_risk_tier)
            )

            # Epistemic check on assessment
            if latest_assessment.clinical_risk_tier == ClinicalRiskTier.INSUFFICIENT_DATA or latest_assessment.assessment_status == AssessmentStatus.NEEDS_DATA:
                epistemic_status = "NEEDS_DATA"
                evidence_dict["epistemic_abstention"] = True

            # Parse contributing factors for clinical risk indicators
            try:
                factors = (
                    json.loads(latest_assessment.contributing_factors)
                    if isinstance(latest_assessment.contributing_factors, str)
                    else (latest_assessment.contributing_factors or [])
                )
            except Exception:
                factors = []
            evidence_dict["risk_indicators"] = factors

            for ind in factors:
                if isinstance(ind, dict):
                    ind_name = ind.get("indicator", "") or ind.get("factor", "")
                    if "BLOOD_PRESSURE" in ind_name or "HYPERTENSION" in ind_name:
                        care_gaps_identified.append("GAP-BP-OVERDUE")
                    elif "GLUCOSE" in ind_name or "DIABETES" in ind_name:
                        care_gaps_identified.append("GAP-DM-GLUCOSE-OVERDUE")
        else:
            # Check if any observations exist
            obs_count = db.query(ClinicalObservation).filter(ClinicalObservation.patient_id == person_id).count()
            if obs_count == 0:
                epistemic_status = "NEEDS_DATA"
                evidence_dict["baseline_data_missing"] = True

        # 3. Consume Verified Actionable Genomic Findings
        for gf in genomic_findings:
            care_gaps_identified.append("GAP-GENOMIC-COUNSELING-DUE")
            # Map specific gene actions through governed rules matching
            for r in rules:
                rule_trigger = r.get("trigger_condition", {})
                req_gene = rule_trigger.get("gene_symbol")
                if req_gene and req_gene.upper() == (gf.gene_symbol or "").upper():
                    gap_code = rule_trigger.get("care_gap_code")
                    if gap_code:
                        care_gaps_identified.append(gap_code)

        evidence_dict["care_gaps_identified"] = list(set(care_gaps_identified))
        evidence_dict["epistemic_status"] = epistemic_status

        # Query existing active recommendations for this patient to prevent duplicate active recs
        active_statuses = [
            RecommendationLifecycleStatus.DRAFT,
            RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW,
            RecommendationLifecycleStatus.APPROVED,
            RecommendationLifecycleStatus.MODIFIED,
            RecommendationLifecycleStatus.OFFERED,
            RecommendationLifecycleStatus.ACCEPTED,
            RecommendationLifecycleStatus.SCHEDULED,
            RecommendationLifecycleStatus.IN_PROGRESS,
            RecommendationLifecycleStatus.DECLINED,
            RecommendationLifecycleStatus.COMPLETED,
        ]
        existing_recs = (
            db.query(PreventiveRecommendation)
            .filter(
                PreventiveRecommendation.person_id == person_id,
                PreventiveRecommendation.lifecycle_status.in_(active_statuses),
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

            uncertainty_val = "INSUFFICIENT_DATA_ABSTAINED" if epistemic_status == "NEEDS_DATA" else "ROBUST"

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
                uncertainty_status=uncertainty_val,
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
        # Strict Clinical Persona Authorization: System Admin CANNOT perform clinical review
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

        # State Machine Integrity
        if rec.lifecycle_status in [RecommendationLifecycleStatus.CANCELLED, RecommendationLifecycleStatus.COMPLETED]:
            if new_status not in [RecommendationLifecycleStatus.CANCELLED, RecommendationLifecycleStatus.COMPLETED]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot transition finalized recommendation from '{rec.lifecycle_status.value}' to '{new_status.value}'.",
                )

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

        role_str = _get_role_str(current_user)
        # Verify cross-organization access
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization care plan creation denied.")

        # Strict validation of every recommendation ID
        verified_recs: List[PreventiveRecommendation] = []
        for rec_id in recommendation_ids:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == rec_id).first()
            if not rec:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recommendation '{rec_id}' not found.")
            if rec.person_id != person_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recommendation '{rec_id}' does not belong to patient '{person_id}'.")
            if role_str != UserRole.SYSTEM_ADMIN.value and rec.organization_id != current_user.organization_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Recommendation '{rec_id}' belongs to another healthcare organization.")
            verified_recs.append(rec)

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

        seen_rec_ids: Set[str] = set()
        for rec in verified_recs:
            if rec.id in seen_rec_ids:
                continue
            seen_rec_ids.add(rec.id)
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
            details={"plan_name": plan_name, "pathway_id": pathway_id, "item_count": len(seen_rec_ids)},
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

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization care plan access denied.")

        plans = (
            db.query(PreventiveCarePlan)
            .filter(PreventiveCarePlan.person_id == person_id)
            .order_by(PreventiveCarePlan.created_at.desc())
            .all()
        )
        return plans
