import enum
import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.models.clinical import ScreeningRecord, ScreeningStatus
from backend.app.models.person import Gender, Person, PopulationClassification
from backend.app.models.screening import EpistemicStatus


class PathwayStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class EligibilityStatus(str, enum.Enum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    NEEDS_DATA = "NEEDS_DATA"


class CareGapStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    DUE = "DUE"
    OVERDUE = "OVERDUE"
    POTENTIALLY_OVERDUE = "POTENTIALLY_OVERDUE"
    NEEDS_DATA = "NEEDS_DATA"
    REQUIRES_CLINICAL_REVIEW = "REQUIRES_CLINICAL_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PathwayEngine:
    """
    Deterministic, auditable, versioned preventive pathway engine.
    Strictly separates Eligibility (program qualification) from Care Gaps (missed or due screenings).
    """

    PATHWAYS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "configs", "pathways")

    @classmethod
    def get_pathways_directory(cls) -> str:
        return cls.PATHWAYS_DIR

    @classmethod
    def list_available_pathways(cls) -> List[Dict[str, Any]]:
        """Lists all registered versioned preventive pathways."""
        pathways = []
        if not os.path.exists(cls.PATHWAYS_DIR):
            return pathways

        for filename in os.listdir(cls.PATHWAYS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(cls.PATHWAYS_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pathways.append({
                            "pathway_id": data.get("pathway_id"),
                            "name": data.get("name"),
                            "version": data.get("version"),
                            "population": data.get("population"),
                            "publishing_authority": data.get("publishing_authority"),
                            "effective_date": data.get("effective_date"),
                            "status": data.get("status"),
                            "guideline_reference": data.get("guideline_reference"),
                            "milestone_count": len(data.get("milestones", []))
                        })
                except Exception:
                    continue
        return pathways

    @classmethod
    def load_pathway(cls, pathway_id: str) -> Optional[Dict[str, Any]]:
        """Loads and parses a versioned pathway configuration by ID."""
        for filename in os.listdir(cls.PATHWAYS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(cls.PATHWAYS_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pid = data.get("pathway_id", "")
                        fname_base = filename[:-5].lower()
                        target = pathway_id.lower()
                        norm_target = pathway_id.lower().replace("_", "").replace("-", "")
                        norm_pid = pid.lower().replace("_", "").replace("-", "")
                        norm_fname = fname_base.lower().replace("_", "").replace("-", "")
                        if (
                            pid == pathway_id
                            or fname_base == target
                            or norm_target in norm_pid
                            or norm_pid in norm_target
                            or norm_target in norm_fname
                            or norm_fname in norm_target
                            or ("pediatric" in norm_target and "pediatric" in norm_fname)
                            or ("ifhas" in norm_target and ("ifhas" in norm_fname or "ifhas" in norm_pid))
                            or ("wellchild" in norm_target and "wellchild" in norm_fname)
                        ):
                            return data
                except Exception:
                    continue
        return None

    @classmethod
    def evaluate_eligibility(
        cls,
        pathway: Dict[str, Any],
        person: Person,
        evaluation_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Evaluates whether an individual satisfies the program eligibility criteria.
        Deterministic: Given the same person demographics and evaluation date, returns identical result.
        """
        eval_date = evaluation_date or date.today()
        dob = person.date_of_birth
        
        # Calculate precise age
        age_years = eval_date.year - dob.year - ((eval_date.month, eval_date.day) < (dob.month, dob.day))
        age_months = (eval_date.year - dob.year) * 12 + (eval_date.month - dob.month)
        if eval_date.day < dob.day:
            age_months -= 1
        age_months = max(0, age_months)

        criteria = pathway.get("eligibility_criteria", {})
        reasons: List[str] = []
        is_eligible = True

        # Check Active Status
        if pathway.get("status") != PathwayStatus.ACTIVE.value:
            return {
                "pathway_id": pathway.get("pathway_id"),
                "status": EligibilityStatus.NOT_ELIGIBLE.value,
                "reason_code": "PATHWAY_INACTIVE",
                "explanation": f"Pathway status is {pathway.get('status')}; only ACTIVE pathways can evaluate eligibility.",
                "clinical_authority": pathway.get("publishing_authority")
            }

        # Check Population
        pop = pathway.get("population")
        if pop == "PEDIATRIC" and age_years >= 18:
            is_eligible = False
            reasons.append(f"Person age ({age_years}y) exceeds pediatric boundary (18y)")
        elif pop == "ADULT" and age_years < 18:
            is_eligible = False
            reasons.append(f"Person age ({age_years}y) is below adult boundary (18y)")

        # Check Age Bounds (Months)
        min_m = criteria.get("min_age_months")
        max_m = criteria.get("max_age_months")
        if min_m is not None and age_months < min_m:
            is_eligible = False
            reasons.append(f"Age {age_months}m is below minimum {min_m}m")
        if max_m is not None and age_months > max_m:
            is_eligible = False
            reasons.append(f"Age {age_months}m exceeds maximum {max_m}m")

        # Check Age Bounds (Years)
        min_y = criteria.get("min_age_years")
        max_y = criteria.get("max_age_years")
        if min_y is not None and age_years < min_y:
            is_eligible = False
            reasons.append(f"Age {age_years}y is below minimum {min_y}y")
        if max_y is not None and age_years > max_y:
            is_eligible = False
            reasons.append(f"Age {age_years}y exceeds maximum {max_y}y")

        # Check Gender
        required_gender = criteria.get("gender", "ALL")
        if required_gender != "ALL" and person.gender.value != required_gender:
            is_eligible = False
            reasons.append(f"Gender {person.gender.value} does not match required gender {required_gender}")

        status_val = EligibilityStatus.ELIGIBLE if is_eligible else EligibilityStatus.NOT_ELIGIBLE
        explanation = "Eligible for preventive pathway." if is_eligible else "; ".join(reasons)

        return {
            "pathway_id": pathway.get("pathway_id"),
            "pathway_version": pathway.get("version"),
            "status": status_val.value,
            "evaluation_date": eval_date.isoformat(),
            "age_evaluated": {"years": age_years, "months": age_months},
            "is_eligible": is_eligible,
            "explanation": explanation,
            "clinical_authority": pathway.get("publishing_authority"),
            "guideline_reference": pathway.get("guideline_reference")
        }

    @classmethod
    def evaluate_care_gaps(
        cls,
        db: Session,
        pathway: Dict[str, Any],
        person: Person,
        evaluation_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates whether required milestone screenings are completed, due, overdue, or uncertain.
        CRITICAL SEPARATION: Missing historical data is tagged as NEEDS_DATA / POTENTIALLY_OVERDUE,
        NEVER assumed to be confirmed overdue without clinical evidence.
        """
        eval_date = evaluation_date or date.today()
        dob = person.date_of_birth
        
        age_years = eval_date.year - dob.year - ((eval_date.month, eval_date.day) < (dob.month, dob.day))
        age_months = (eval_date.year - dob.year) * 12 + (eval_date.month - dob.month)
        if eval_date.day < dob.day:
            age_months -= 1
        age_months = max(0, age_months)

        milestones = pathway.get("milestones", [])
        results: List[Dict[str, Any]] = []

        # Retrieve existing screening history for this patient
        screening_records = db.query(ScreeningRecord).filter(
            ScreeningRecord.patient_id == person.id
        ).all()

        for m in milestones:
            m_id = m.get("milestone_id")
            m_name = m.get("name")
            target = m.get("target_window", {})
            required_components = m.get("required_components", [])
            
            # Check milestone-level age / gender applicability
            m_gender = target.get("gender")
            if m_gender and person.gender.value != m_gender:
                results.append({
                    "milestone_id": m_id,
                    "milestone_name": m_name,
                    "status": CareGapStatus.NOT_APPLICABLE.value,
                    "reason": f"Gender specific screening ({m_gender} only).",
                    "components": required_components
                })
                continue

            # Check milestone minimum age
            m_min_age_years = target.get("min_age_years")
            if m_min_age_years is not None and age_years < m_min_age_years:
                results.append({
                    "milestone_id": m_id,
                    "milestone_name": m_name,
                    "status": CareGapStatus.NOT_APPLICABLE.value,
                    "reason": f"Patient age ({age_years}y) is below milestone threshold ({m_min_age_years}y).",
                    "components": required_components
                })
                continue

            # Find matching completed screening records
            matching_screenings = [
                s for s in screening_records
                if s.screening_type == m_id or m_id in s.screening_type
            ]

            # Case A: Completed screening exists
            completed_screenings = [
                s for s in matching_screenings
                if s.screening_status == ScreeningStatus.COMPLETED
            ]
            if completed_screenings:
                latest = max(completed_screenings, key=lambda s: s.screening_date)
                interval_months = target.get("interval_months", 36)
                months_since = (eval_date.year - latest.screening_date.year) * 12 + (eval_date.month - latest.screening_date.month)
                
                if months_since < interval_months:
                    results.append({
                        "milestone_id": m_id,
                        "milestone_name": m_name,
                        "status": CareGapStatus.COMPLETED.value,
                        "screening_date": latest.screening_date.isoformat(),
                        "months_since": months_since,
                        "reason": f"Screening completed on {latest.screening_date.isoformat()} within recurrence interval ({interval_months}m).",
                        "components": required_components
                    })
                    continue
                else:
                    # Recurrence interval expired -> DUE for repeat screening
                    results.append({
                        "milestone_id": m_id,
                        "milestone_name": m_name,
                        "status": CareGapStatus.DUE.value,
                        "last_screening_date": latest.screening_date.isoformat(),
                        "months_since": months_since,
                        "reason": f"Previous screening was {months_since}m ago, exceeding the {interval_months}m cycle interval. Repeat screening is due.",
                        "components": required_components
                    })
                    continue

            # Case B: Scheduled screening in progress
            scheduled = [s for s in matching_screenings if s.screening_status == ScreeningStatus.SCHEDULED]
            if scheduled:
                next_scr = min(scheduled, key=lambda s: s.screening_date)
                results.append({
                    "milestone_id": m_id,
                    "milestone_name": m_name,
                    "status": CareGapStatus.DUE.value,
                    "scheduled_date": next_scr.screening_date.isoformat(),
                    "reason": f"Screening appointment is scheduled for {next_scr.screening_date.isoformat()}.",
                    "components": required_components
                })
                continue

            # Case C: No record found — Check whether milestone age window has passed
            target_age_m = target.get("target_age_months")
            max_age_m = target.get("max_age_months")

            if target_age_m is not None:
                # Pediatric milestone logic
                if age_months < target.get("min_age_months", 0):
                    # Future milestone
                    results.append({
                        "milestone_id": m_id,
                        "milestone_name": m_name,
                        "status": CareGapStatus.NOT_APPLICABLE.value,
                        "reason": f"Future pediatric visit scheduled at {target_age_m}m (current age {age_months}m).",
                        "components": required_components
                    })
                elif target.get("min_age_months", 0) <= age_months <= (max_age_m or 999):
                    # Current target window -> DUE
                    results.append({
                        "milestone_id": m_id,
                        "milestone_name": m_name,
                        "status": CareGapStatus.DUE.value,
                        "reason": f"Child is currently within the recommended milestone age window ({age_months}m vs target {target_age_m}m).",
                        "components": required_components
                    })
                else:
                    # Past window and no record exists: Check if records are complete or unknown
                    has_verified_history = len(screening_records) > 0
                    if has_verified_history:
                        results.append({
                            "milestone_id": m_id,
                            "milestone_name": m_name,
                            "status": CareGapStatus.OVERDUE.value,
                            "reason": f"Milestone age window ({target_age_m}m) has elapsed with verified absence of completion record.",
                            "components": required_components
                        })
                    else:
                        results.append({
                            "milestone_id": m_id,
                            "milestone_name": m_name,
                            "status": CareGapStatus.POTENTIALLY_OVERDUE.value,
                            "reason": f"Milestone age window ({target_age_m}m) elapsed, but external clinical history is unverified. Patient history review required.",
                            "components": required_components
                        })
            else:
                # Adult milestone with no prior record
                results.append({
                    "milestone_id": m_id,
                    "milestone_name": m_name,
                    "status": CareGapStatus.DUE.value,
                    "reason": "No baseline screening recorded on file. Screening is recommended.",
                    "components": required_components
                })

        return results
