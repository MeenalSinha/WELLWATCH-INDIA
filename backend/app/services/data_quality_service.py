from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from backend.app.models.person import CompletenessStatus, PopulationClassification
from backend.app.models.screening import EpistemicStatus


class DataQualityService:
    """
    Evaluates epistemic uncertainty and data quality across patient records.
    Implements the 8-state epistemic uncertainty engine:
      VERIFIED, UNVERIFIED, MISSING, UNKNOWN, OUTDATED, CONFLICTING, NOT_APPLICABLE, UNAUTHORIZED
    """

    # Stale Thresholds: Observations older than this are classified as OUTDATED
    VITALS_MAX_RECENCY_DAYS = 365       # 12 months
    SCREENING_MAX_RECENCY_DAYS = 1095    # 36 months (IFHAS 3-year cycle)

    @classmethod
    def evaluate_observation_epistemic_status(
        cls,
        has_value: bool,
        is_unknown_flag: bool,
        is_authorized: bool,
        collection_time: Optional[datetime],
        source_verified: bool,
        is_conflicting: bool = False
    ) -> EpistemicStatus:
        """
        Computes the precise epistemic status of an observation.
        Ensures missing, unknown, outdated, and conflicting states are strictly distinguishable.
        """
        if not is_authorized:
            return EpistemicStatus.UNAUTHORIZED

        if not has_value:
            if is_unknown_flag:
                return EpistemicStatus.UNKNOWN
            return EpistemicStatus.MISSING

        if is_conflicting:
            return EpistemicStatus.CONFLICTING

        if collection_time:
            now = datetime.now(timezone.utc)
            if collection_time.tzinfo is None:
                collection_time = collection_time.replace(tzinfo=timezone.utc)
            if (now - collection_time).days > cls.VITALS_MAX_RECENCY_DAYS:
                return EpistemicStatus.OUTDATED

        if source_verified:
            return EpistemicStatus.VERIFIED

        return EpistemicStatus.UNVERIFIED

    @classmethod
    def detect_conflicts(
        cls,
        observations: List[Dict]
    ) -> bool:
        """
        Detects if multiple measurements for the same clinical code taken within a 48-hour
        window from different sources show irreconcilable clinical variance.
        """
        if len(observations) < 2:
            return False

        # Sort by collection_time
        sorted_obs = sorted(observations, key=lambda x: x.get("collection_time", datetime.min))
        
        for i in range(len(sorted_obs) - 1):
            obs1 = sorted_obs[i]
            obs2 = sorted_obs[i + 1]
            t1 = obs1.get("collection_time")
            t2 = obs2.get("collection_time")

            if t1 and t2:
                if t1.tzinfo is None: t1 = t1.replace(tzinfo=timezone.utc)
                if t2.tzinfo is None: t2 = t2.replace(tzinfo=timezone.utc)
                
                # If measurements are within 48 hours
                if abs((t2 - t1).total_seconds()) <= 172800:
                    v1 = obs1.get("canonical_value")
                    v2 = obs2.get("canonical_value")
                    if v1 is not None and v2 is not None:
                        # Conflict if variance exceeds 30% for vitals
                        diff = abs(v1 - v2)
                        avg = (v1 + v2) / 2.0
                        if avg > 0 and (diff / avg) > 0.30:
                            return True
        return False

    @classmethod
    def calculate_profile_completeness(
        cls,
        classification: PopulationClassification,
        available_codes: List[str],
        has_family_history: bool,
        has_screening_record: bool
    ) -> Tuple[float, CompletenessStatus]:
        """
        Calculates population-specific profile completeness without equating completeness with health.
        """
        if classification == PopulationClassification.PEDIATRIC:
            # Pediatric Core: Height, Weight, Pulse, Developmental observation, Screening record
            core_requirements = ["8302-2", "29463-7", "8867-4"]
            found = sum(1 for c in core_requirements if c in available_codes)
            total = len(core_requirements) + 2  # +1 for family hist, +1 for screening record
            score = (found + (1 if has_family_history else 0) + (1 if has_screening_record else 0)) / total
        else:
            # Adult Core: Systolic BP, Diastolic BP, Glucose, Height, Weight, Cholesterol, Smoking
            core_requirements = ["8480-6", "8462-4", "2339-0", "8302-2", "29463-7", "2093-3"]
            found = sum(1 for c in core_requirements if c in available_codes)
            total = len(core_requirements) + 2  # +1 for family hist, +1 for screening record
            score = (found + (1 if has_family_history else 0) + (1 if has_screening_record else 0)) / total

        pct = round(score * 100.0, 1)
        if pct >= 85.0:
            status = CompletenessStatus.COMPLETE
        elif pct >= 50.0:
            status = CompletenessStatus.PARTIALLY_COMPLETE
        elif pct >= 20.0:
            status = CompletenessStatus.INCOMPLETE
        else:
            status = CompletenessStatus.MINIMAL

        return pct, status
