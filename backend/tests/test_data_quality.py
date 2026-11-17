from datetime import datetime, timezone, timedelta
import pytest
from backend.app.models.person import CompletenessStatus, PopulationClassification
from backend.app.models.screening import EpistemicStatus
from backend.app.services.data_quality_service import DataQualityService


def test_epistemic_status_distinctions():
    # 1. Unauthorized
    status_unauth = DataQualityService.evaluate_observation_epistemic_status(
        has_value=True,
        is_unknown_flag=False,
        is_authorized=False,
        collection_time=datetime.now(timezone.utc),
        source_verified=True
    )
    assert status_unauth == EpistemicStatus.UNAUTHORIZED

    # 2. Missing
    status_missing = DataQualityService.evaluate_observation_epistemic_status(
        has_value=False,
        is_unknown_flag=False,
        is_authorized=True,
        collection_time=None,
        source_verified=True
    )
    assert status_missing == EpistemicStatus.MISSING

    # 3. Unknown
    status_unknown = DataQualityService.evaluate_observation_epistemic_status(
        has_value=False,
        is_unknown_flag=True,
        is_authorized=True,
        collection_time=None,
        source_verified=True
    )
    assert status_unknown == EpistemicStatus.UNKNOWN

    # 4. Outdated (older than 365 days)
    old_date = datetime.now(timezone.utc) - timedelta(days=400)
    status_outdated = DataQualityService.evaluate_observation_epistemic_status(
        has_value=True,
        is_unknown_flag=False,
        is_authorized=True,
        collection_time=old_date,
        source_verified=True
    )
    assert status_outdated == EpistemicStatus.OUTDATED

    # 5. Verified
    recent_date = datetime.now(timezone.utc) - timedelta(days=10)
    status_verified = DataQualityService.evaluate_observation_epistemic_status(
        has_value=True,
        is_unknown_flag=False,
        is_authorized=True,
        collection_time=recent_date,
        source_verified=True
    )
    assert status_verified == EpistemicStatus.VERIFIED


def test_conflict_detection():
    now = datetime.now(timezone.utc)
    # Two measurements within 2 hours with > 30% difference (110 vs 170)
    obs = [
        {"collection_time": now - timedelta(hours=2), "canonical_value": 110.0},
        {"collection_time": now, "canonical_value": 170.0},
    ]
    is_conflicting = DataQualityService.detect_conflicts(obs)
    assert is_conflicting is True

    # Two measurements within 2 hours with minimal variance (118 vs 122)
    obs_ok = [
        {"collection_time": now - timedelta(hours=2), "canonical_value": 118.0},
        {"collection_time": now, "canonical_value": 122.0},
    ]
    assert DataQualityService.detect_conflicts(obs_ok) is False


def test_profile_completeness_calculation():
    # Adult with full set of observations + family history + screening
    adult_codes = ["8480-6", "8462-4", "2339-0", "8302-2", "29463-7", "2093-3"]
    score, status = DataQualityService.calculate_profile_completeness(
        classification=PopulationClassification.ADULT,
        available_codes=adult_codes,
        has_family_history=True,
        has_screening_record=True
    )
    assert score == 100.0
    assert status == CompletenessStatus.COMPLETE

    # Adult with only 1 vital and no family history or screening
    score_low, status_low = DataQualityService.calculate_profile_completeness(
        classification=PopulationClassification.ADULT,
        available_codes=["8480-6"],
        has_family_history=False,
        has_screening_record=False
    )
    assert score_low < 20.0
    assert status_low == CompletenessStatus.MINIMAL
