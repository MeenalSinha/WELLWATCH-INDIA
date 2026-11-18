from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field


class PopulationOverviewResponse(BaseModel):
    total_registered: int
    pediatric_population: int
    adult_population: int
    geriatric_population: int
    incomplete_profiles: int
    eligible_population: int
    completed_screenings: int
    screening_coverage_percentage: float
    confirmed_overdue: int
    potentially_overdue: int
    recommendations_generated: int
    recommendations_approved: int
    recommendations_accepted: int
    recommendations_declined: int
    recommendations_completed: int
    appointments_booked: int
    appointments_attended: int
    no_show_rate: float
    referrals_created: int
    referrals_completed: int
    care_gaps_closed: int
    reporting_window_days: int
    generated_at: datetime


class PediatricPopulationResponse(BaseModel):
    total_pediatric: int
    eligible_for_well_child: int
    completed_well_child: int
    well_child_coverage_percentage: float
    overdue_well_child: int
    developmental_stages: Dict[str, int]
    open_pediatric_care_gaps: int
    guardian_authorized_rate: float
    generated_at: datetime


class AdultPopulationResponse(BaseModel):
    total_adult: int
    eligible_for_ifhas: int
    completed_ifhas: int
    ifhas_coverage_percentage: float
    overdue_ifhas: int
    cardiometabolic_risk_distribution: Dict[str, int]
    open_adult_care_gaps: int
    referral_completion_rate: float
    generated_at: datetime


class GeographicRegionMetric(BaseModel):
    region_code: str
    region_name_en: str
    region_name_ar: str
    registered_population: Union[int, str]
    eligible_population: Union[int, str]
    completed_screenings: Union[int, str]
    screening_coverage_percentage: Optional[float] = None
    is_suppressed: bool = False


class GeographicAnalyticsResponse(BaseModel):
    regions: List[GeographicRegionMetric]
    small_cell_threshold: int
    suppression_applied: bool
    generated_at: datetime


class PopulationEquityResponse(BaseModel):
    benchmark_region: str
    subgroup_metrics: List[Dict[str, Any]]
    disparity_ratios: Dict[str, float]
    data_completeness_by_subgroup: Dict[str, float]
    ml_fairness_evaluation: Dict[str, Any]
    generated_at: datetime


class TrendPoint(BaseModel):
    period: str
    completed_screenings: int
    new_care_gaps: int
    closed_care_gaps: int
    screening_coverage_percentage: float


class PopulationTrendsResponse(BaseModel):
    timeframe: str
    trend_points: List[TrendPoint]
    generated_at: datetime


class ReportGenerateRequest(BaseModel):
    report_type: str = Field(..., description="POPULATION_SUMMARY, PEDIATRIC_COVERAGE, ADULT_IFHAS, or EQUITY_AUDIT")
    time_window_days: int = Field(90, ge=7, le=365)
    organization_id: Optional[str] = None


class PopulationReportResponse(BaseModel):
    report_id: str
    report_type: str
    organization_id: Optional[str]
    time_window_days: int
    generated_by: str
    generated_at: datetime
    metrics_summary: Dict[str, Any]
    policy_governance: Dict[str, Any]
