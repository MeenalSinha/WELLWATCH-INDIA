import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Float, Boolean, Text, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.models.base import BaseDBModel


class MetricCategory(str, enum.Enum):
    OVERVIEW = "OVERVIEW"
    PEDIATRIC = "PEDIATRIC"
    ADULT = "ADULT"
    GEOGRAPHIC = "GEOGRAPHIC"
    EQUITY = "EQUITY"
    NAVIGATION = "NAVIGATION"
    CARE_GAP = "CARE_GAP"


class PopulationMetricSnapshot(BaseDBModel):
    """Materialized snapshot of population health metrics for longitudinal trends and performance tracking."""
    __tablename__ = "population_metric_snapshots"

    organization_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    metric_category: Mapped[str] = mapped_column(String(50), index=True)
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    dimension_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dimension_value: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    numerator: Mapped[float] = mapped_column(Float, default=0.0)
    denominator: Mapped[float] = mapped_column(Float, default=0.0)
    metric_value: Mapped[float] = mapped_column(Float, default=0.0)
    is_suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AnalyticsAccessLog(BaseDBModel):
    """Audit log for aggregate population queries to track access and mitigate differencing attacks."""
    __tablename__ = "analytics_access_logs"

    user_id: Mapped[str] = mapped_column(String(36), index=True)
    organization_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    endpoint: Mapped[str] = mapped_column(String(255))
    query_params: Mapped[str] = mapped_column(Text, default="{}")
    cell_count: Mapped[int] = mapped_column(Integer, default=0)
    was_suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    accessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
