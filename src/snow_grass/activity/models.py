from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from snow_grass.persistence.models import Base, utc_now


class ActivityEventRecord(Base):
    __tablename__ = "activity_events"
    __table_args__ = (
        UniqueConstraint("client_event_id", name="uq_activity_client_event"),
        Index("ix_activity_occurred", "occurred_at"),
        Index("ix_activity_bundle_occurred", "bundle_id", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    client_event_id: Mapped[str] = mapped_column(String(64))
    app_name: Mapped[str] = mapped_column(String(200))
    bundle_id: Mapped[str] = mapped_column(String(240))
    text: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    analysis_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    analysis_category: Mapped[str | None] = mapped_column(String(24), nullable=True)
    analysis_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    analysis_severity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analysis_should_respond: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class HourlyActivitySummaryRecord(Base):
    __tablename__ = "hourly_activity_summaries"
    __table_args__ = (
        UniqueConstraint("period_start", name="uq_hourly_activity_summary_period"),
        Index("ix_hourly_activity_summary_period", "period_start"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(240), default="", server_default="")
    keywords_json: Mapped[str] = mapped_column(Text, default="[]", server_default="[]")
    completed_json: Mapped[str] = mapped_column(Text, default="[]")
    in_progress_json: Mapped[str] = mapped_column(Text, default="[]")
    blockers_json: Mapped[str] = mapped_column(Text, default="[]")
    next_steps_json: Mapped[str] = mapped_column(Text, default="[]")
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    flow_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    flow_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    quality_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
