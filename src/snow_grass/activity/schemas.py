from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ActivityEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_event_id: str = Field(min_length=1, max_length=64)
    app_name: str = Field(min_length=1, max_length=200)
    bundle_id: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=4096)
    occurred_at: datetime


class ActivityBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[ActivityEventInput] = Field(min_length=1, max_length=100)


class ActivityBatchResponse(BaseModel):
    accepted: int
    duplicate: int


class ActivityDeleteResponse(BaseModel):
    deleted: int


class ActivityEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    client_event_id: str
    app_name: str
    bundle_id: str
    text: str
    occurred_at: datetime
    analysis_source: ActivityAnalysisSource | None = None
    analysis_category: ActivityAnalysisCategory | None = None
    analysis_action: ActivityAction | None = None
    analysis_severity: int | None = None
    analysis_should_respond: bool | None = None
    analyzed_at: datetime | None = None

    _normalize_response_datetimes = field_validator(
        "occurred_at", "analyzed_at", mode="after"
    )(_as_utc)


class ActivityPageResponse(BaseModel):
    items: list[ActivityEventResponse]
    total: int
    raw_total: int
    duplicate_total: int
    deduplicated: bool
    page: int
    page_size: int


class ActivityDailySummary(BaseModel):
    date: date
    event_count: int
    character_count: int


class ActivityAppSummary(BaseModel):
    app_name: str
    bundle_id: str
    event_count: int
    character_count: int


class ActivitySummaryResponse(BaseModel):
    raw_event_count: int
    unique_event_count: int
    duplicate_event_count: int
    character_count: int
    active_app_count: int
    daily: list[ActivityDailySummary]
    apps: list[ActivityAppSummary]
    generated_at: datetime


class ActivityAction(StrEnum):
    idle = "idle"
    like = "like"
    wave = "wave"
    dance = "dance"
    sleep = "sleep"
    angry = "angry"
    comfort = "comfort"


class ActivityAnalysisCategory(StrEnum):
    none = "none"
    frustrated = "frustrated"
    anxious = "anxious"
    angry = "angry"
    tired = "tired"
    sad = "sad"
    high_risk = "high_risk"


class ActivityAnalysisSource(StrEnum):
    rules = "rules"
    local = "local"
    deepseek = "deepseek"


class ActivityAnalyzeRequest(BaseModel):
    client_event_id: str = Field(min_length=1, max_length=64)
    available_actions: list[ActivityAction] = Field(min_length=1, max_length=7)


class ActivityAnalyzeResponse(BaseModel):
    should_respond: bool
    category: ActivityAnalysisCategory = ActivityAnalysisCategory.none
    severity: int = Field(default=0, ge=0, le=3)
    reply: str | None = Field(default=None, max_length=120)
    action: ActivityAction = ActivityAction.idle
    is_fixed_safety_reply: bool = False
    analysis_source: ActivityAnalysisSource


class ActivityAnalysisCategoryStats(BaseModel):
    category: ActivityAnalysisCategory
    total: int
    local: int = 0
    deepseek: int = 0
    rules: int = 0


class ActivityAnalysisStatsResponse(BaseModel):
    total_event_count: int
    analyzed_count: int
    pending_count: int
    local_count: int
    deepseek_count: int
    rules_count: int
    responded_count: int
    categories: list[ActivityAnalysisCategoryStats]
    generated_at: datetime


class HourlySummaryStatus(StrEnum):
    generated = "generated"
    insufficient_activity = "insufficient_activity"
    already_generated = "already_generated"


class HourlySummaryRequest(BaseModel):
    period_start: datetime
    timezone_offset_minutes: int = Field(default=480, ge=-720, le=840)


class HourlySummaryResponse(BaseModel):
    id: str
    period_start: datetime
    period_end: datetime
    status: HourlySummaryStatus
    title: str = Field(default="", max_length=240)
    keywords: list[str] = Field(default_factory=list, max_length=30)
    completed: list[str] = Field(default_factory=list)
    in_progress: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    event_count: int = Field(ge=0)
    generated_at: datetime
    flow_version_id: str | None = None
    flow_run_id: str | None = None
    quality_status: str | None = None

    _normalize_response_datetimes = field_validator(
        "period_start", "period_end", "generated_at", mode="after"
    )(_as_utc)
