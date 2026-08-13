from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from snow_grass.skills.schema import SkillDraftContent, SkillValidationReport


class CreateSessionRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=200)
    model_id: str | None = None
    skill_id: str | None = None
    knowledge_enabled: bool = False
    runtime_id: str = Field(default="native", pattern=r"^[a-z][a-z0-9-]*$", max_length=40)


class AgentRuntimeResponse(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    capabilities: list[str]
    unavailable_reason: str | None = None


class UpdateSessionKnowledgeRequest(BaseModel):
    enabled: bool


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    model_id: str
    skill_id: str | None
    knowledge_enabled: bool
    runtime_id: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    model_id: str | None
    skill_id: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    created_at: datetime


class ModelUsageSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int


class UsageSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    period_days: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    sessions: int
    by_model: list[ModelUsageSummaryResponse]


class LocalUsageSourceResponse(BaseModel):
    id: str
    name: str
    available: bool
    total_tokens: int
    sessions: int


class LocalUsagePointResponse(BaseModel):
    date: str
    total_tokens: int


class LocalUsageModelResponse(BaseModel):
    source: str
    name: str
    total_tokens: int
    sessions: int


class LocalUsageProjectResponse(BaseModel):
    name: str
    path: str
    total_tokens: int
    sessions: int


class LocalUsageResponse(BaseModel):
    period_days: int
    total_tokens: int
    sessions: int
    sources: list[LocalUsageSourceResponse]
    daily: list[LocalUsagePointResponse]
    models: list[LocalUsageModelResponse]
    projects: list[LocalUsageProjectResponse]
    updated_at: datetime


class StreamMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    model_id: str | None = None
    skill_id: str | None = None
    knowledge_enabled: bool | None = None


MemoryScope = Literal["session", "workspace"]
MemoryStatus = Literal["candidate", "active", "rejected", "deleted"]


class SessionSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    content: str
    covered_until_message_id: str
    covered_message_count: int
    estimated_tokens: int
    created_at: datetime


class ContextStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    token_budget: int
    estimated_tokens: int
    summary_tokens: int
    memory_tokens: int
    recent_message_tokens: int
    recent_message_count: int
    memory_count: int
    knowledge_tokens: int = 0
    knowledge_count: int = 0
    knowledge_degraded_reason: str | None = None
    trimmed_message_count: int
    summary_version: int | None
    degraded_reason: str | None


class SessionMemoryResponse(BaseModel):
    summary: SessionSummaryResponse | None
    hot_message_count: int
    cold_message_count: int
    last_context_stats: ContextStatsResponse | None


class CompactMemoryRequest(BaseModel):
    force: bool = False


class CompactMemoryResponse(BaseModel):
    compacted: bool
    summary_version: int | None
    covered_message_count: int
    reason: str | None


class CreateMemoryRequest(BaseModel):
    scope_type: MemoryScope
    scope_id: str = Field(min_length=1, max_length=120)
    memory_type: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=20_000)
    data: dict[str, object] | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    expires_at: datetime | None = None


class UpdateMemoryRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    status: MemoryStatus | None = None
    importance: float | None = Field(default=None, ge=0, le=1)
    expires_at: datetime | None = None


class MemoryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scope_type: str
    scope_id: str
    memory_type: str
    content: str
    data: dict[str, object] | None
    importance: float
    confidence: float
    source_session_id: str | None
    source_message_id: str | None
    status: str
    access_count: int
    last_accessed_at: datetime | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CreateSkillRequest(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$", max_length=120)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1_000)


class SaveSkillDraftRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    content: SkillDraftContent


class ValidateSkillRequest(BaseModel):
    expected_revision: int = Field(ge=1)


class PublishSkillRequest(BaseModel):
    version: str = Field(min_length=5, max_length=40)
    expected_revision: int = Field(ge=1)
    activate: bool = True
    release_notes: str | None = Field(default=None, max_length=5_000)


class SetSkillEnabledRequest(BaseModel):
    enabled: bool


class CloneSkillRequest(BaseModel):
    new_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$", max_length=120)
    new_name: str = Field(min_length=1, max_length=120)
    release_id: str | None = None


class ArchiveSkillRequest(BaseModel):
    expected_revision: int = Field(ge=1)


class SkillDraftResponse(BaseModel):
    revision: int
    content: SkillDraftContent
    content_hash: str
    validated_hash: str | None
    validation_report: SkillValidationReport | None
    updated_at: datetime | None


class SkillReleaseResponse(BaseModel):
    id: str
    version: str
    checksum: str
    release_notes: str | None
    published_at: datetime | None
    active: bool


class SkillAuditResponse(BaseModel):
    id: str
    action: str
    release_id: str | None
    revision: int | None
    details: dict[str, object] | None
    actor: str
    created_at: datetime


class AdminSkillSummaryResponse(BaseModel):
    id: str
    source: Literal["builtin", "managed"]
    name: str
    description: str
    lifecycle_status: str
    enabled: bool
    draft_revision: int | None
    active_version: str | None
    has_unpublished_changes: bool
    updated_at: datetime | None


class SkillDetailResponse(BaseModel):
    id: str
    source: Literal["builtin", "managed"]
    lifecycle_status: str
    enabled: bool
    draft: SkillDraftResponse | None
    active_release: SkillReleaseResponse | None
    releases: list[SkillReleaseResponse]
    audits: list[SkillAuditResponse]
    readonly: bool


class SkillActivationResponse(BaseModel):
    previous_version: str | None
    active_version: str
    activated_at: datetime


class ArchiveSkillResponse(BaseModel):
    id: str
    lifecycle_status: str
    enabled: bool
