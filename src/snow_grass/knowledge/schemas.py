from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnowledgeSourceType(StrEnum):
    activity_event = "activity_event"
    hourly_summary = "hourly_summary"


class KnowledgeDocumentStatus(StrEnum):
    active = "active"
    disabled = "disabled"
    deleted = "deleted"


class KnowledgeBatchAction(StrEnum):
    enable = "enable"
    disable = "disable"
    delete = "delete"


class KnowledgeDocumentInput(BaseModel):
    source_type: KnowledgeSourceType
    source_id: str = Field(min_length=1, max_length=64)
    title: str = Field(default="", max_length=240)
    content: str = Field(min_length=1, max_length=50_000)
    keywords: list[str] = Field(default_factory=list, max_length=30)
    source_app: str | None = Field(default=None, max_length=240)
    occurred_at: datetime


class KnowledgeDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_type: KnowledgeSourceType
    source_id: str
    title: str
    content: str
    keywords: list[str] = Field(default_factory=list)
    source_app: str | None
    occurred_at: datetime
    status: KnowledgeDocumentStatus
    created_at: datetime
    updated_at: datetime


class KnowledgePageResponse(BaseModel):
    items: list[KnowledgeDocumentResponse]
    total: int
    page: int
    page_size: int


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=8, ge=1, le=20)
    source_types: list[KnowledgeSourceType] | None = None
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class KnowledgeSearchItem(BaseModel):
    id: str
    content: str
    title: str
    source_type: KnowledgeSourceType
    source_id: str
    source_app: str | None
    occurred_at: datetime
    score: float


class KnowledgeSearchResponse(BaseModel):
    items: list[KnowledgeSearchItem]
    degraded_reason: str | None = None


class KnowledgeStatsResponse(BaseModel):
    globally_enabled: bool
    index_mode: str
    total: int
    active: int
    disabled: int
    deleted: int
    activity_events: int
    hourly_summaries: int
    last_indexed_at: datetime | None


class KnowledgeBaseResponse(BaseModel):
    id: str
    source_type: KnowledgeSourceType
    name: str
    description: str
    enabled: bool
    total: int
    active: int
    deleted: int
    last_indexed_at: datetime | None


class KnowledgeBaseUpdateRequest(BaseModel):
    enabled: bool


class KnowledgeStatusUpdateRequest(BaseModel):
    status: KnowledgeDocumentStatus

    @field_validator("status")
    @classmethod
    def deleted_requires_delete_endpoint(
        cls, value: KnowledgeDocumentStatus
    ) -> KnowledgeDocumentStatus:
        if value is KnowledgeDocumentStatus.deleted:
            raise ValueError("Use DELETE to remove a knowledge document")
        return value


class KnowledgeBatchRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)
    action: KnowledgeBatchAction


class KnowledgeBatchResponse(BaseModel):
    matched: int
    updated: int
    skipped: int


class KnowledgeReindexRequest(BaseModel):
    source_types: list[KnowledgeSourceType] | None = None
    start: datetime | None = None
    restore_deleted: bool = False


class KnowledgeReindexResponse(BaseModel):
    indexed: int
    updated: int
    skipped: int
    failed: int
