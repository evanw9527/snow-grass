from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    model_id: Mapped[str] = mapped_column(String(120))
    skill_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    knowledge_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class MessageRecord(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    skill_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SessionSummaryRecord(Base):
    __tablename__ = "session_summaries"
    __table_args__ = (
        UniqueConstraint("session_id", "version", name="uq_session_summary_version"),
        Index("ix_session_summary_current", "session_id", "is_current"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    covered_until_message_id: Mapped[str] = mapped_column(String(36))
    covered_message_count: Mapped[int] = mapped_column(Integer)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryItemRecord(Base):
    __tablename__ = "memory_items"
    __table_args__ = (
        Index("ix_memory_scope_status", "scope_type", "scope_id", "status"),
        Index("ix_memory_expiration", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(20))
    scope_id: Mapped[str] = mapped_column(String(120))
    memory_type: Mapped[str] = mapped_column(String(40))
    content: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_message_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ContextRunRecord(Base):
    __tablename__ = "context_runs"
    __table_args__ = (Index("ix_context_runs_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(String(120))
    token_budget: Mapped[int] = mapped_column(Integer)
    estimated_tokens: Mapped[int] = mapped_column(Integer)
    summary_tokens: Mapped[int] = mapped_column(Integer, default=0)
    memory_tokens: Mapped[int] = mapped_column(Integer, default=0)
    recent_message_tokens: Mapped[int] = mapped_column(Integer, default=0)
    recent_message_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_count: Mapped[int] = mapped_column(Integer, default=0)
    trimmed_message_count: Mapped[int] = mapped_column(Integer, default=0)
    summary_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    degraded_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ToolResultCacheRecord(Base):
    __tablename__ = "tool_result_cache"
    __table_args__ = (
        UniqueConstraint("cache_key", name="uq_tool_result_cache_key"),
        Index("ix_tool_result_cache_expires", "expires_at"),
        Index("ix_tool_result_cache_skill", "skill_id", "skill_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(120))
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    scope_type: Mapped[str] = mapped_column(String(20))
    scope_id: Mapped[str] = mapped_column(String(120))
    skill_id: Mapped[str] = mapped_column(String(120))
    skill_version: Mapped[str] = mapped_column(String(40))
    script: Mapped[str] = mapped_column(String(240))
    arguments_json: Mapped[list[str]] = mapped_column(JSON)
    cache_key: Mapped[str] = mapped_column(String(64))
    result_json: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20))
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class CodexUsageEventRecord(Base):
    __tablename__ = "codex_usage_events"
    __table_args__ = (
        Index("ix_codex_usage_events_occurred", "occurred_at"),
        Index("ix_codex_usage_events_model_occurred", "model_id", "occurred_at"),
        Index("ix_codex_usage_events_project_occurred", "project_path", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(36), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_id: Mapped[str] = mapped_column(String(120))
    project_path: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CodexUsageCursorRecord(Base):
    __tablename__ = "codex_usage_cursors"

    source_path: Mapped[str] = mapped_column(Text, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(36))
    byte_offset: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    current_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    project_path: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ManagedSkillRecord(Base):
    __tablename__ = "managed_skills"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(String(1000))
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="draft")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    active_release_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class SkillDraftRecord(Base):
    __tablename__ = "skill_drafts"

    skill_id: Mapped[str] = mapped_column(
        ForeignKey("managed_skills.id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[dict[str, object]] = mapped_column(JSON)
    content_hash: Mapped[str] = mapped_column(String(64))
    validated_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_report: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class SkillReleaseRecord(Base):
    __tablename__ = "skill_releases"
    __table_args__ = (
        UniqueConstraint("skill_id", "version", name="uq_skill_release_version"),
        Index("ix_skill_releases_skill_published", "skill_id", "published_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    skill_id: Mapped[str] = mapped_column(
        ForeignKey("managed_skills.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(40))
    content: Mapped[dict[str, object]] = mapped_column(JSON)
    checksum: Mapped[str] = mapped_column(String(64))
    release_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SkillRuntimeStateRecord(Base):
    __tablename__ = "skill_runtime_states"

    skill_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class SkillAuditRecord(Base):
    __tablename__ = "skill_audits"
    __table_args__ = (Index("ix_skill_audits_skill_created", "skill_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    skill_id: Mapped[str] = mapped_column(String(120), index=True)
    action: Mapped[str] = mapped_column(String(40))
    release_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    actor: Mapped[str] = mapped_column(String(120), default="local-user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
