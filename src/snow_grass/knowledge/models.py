from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from snow_grass.persistence.models import Base, utc_now


class KnowledgeDocumentRecord(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "source_type", "source_id", name="uq_knowledge_source"
        ),
        Index("ix_knowledge_workspace_status", "workspace_id", "status"),
        Index("ix_knowledge_source_time", "source_type", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(120), default="default")
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(240), default="")
    content: Mapped[str] = mapped_column(Text)
    keywords_json: Mapped[str] = mapped_column(Text, default="[]")
    source_app: Mapped[str | None] = mapped_column(String(240), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class KnowledgeBaseRecord(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "source_type", name="uq_knowledge_base_source"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(120), default="default")
    source_type: Mapped[str] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
