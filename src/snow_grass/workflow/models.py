from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from snow_grass.persistence.models import Base, utc_now


class ComponentDefinitionRecord(Base):
    __tablename__ = "component_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(120), index=True)
    icon: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    draft_revision: Mapped[int] = mapped_column(Integer, default=0)
    draft_manifest: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ComponentVersionRecord(Base):
    __tablename__ = "component_versions"
    __table_args__ = (
        UniqueConstraint("component_id", "version_number", name="uq_component_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    component_id: Mapped[str] = mapped_column(
        ForeignKey("component_definitions.id", ondelete="CASCADE"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    icon: Mapped[str] = mapped_column(String(120), default="tool")
    implementation_key: Mapped[str] = mapped_column(String(200), index=True)
    implementation_digest: Mapped[str] = mapped_column(String(128))
    implementation_source: Mapped[str] = mapped_column(Text, default="")
    source_language: Mapped[str] = mapped_column(String(40), default="python")
    source_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    executor_contract_version: Mapped[int] = mapped_column(Integer, default=1)
    config_schema: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    ui_schema: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    input_ports: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    output_ports: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    release_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkflowRecord(Base):
    __tablename__ = "workflows"
    __table_args__ = (UniqueConstraint("business_type", "name", name="uq_workflow_business_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    business_type: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    draft_revision: Mapped[int] = mapped_column(Integer, default=0)
    draft_graph: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkflowVersionRecord(Base):
    __tablename__ = "workflow_versions"
    __table_args__ = (
        UniqueConstraint("workflow_id", "version_number", name="uq_workflow_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    graph: Mapped[dict[str, object]] = mapped_column(JSON)
    component_manifest_digest: Mapped[str] = mapped_column(String(128), default="")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkflowDeploymentRecord(Base):
    __tablename__ = "workflow_deployments"
    __table_args__ = (UniqueConstraint("workflow_id", name="uq_workflow_active_deployment"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[str] = mapped_column(ForeignKey("workflow_versions.id"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkflowDeploymentEventRecord(Base):
    __tablename__ = "workflow_deployment_events"
    __table_args__ = (
        Index("ix_workflow_deployment_events_workflow_time", "workflow_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    from_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_versions.id"), nullable=True
    )
    to_version_id: Mapped[str] = mapped_column(ForeignKey("workflow_versions.id"))
    event_type: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WorkflowRunRecord(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (Index("ix_workflow_runs_workflow_started", "workflow_id", "started_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_versions.id"), nullable=True
    )
    resolved_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_versions.id"), nullable=True
    )
    business_type: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20))
    preview: Mapped[bool] = mapped_column(Boolean, default=False)
    input_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    output_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowNodeRunRecord(Base):
    __tablename__ = "workflow_node_runs"
    __table_args__ = (Index("ix_workflow_node_runs_run_order", "run_id", "sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    node_id: Mapped[str] = mapped_column(String(120))
    node_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    component_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("component_versions.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(20))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    output_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    input_port_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    output_port_summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    skip_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
