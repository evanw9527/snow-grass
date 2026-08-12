from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComponentStatus = Literal["active", "disabled", "archived"]


class PortDefinition(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    name: str = Field(min_length=1, max_length=120)
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    required: bool = True
    multiple: bool = False

    model_config = ConfigDict(populate_by_name=True)


class ComponentManifest(BaseModel):
    icon: str = Field(default="tool", min_length=1, max_length=120)
    implementation_key: str = Field(min_length=1, max_length=200)
    executor_contract_version: int = Field(default=1, ge=1)
    config_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "additionalProperties": False}
    )
    ui_schema: dict[str, Any] = Field(default_factory=dict)
    input_ports: list[PortDefinition] = Field(default_factory=list, max_length=32)
    output_ports: list[PortDefinition] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def unique_port_keys(self) -> ComponentManifest:
        for direction, ports in (("input", self.input_ports), ("output", self.output_ports)):
            keys = [port.key for port in ports]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {direction} port key")
        return self


class ComponentVersionManifest(ComponentManifest):
    id: str
    component_id: str
    component_key: str
    version_number: int = Field(ge=1)
    implementation_digest: str = Field(min_length=1, max_length=128)
    status: ComponentStatus = "active"


class ComponentCreateRequest(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$", max_length=160)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    category: str = Field(min_length=1, max_length=120)
    icon: str = Field(default="tool", min_length=1, max_length=120)
    implementation_key: str = Field(min_length=1, max_length=200)


class ComponentDraftRequest(ComponentManifest):
    expected_revision: int = Field(ge=0)


class ComponentPublishRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    release_note: str | None = Field(default=None, max_length=500)


class ComponentStatusRequest(BaseModel):
    status: ComponentStatus


class ComponentTestRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)


class ComponentSummary(BaseModel):
    id: str
    key: str
    name: str
    description: str = ""
    category: str
    icon: str | None = None
    status: ComponentStatus
    draft_revision: int
    latest_version: int | None = None
    reference_count: int = 0
    updated_at: datetime


class ComponentVersionResponse(ComponentVersionManifest):
    component_name: str = ""
    component_description: str = ""
    category: str = ""
    release_note: str | None = None
    published_at: datetime
    implementation_source: str = ""
    source_language: str = "python"
    source_path: str | None = None


class ComponentDetail(ComponentSummary):
    draft: ComponentManifest
    versions: list[ComponentVersionResponse] = Field(default_factory=list)


class ComponentPublishResponse(BaseModel):
    component_version_id: str
    version_number: int
    implementation_digest: str
    published_at: datetime


class ComponentTestResponse(BaseModel):
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = Field(ge=0)
    valid: bool = True
    logs: list[str] = Field(default_factory=list)


class FlowNode(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    component_version_id: str = Field(min_length=1, max_length=120)
    name: str = Field(default="", max_length=200)
    config: dict[str, Any] = Field(default_factory=dict)
    x: float = 0
    y: float = 0
    disabled: bool = False


class EdgeCondition(BaseModel):
    path: str = Field(default="", max_length=240)
    operator: Literal[
        "equals",
        "not_equals",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "contains",
        "truthy",
        "falsy",
        "exists",
        "not_exists",
    ] = "truthy"
    value: Any = None


class FlowEdge(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    source_node_id: str = Field(min_length=1, max_length=120)
    source_port: str = Field(min_length=1, max_length=80)
    target_node_id: str = Field(min_length=1, max_length=120)
    target_port: str = Field(min_length=1, max_length=80)
    condition: EdgeCondition | None = None


class FlowGraph(BaseModel):
    nodes: list[FlowNode] = Field(default_factory=list, max_length=100)
    edges: list[FlowEdge] = Field(default_factory=list, max_length=300)


class ValidationIssue(BaseModel):
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    port: str | None = None


class FlowValidation(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)


class WorkflowCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    business_type: str = Field(min_length=1, max_length=120)
    template_id: str | None = None


class WorkflowCopyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class WorkflowStatusRequest(BaseModel):
    status: Literal["active", "archived"]


class WorkflowVersionSummary(BaseModel):
    id: str
    workflow_id: str
    version_number: int
    note: str | None = None
    component_manifest_digest: str
    published_at: datetime


class WorkflowSummary(BaseModel):
    id: str
    business_type: str
    name: str
    description: str = ""
    status: Literal["active", "archived"] = "active"
    draft_revision: int
    latest_version: int | None = None
    deployed_version_id: str | None = None
    updated_at: datetime


class WorkflowDetail(WorkflowSummary):
    graph: FlowGraph
    validation: FlowValidation | None = None


class SaveDraftRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    graph: FlowGraph


class PublishRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    version_note: str | None = Field(default=None, max_length=500)
    activate: bool = False


class PublishResponse(BaseModel):
    version_id: str
    version_number: int
    deployment_id: str | None = None
    active: bool
    published_at: datetime


class DeploymentRequest(BaseModel):
    version_id: str
    expected_current_version_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class RollbackRequest(BaseModel):
    target_version_id: str
    expected_current_version_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class DeploymentResponse(BaseModel):
    deployment_id: str
    workflow_id: str
    from_version_id: str | None = None
    version_id: str
    event_type: Literal["deploy", "rollback"]
    reason: str | None = None
    updated_at: datetime


class PreviewRequest(BaseModel):
    draft_revision: int = Field(ge=0)
    input: dict[str, Any] = Field(default_factory=dict)


class NodeTrace(BaseModel):
    node_id: str
    component_version_id: str | None = None
    node_type: str | None = None
    status: Literal["succeeded", "failed", "skipped"]
    duration_ms: int = Field(ge=0)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    input_port_summary: dict[str, Any] = Field(default_factory=dict)
    output_port_summary: dict[str, Any] = Field(default_factory=dict)
    skip_reason: str | None = None
    error: str | None = None


class FlowRunResponse(BaseModel):
    run_id: str
    workflow_id: str
    flow_version_id: str | None = None
    resolved_version_id: str | None = None
    business_type: str
    status: Literal["running", "succeeded", "failed"]
    preview: bool
    output: dict[str, Any] = Field(default_factory=dict)
    trace: list[NodeTrace] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None


class BusinessPackResponse(BaseModel):
    business_type: str
    title: str
    description: str
    allowed_component_keys: list[str]
