from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from snow_grass.workflow.registry import BusinessPack, RuntimeFactoryCatalog
from snow_grass.workflow.repository import WorkflowRepository
from snow_grass.workflow.schemas import (
    ComponentVersionManifest,
    FlowEdge,
    FlowGraph,
    FlowNode,
    FlowValidation,
    PortDefinition,
    ValidationIssue,
)


@dataclass(frozen=True)
class CompiledNode:
    node: FlowNode
    component: ComponentVersionManifest
    incoming_edges: tuple[FlowEdge, ...]
    outgoing_edges: tuple[FlowEdge, ...]


@dataclass(frozen=True)
class CompiledPlan:
    graph: FlowGraph
    nodes: dict[str, CompiledNode]
    execution_order: tuple[str, ...]
    component_manifest_digest: str


class WorkflowCompiler:
    def __init__(self, repository: WorkflowRepository, factories: RuntimeFactoryCatalog) -> None:
        self._repository = repository
        self._factories = factories

    async def compile(self, graph: FlowGraph, pack: BusinessPack) -> CompiledPlan:
        errors: list[ValidationIssue] = []
        ids = [node.id for node in graph.nodes]
        if not ids:
            errors.append(ValidationIssue(code="empty_graph", message="Flow 至少需要一个组件"))
        if len(ids) != len(set(ids)):
            errors.append(ValidationIssue(code="duplicate_node", message="节点 ID 不能重复"))
        rows = await self._repository.get_component_versions(
            {n.component_version_id for n in graph.nodes}
        )
        versions: dict[str, ComponentVersionManifest] = {}
        for version_record, definition in rows:
            manifest = ComponentVersionManifest(
                id=version_record.id,
                component_id=definition.id,
                component_key=definition.key,
                version_number=version_record.version_number,
                status=cast(Any, definition.status),
                implementation_key=version_record.implementation_key,
                implementation_digest=version_record.implementation_digest,
                executor_contract_version=version_record.executor_contract_version,
                config_schema=version_record.config_schema,
                ui_schema=version_record.ui_schema,
                input_ports=[PortDefinition.model_validate(p) for p in version_record.input_ports],
                output_ports=[
                    PortDefinition.model_validate(p) for p in version_record.output_ports
                ],
            )
            versions[version_record.id] = manifest
        node_map = {node.id: node for node in graph.nodes}
        incoming: dict[str, list[FlowEdge]] = {node_id: [] for node_id in node_map}
        outgoing: dict[str, list[FlowEdge]] = {node_id: [] for node_id in node_map}
        for node in graph.nodes:
            component = versions.get(node.component_version_id)
            if component is None:
                errors.append(
                    ValidationIssue(
                        code="component_version_missing", message="组件版本不存在", node_id=node.id
                    )
                )
                continue
            if component.component_key not in pack.allowed_component_keys:
                errors.append(
                    ValidationIssue(
                        code="component_not_allowed",
                        message=f"业务不允许组件 {component.component_key}",
                        node_id=node.id,
                    )
                )
            try:
                factory = self._factories.get(component.implementation_key)
                if factory.implementation_digest != component.implementation_digest:
                    errors.append(
                        ValidationIssue(
                            code="implementation_digest_mismatch",
                            message="组件实现摘要不一致",
                            node_id=node.id,
                        )
                    )
            except LookupError:
                errors.append(
                    ValidationIssue(
                        code="implementation_missing", message="组件实现未注册", node_id=node.id
                    )
                )
            for message in _validate_json_value(node.config, component.config_schema, "config"):
                errors.append(
                    ValidationIssue(code="invalid_config", message=message, node_id=node.id)
                )
        for edge in graph.edges:
            if edge.source_node_id not in node_map or edge.target_node_id not in node_map:
                errors.append(
                    ValidationIssue(
                        code="dangling_edge", message="连线引用不存在的节点", edge_id=edge.id
                    )
                )
                continue
            incoming[edge.target_node_id].append(edge)
            outgoing[edge.source_node_id].append(edge)
            source = versions.get(node_map[edge.source_node_id].component_version_id)
            target = versions.get(node_map[edge.target_node_id].component_version_id)
            if source is None or target is None:
                continue
            source_port = next((p for p in source.output_ports if p.key == edge.source_port), None)
            target_port = next((p for p in target.input_ports if p.key == edge.target_port), None)
            if source_port is None:
                errors.append(
                    ValidationIssue(
                        code="source_port_missing",
                        message="源输出端口不存在",
                        edge_id=edge.id,
                        port=edge.source_port,
                    )
                )
            if target_port is None:
                errors.append(
                    ValidationIssue(
                        code="target_port_missing",
                        message="目标输入端口不存在",
                        edge_id=edge.id,
                        port=edge.target_port,
                    )
                )
            if source_port and target_port and not _ports_compatible(source_port, target_port):
                errors.append(
                    ValidationIssue(
                        code="port_incompatible", message="端口类型不兼容", edge_id=edge.id
                    )
                )
        for node in graph.nodes:
            component = versions.get(node.component_version_id)
            if component is None:
                continue
            connected = {edge.target_port for edge in incoming[node.id]}
            for port in component.input_ports:
                count = sum(edge.target_port == port.key for edge in incoming[node.id])
                if port.required and port.key not in connected:
                    errors.append(
                        ValidationIssue(
                            code="required_port_unconnected",
                            message=f"必填端口未连接：{port.key}",
                            node_id=node.id,
                            port=port.key,
                        )
                    )
                if count > 1 and not port.multiple:
                    errors.append(
                        ValidationIssue(
                            code="port_multiple_not_allowed",
                            message=f"端口不允许多输入：{port.key}",
                            node_id=node.id,
                            port=port.key,
                        )
                    )
            if (
                len(graph.nodes) > 1
                and not incoming[node.id]
                and not outgoing[node.id]
                and component.input_ports
            ):
                errors.append(
                    ValidationIssue(code="isolated_node", message="组件未连接", node_id=node.id)
                )
        indegree = {node_id: len(incoming[node_id]) for node_id in node_map}
        queue = [node_id for node_id in ids if indegree[node_id] == 0]
        order: list[str] = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for edge in outgoing[node_id]:
                indegree[edge.target_node_id] -= 1
                if indegree[edge.target_node_id] == 0:
                    queue.append(edge.target_node_id)
        if len(order) != len(node_map):
            errors.append(ValidationIssue(code="cycle", message="Flow 不能包含循环"))
        errors.extend(pack.validator(graph) if pack.validator else [])
        if errors:
            raise WorkflowCompileError(
                FlowValidation(valid=False, errors=errors, execution_order=order)
            )
        compiled = {
            node_id: CompiledNode(
                node_map[node_id],
                versions[node_map[node_id].component_version_id],
                tuple(incoming[node_id]),
                tuple(outgoing[node_id]),
            )
            for node_id in order
        }
        digest_data = [
            (c.component.id, c.component.implementation_digest) for c in compiled.values()
        ]
        digest = hashlib.sha256(json.dumps(digest_data, separators=(",", ":")).encode()).hexdigest()
        return CompiledPlan(graph, compiled, tuple(order), digest)

    async def validate(self, graph: FlowGraph, pack: BusinessPack) -> FlowValidation:
        try:
            plan = await self.compile(graph, pack)
            return FlowValidation(valid=True, execution_order=list(plan.execution_order))
        except WorkflowCompileError as exc:
            return exc.validation


class WorkflowCompileError(ValueError):
    def __init__(self, validation: FlowValidation) -> None:
        self.validation = validation
        super().__init__(validation.errors[0].message if validation.errors else "Flow 校验失败")


def _ports_compatible(source: PortDefinition, target: PortDefinition) -> bool:
    source_type = source.schema_.get("type")
    target_type = target.schema_.get("type")
    return not source_type or not target_type or source_type == target_type


def _validate_json_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    expected = schema.get("type")
    checks: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    if expected in checks and (
        not isinstance(value, checks[expected]) or expected == "integer" and isinstance(value, bool)
    ):
        return [f"{path} 应为 {expected}"]
    errors: list[str] = []
    if expected == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} 为必填项")
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(f"{path}.{key} 未定义")
        for key, child in properties.items():
            if key in value:
                errors.extend(_validate_json_value(value[key], child, f"{path}.{key}"))
    return errors
