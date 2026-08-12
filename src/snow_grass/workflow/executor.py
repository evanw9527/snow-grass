from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from snow_grass.workflow.compiler import CompiledPlan
from snow_grass.workflow.registry import BusinessPack
from snow_grass.workflow.runtime import ComponentRuntimeRegistry, NodeExecutionContext
from snow_grass.workflow.schemas import EdgeCondition, FlowRunResponse, NodeTrace

_MISSING = object()


class WorkflowExecutor:
    def __init__(self, runtimes: ComponentRuntimeRegistry) -> None:
        self._runtimes = runtimes

    async def execute(
        self,
        *,
        workflow_id: str,
        version_id: str | None,
        pack: BusinessPack,
        plan: CompiledPlan,
        workflow_input: dict[str, Any],
        preview: bool,
    ) -> FlowRunResponse:
        run_id = str(uuid4())
        started_at = datetime.now(UTC)
        delivered: dict[tuple[str, str], list[Any]] = {}
        traces: list[NodeTrace] = []
        workflow_output: dict[str, Any] = {}
        failed = False
        for node_id in plan.execution_order:
            compiled = plan.nodes[node_id]
            node = compiled.node
            component = compiled.component
            inputs: dict[str, Any] = {}
            for port in component.input_ports:
                values = delivered.get((node_id, port.key), [])
                if values:
                    inputs[port.key] = values if port.multiple else values[-1]
            missing = [p.key for p in component.input_ports if p.required and p.key not in inputs]
            if failed or node.disabled or missing:
                reason = (
                    "upstream_failed" if failed else "disabled" if node.disabled else "unreachable"
                )
                traces.append(
                    NodeTrace(
                        node_id=node.id,
                        component_version_id=component.id,
                        node_type=component.component_key,
                        status="skipped",
                        duration_ms=0,
                        input_port_summary=_summarize_ports(inputs),
                        skip_reason=reason,
                    )
                )
                continue
            tick = perf_counter()
            try:
                runtime = await self._runtimes.resolve(component)
                result = await runtime.execute(
                    NodeExecutionContext(
                        run_id=run_id,
                        flow_version_id=version_id,
                        node_id=node.id,
                        component_version_id=component.id,
                        config=dict(node.config),
                        inputs_by_port=inputs,
                        workflow_input=workflow_input,
                    )
                )
                declared = {p.key for p in component.output_ports}
                unknown = result.outputs_by_port.keys() - declared
                if unknown:
                    raise ValueError(f"组件返回未声明端口：{', '.join(sorted(unknown))}")
                for edge in compiled.outgoing_edges:
                    output_value = result.outputs_by_port.get(edge.source_port, _MISSING)
                    if (
                        result.is_active(edge.source_port)
                        and output_value is not _MISSING
                        and _edge_matches(output_value, edge.condition)
                    ):
                        delivered.setdefault((edge.target_node_id, edge.target_port), []).append(
                            output_value
                        )
                if "workflow_output" in result.outputs_by_port:
                    value = result.outputs_by_port["workflow_output"]
                    workflow_output = dict(value) if isinstance(value, dict) else {"result": value}
                traces.append(
                    NodeTrace(
                        node_id=node.id,
                        component_version_id=component.id,
                        node_type=component.component_key,
                        status="succeeded",
                        duration_ms=max(0, int((perf_counter() - tick) * 1000)),
                        input_summary=_summarize(inputs),
                        output_summary=_summarize(result.outputs_by_port),
                        input_port_summary=_summarize_ports(inputs),
                        output_port_summary=_summarize_ports(result.outputs_by_port),
                    )
                )
            except Exception as exc:
                failed = True
                traces.append(
                    NodeTrace(
                        node_id=node.id,
                        component_version_id=component.id,
                        node_type=component.component_key,
                        status="failed",
                        duration_ms=max(0, int((perf_counter() - tick) * 1000)),
                        input_port_summary=_summarize_ports(inputs),
                        error=str(exc),
                    )
                )
        if pack.output_adapter and not failed:
            workflow_output = await pack.output_adapter(workflow_output)
        return FlowRunResponse(
            run_id=run_id,
            workflow_id=workflow_id,
            flow_version_id=version_id,
            resolved_version_id=version_id,
            business_type=pack.business_type,
            status="failed" if failed else "succeeded",
            preview=preview,
            output=workflow_output,
            trace=traces,
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )


def _summarize_ports(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _summary_value(value) for key, value in values.items()}


def _summarize(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {key: _summary_value(item) for key, item in value.items()}
    return {"value": _summary_value(value)}


def _summary_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(value.keys())[:20]}
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return type(value).__name__


def _edge_matches(value: Any, condition: EdgeCondition | None) -> bool:
    if condition is None:
        return True
    actual = _resolve_path(value, condition.path)
    operator = condition.operator
    if operator == "exists":
        return actual is not _MISSING
    if operator == "not_exists":
        return actual is _MISSING
    if actual is _MISSING:
        return False
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    if operator == "equals":
        return bool(actual == condition.value)
    if operator == "not_equals":
        return bool(actual != condition.value)
    if operator == "contains":
        try:
            return bool(condition.value in actual)
        except (TypeError, AttributeError):
            return False
    try:
        if operator == "greater_than":
            return bool(actual > condition.value)
        if operator == "greater_than_or_equal":
            return bool(actual >= condition.value)
        if operator == "less_than":
            return bool(actual < condition.value)
        if operator == "less_than_or_equal":
            return bool(actual <= condition.value)
    except TypeError:
        return False
    return False


def _resolve_path(value: Any, path: str) -> Any:
    normalized = path.strip()
    if normalized in {"", "$"}:
        return value
    if normalized.startswith("$."):
        normalized = normalized[2:]
    current = value
    for part in normalized.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return _MISSING
    return current
