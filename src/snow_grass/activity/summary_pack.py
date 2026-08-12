from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from snow_grass.activity.intelligence import ActivityIntelligence
from snow_grass.activity.repository import ActivityRepository
from snow_grass.workflow.registry import (
    BusinessPack,
    ComponentRuntimeBuilder,
    ComponentRuntimeFactory,
    RuntimeFactoryCatalog,
    SystemComponentSpec,
    system_component_version_id,
)
from snow_grass.workflow.runtime import NodeExecutionContext, NodeExecutionResult
from snow_grass.workflow.schemas import FlowEdge, FlowGraph, FlowNode, ValidationIssue


def _port(key: str, name: str, json_type: str = "object") -> dict[str, Any]:
    return {"key": key, "name": name, "schema": {"type": json_type}, "required": True}


class _InputRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        return NodeExecutionResult({"payload": dict(context.workflow_input)})


class _NormalizeRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["payload"])
        return NodeExecutionResult({"events": {**payload, "events": payload.get("events", [])}})


class _FilterRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["events"])
        min_length = int(context.config.get("min_length", 2))
        kept = []
        filtered = 0
        for event in payload.get("events", []):
            text = " ".join(str(getattr(event, "text", "")).split())
            if len(text) < min_length or re.fullmatch(r"https?://\S+|\W+", text):
                filtered += 1
            else:
                kept.append(event)
        return NodeExecutionResult(
            {"events": {**payload, "events": kept, "filtered_count": filtered}}
        )


class _SummarizeRuntime:
    def __init__(self, intelligence: ActivityIntelligence) -> None:
        self._intelligence = intelligence

    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["events"])
        events = payload.get("events", [])
        if not events:
            result = {
                "completed": [],
                "in_progress": [],
                "blockers": [],
                "next_steps": [],
                "event_count": 0,
                "filtered_count": payload.get("filtered_count", 0),
            }
        else:
            content = await self._intelligence.summarize(events)
            result = {
                **content.model_dump(),
                "valid_event_ids": [event.client_event_id for event in events],
                "event_count": len(events),
                "filtered_count": payload.get("filtered_count", 0),
            }
        return NodeExecutionResult({"summary_draft": result})


class _ValidateRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["summary_draft"])
        valid_ids = set(payload.get("valid_event_ids", []))
        validated: dict[str, list[str]] = {}
        for key in ("completed", "in_progress", "blockers", "next_steps"):
            items = []
            for item in payload.get(key, []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                evidence = set(item.get("evidence_event_ids", []))
                if (
                    text
                    and evidence
                    and evidence <= valid_ids
                    and re.search(r"[\u4e00-\u9fff]", text)
                ):
                    items.append(text)
            validated[key] = items[:5]
        result = validated | {
            "title": str(payload.get("title", "")).strip()[:30],
            "keywords": [
                str(v).strip()[:80] for v in payload.get("keywords", []) if str(v).strip()
            ][:8],
            "event_count": int(payload.get("event_count", 0)),
            "filtered_count": int(payload.get("filtered_count", 0)),
        }
        return NodeExecutionResult({"summary": result})


class _OutputRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        return NodeExecutionResult({"workflow_output": context.inputs_by_port["result"]})


def register_activity_summary_pack(
    catalog: RuntimeFactoryCatalog,
    intelligence: ActivityIntelligence,
    repository: ActivityRepository | None = None,
) -> BusinessPack:
    source_code = Path(__file__).read_text(encoding="utf-8")
    specs = (
        SystemComponentSpec(
            "core.input",
            "业务输入",
            "接收业务输入",
            "输入",
            "core.input@1",
            "sha256:core-input-v1",
            {},
            {},
            (),
            (_port("payload", "业务载荷"),),
            icon="trigger",
        ),
        SystemComponentSpec(
            "core.output",
            "业务输出",
            "输出业务结果",
            "输出",
            "core.output@1",
            "sha256:core-output-v1",
            {},
            {},
            (_port("result", "结果"),),
            (_port("workflow_output", "流程输出"),),
            icon="output",
        ),
        SystemComponentSpec(
            "activity.normalize",
            "规范化",
            "整理输入并保留事件引用",
            "处理",
            "activity.normalize@1",
            "sha256:activity-normalize-v1",
            {},
            {},
            (_port("payload", "业务载荷"),),
            (_port("events", "事件"),),
            icon="action",
        ),
        SystemComponentSpec(
            "activity.filter",
            "噪声过滤",
            "过滤 URL、空白与低信息内容",
            "处理",
            "activity.filter@1",
            "sha256:activity-filter-v1",
            {
                "type": "object",
                "properties": {"min_length": {"type": "integer", "minimum": 0, "default": 2}},
                "additionalProperties": False,
            },
            {},
            (_port("events", "事件"),),
            (_port("events", "有效事件"),),
            icon="condition",
        ),
        SystemComponentSpec(
            "activity.summarize",
            "中文总结",
            "调用模型生成四类小时总结",
            "AI",
            "activity.summarize@1",
            "sha256:activity-summarize-v1",
            {},
            {},
            (_port("events", "有效事件"),),
            (_port("summary_draft", "总结草稿"),),
            icon="llm",
        ),
        SystemComponentSpec(
            "activity.validate",
            "结果校验",
            "限制结构、数量和输出字段",
            "校验",
            "activity.validate@1",
            "sha256:activity-validate-v1",
            {},
            {},
            (_port("summary_draft", "总结草稿"),),
            (_port("summary", "总结"),),
            icon="success",
        ),
    )
    builders: tuple[ComponentRuntimeBuilder, ...] = (
        _InputRuntime,
        _OutputRuntime,
        _NormalizeRuntime,
        _FilterRuntime,
        lambda: _SummarizeRuntime(intelligence),
        _ValidateRuntime,
    )
    for spec, builder in zip(specs, builders, strict=True):
        catalog.register(
            ComponentRuntimeFactory(
                spec.implementation_key,
                spec.implementation_digest,
                builder,
                source_code=source_code,
                source_path="snow_grass/activity/summary_pack.py",
            )
        )

    async def adapt_input(payload: dict[str, Any]) -> dict[str, Any]:
        if "events" in payload or repository is None:
            return payload
        period = payload.get("period_start")
        if not isinstance(period, str):
            return {**payload, "events": []}
        parsed = datetime.fromisoformat(period.replace("Z", "+00:00"))
        start = parsed.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        events = await repository.list_window(start=start, end=start + timedelta(hours=1))
        return {**payload, "period_start": start.isoformat(), "events": events}

    keys = [spec.key for spec in specs]
    node_ids = ["input", "normalize", "filter", "summarize", "validate", "output"]
    node_keys = [
        "core.input",
        "activity.normalize",
        "activity.filter",
        "activity.summarize",
        "activity.validate",
        "core.output",
    ]
    ports = [
        ("payload", "payload"),
        ("events", "events"),
        ("events", "events"),
        ("summary_draft", "summary_draft"),
        ("summary", "result"),
    ]
    graph = FlowGraph(
        nodes=[
            FlowNode(
                id=node_id,
                component_version_id=system_component_version_id(key),
                name=next(s.name for s in specs if s.key == key),
                x=40 + i * 260,
                y=250,
                config={"min_length": 2} if key == "activity.filter" else {},
            )
            for i, (node_id, key) in enumerate(zip(node_ids, node_keys, strict=True))
        ],
        edges=[
            FlowEdge(
                id=f"edge-{a}-{b}",
                source_node_id=a,
                source_port=sp,
                target_node_id=b,
                target_port=tp,
            )
            for (a, b), (sp, tp) in zip(
                zip(node_ids, node_ids[1:], strict=False), ports, strict=True
            )
        ],
    )
    return BusinessPack(
        "activity_summary",
        "活动小时总结",
        "将一小时活动整理为可追踪的中文总结",
        frozenset(keys),
        specs,
        graph,
        _validate_summary_graph,
        adapt_input,
    )


def _validate_summary_graph(graph: FlowGraph) -> list[ValidationIssue]:
    required = {
        system_component_version_id(key)
        for key in ("core.input", "activity.summarize", "activity.validate", "core.output")
    }
    missing = required - {node.component_version_id for node in graph.nodes}
    return [
        ValidationIssue(code="required_component", message=f"活动总结缺少组件版本：{item}")
        for item in sorted(missing)
    ]
