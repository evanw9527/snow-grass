from __future__ import annotations

from pathlib import Path
from typing import Any

from snow_grass.workflow.registry import (
    BusinessPack,
    ComponentRuntimeBuilder,
    ComponentRuntimeFactory,
    RuntimeFactoryCatalog,
    SystemComponentSpec,
    system_component_version_id,
)
from snow_grass.workflow.runtime import NodeExecutionContext, NodeExecutionResult
from snow_grass.workflow.schemas import FlowEdge, FlowGraph, FlowNode


def _port(key: str, name: str) -> dict[str, Any]:
    return {"key": key, "name": name, "schema": {"type": "object"}, "required": True}


class _ClassifyRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["payload"])
        text = str(payload.get("text", "")).casefold()
        markers = ("烦", "难受", "焦虑", "生气", "累", "sad", "angry", "tired")
        result = {**payload, "negative": any(marker in text for marker in markers)}
        return NodeExecutionResult({"classification": result})


class _ReplyRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        payload = dict(context.inputs_by_port["classification"])
        result = {
            "should_respond": bool(payload.get("negative")),
            "reply": "我听到了。先停一下，我们把眼前最小的一步处理好。"
            if payload.get("negative")
            else None,
        }
        return NodeExecutionResult({"response": result})


def register_negative_feedback_pack(catalog: RuntimeFactoryCatalog) -> BusinessPack:
    source_code = Path(__file__).read_text(encoding="utf-8")
    specs = (
        SystemComponentSpec(
            "feedback.classify",
            "消极分类",
            "识别需要反馈的消极输入",
            "理解",
            "feedback.classify@1",
            "sha256:feedback-classify-v1",
            {},
            {},
            (_port("payload", "输入"),),
            (_port("classification", "分类结果"),),
            icon="classifier",
        ),
        SystemComponentSpec(
            "feedback.reply",
            "生成反馈",
            "生成受控的简短回应",
            "输出",
            "feedback.reply@1",
            "sha256:feedback-reply-v1",
            {},
            {},
            (_port("classification", "分类结果"),),
            (_port("response", "回应"),),
            icon="chats",
        ),
    )
    builders: tuple[ComponentRuntimeBuilder, ...] = (_ClassifyRuntime, _ReplyRuntime)
    for spec, builder in zip(specs, builders, strict=True):
        catalog.register(
            ComponentRuntimeFactory(
                spec.implementation_key,
                spec.implementation_digest,
                builder,
                source_code=source_code,
                source_path="snow_grass/feedback/flow_pack.py",
            )
        )
    node_ids = ["input", "classify", "reply", "output"]
    keys = ["core.input", "feedback.classify", "feedback.reply", "core.output"]
    graph = FlowGraph(
        nodes=[
            FlowNode(
                id=node_id,
                component_version_id=system_component_version_id(key),
                name=key,
                x=60 + i * 280,
                y=240,
            )
            for i, (node_id, key) in enumerate(zip(node_ids, keys, strict=True))
        ],
        edges=[
            FlowEdge(
                id="edge-input-classify",
                source_node_id="input",
                source_port="payload",
                target_node_id="classify",
                target_port="payload",
            ),
            FlowEdge(
                id="edge-classify-reply",
                source_node_id="classify",
                source_port="classification",
                target_node_id="reply",
                target_port="classification",
            ),
            FlowEdge(
                id="edge-reply-output",
                source_node_id="reply",
                source_port="response",
                target_node_id="output",
                target_port="result",
            ),
        ],
    )
    return BusinessPack(
        "negative_feedback",
        "消极输入反馈",
        "对消极输入生成受控反馈",
        frozenset(keys),
        specs,
        graph,
    )
