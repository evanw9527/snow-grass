from __future__ import annotations

import pytest

from snow_grass.workflow.compiler import CompiledNode, CompiledPlan
from snow_grass.workflow.executor import WorkflowExecutor, _edge_matches
from snow_grass.workflow.registry import (
    BusinessPack,
    ComponentRuntimeFactory,
    RuntimeFactoryCatalog,
)
from snow_grass.workflow.runtime import (
    ComponentRuntimeRegistry,
    NodeExecutionContext,
    NodeExecutionResult,
)
from snow_grass.workflow.schemas import (
    ComponentVersionManifest,
    EdgeCondition,
    FlowEdge,
    FlowGraph,
    FlowNode,
    PortDefinition,
)


class _ClosedBranch:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        return NodeExecutionResult({"yes": {"value": 1}}, active_ports=frozenset())


class _MustNotRun:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        raise AssertionError("unreachable component executed")


class _OpenBranch:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        return NodeExecutionResult({"yes": {"score": 8, "approved": True}})


@pytest.mark.asyncio
async def test_inactive_output_marks_downstream_unreachable() -> None:
    catalog = RuntimeFactoryCatalog()
    catalog.register(ComponentRuntimeFactory("branch@1", "branch", _ClosedBranch))
    catalog.register(ComponentRuntimeFactory("target@1", "target", _MustNotRun))
    runtimes = ComponentRuntimeRegistry(catalog)
    source_manifest = ComponentVersionManifest(
        id="source-v1",
        component_id="source",
        component_key="branch",
        version_number=1,
        implementation_key="branch@1",
        implementation_digest="branch",
        output_ports=[PortDefinition(key="yes", name="是", schema={"type": "object"})],
    )
    target_manifest = ComponentVersionManifest(
        id="target-v1",
        component_id="target",
        component_key="target",
        version_number=1,
        implementation_key="target@1",
        implementation_digest="target",
        input_ports=[PortDefinition(key="input", name="输入", schema={"type": "object"})],
    )
    edge = FlowEdge(
        id="edge",
        source_node_id="source",
        source_port="yes",
        target_node_id="target",
        target_port="input",
    )
    graph = FlowGraph(
        nodes=[
            FlowNode(id="source", component_version_id="source-v1"),
            FlowNode(id="target", component_version_id="target-v1"),
        ],
        edges=[edge],
    )
    plan = CompiledPlan(
        graph=graph,
        nodes={
            "source": CompiledNode(graph.nodes[0], source_manifest, (), (edge,)),
            "target": CompiledNode(graph.nodes[1], target_manifest, (edge,), ()),
        },
        execution_order=("source", "target"),
        component_manifest_digest="digest",
    )
    pack = BusinessPack("test", "测试", "", frozenset({"branch", "target"}), (), graph)
    result = await WorkflowExecutor(runtimes).execute(
        workflow_id="flow", version_id="v1", pack=pack, plan=plan, workflow_input={}, preview=False
    )
    assert result.status == "succeeded"
    assert [trace.status for trace in result.trace] == ["succeeded", "skipped"]
    assert result.trace[1].skip_reason == "unreachable"


def test_edge_condition_supports_nested_paths_and_safe_operators() -> None:
    payload = {"result": {"score": 8}, "labels": ["safe", "reviewed"]}
    assert _edge_matches(
        payload, EdgeCondition(path="result.score", operator="greater_than", value=5)
    )
    assert _edge_matches(payload, EdgeCondition(path="labels", operator="contains", value="safe"))
    assert _edge_matches(payload, EdgeCondition(path="result.score", operator="exists"))
    assert not _edge_matches(payload, EdgeCondition(path="result.missing", operator="truthy"))


@pytest.mark.asyncio
async def test_false_edge_condition_marks_branch_unreachable() -> None:
    catalog = RuntimeFactoryCatalog()
    catalog.register(ComponentRuntimeFactory("branch@1", "branch", _OpenBranch))
    catalog.register(ComponentRuntimeFactory("target@1", "target", _MustNotRun))
    runtimes = ComponentRuntimeRegistry(catalog)
    source_manifest = ComponentVersionManifest(
        id="source-v1",
        component_id="source",
        component_key="branch",
        version_number=1,
        implementation_key="branch@1",
        implementation_digest="branch",
        output_ports=[PortDefinition(key="yes", name="是", schema={"type": "object"})],
    )
    target_manifest = ComponentVersionManifest(
        id="target-v1",
        component_id="target",
        component_key="target",
        version_number=1,
        implementation_key="target@1",
        implementation_digest="target",
        input_ports=[PortDefinition(key="input", name="输入", schema={"type": "object"})],
    )
    edge = FlowEdge(
        id="edge",
        source_node_id="source",
        source_port="yes",
        target_node_id="target",
        target_port="input",
        condition=EdgeCondition(path="approved", operator="equals", value=False),
    )
    graph = FlowGraph(
        nodes=[
            FlowNode(id="source", component_version_id="source-v1"),
            FlowNode(id="target", component_version_id="target-v1"),
        ],
        edges=[edge],
    )
    plan = CompiledPlan(
        graph=graph,
        nodes={
            "source": CompiledNode(graph.nodes[0], source_manifest, (), (edge,)),
            "target": CompiledNode(graph.nodes[1], target_manifest, (edge,), ()),
        },
        execution_order=("source", "target"),
        component_manifest_digest="digest",
    )
    pack = BusinessPack("test", "测试", "", frozenset({"branch", "target"}), (), graph)
    result = await WorkflowExecutor(runtimes).execute(
        workflow_id="flow", version_id="v1", pack=pack, plan=plan, workflow_input={}, preview=False
    )
    assert [trace.status for trace in result.trace] == ["succeeded", "skipped"]
    assert result.trace[1].skip_reason == "unreachable"
