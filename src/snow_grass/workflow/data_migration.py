from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from snow_grass.workflow.models import WorkflowRecord, WorkflowVersionRecord
from snow_grass.workflow.registry import system_component_version_id
from snow_grass.workflow.schemas import FlowGraph

PORTS: dict[str, tuple[str | None, str | None]] = {
    "core.input": (None, "payload"),
    "core.output": ("result", "workflow_output"),
    "activity.normalize": ("payload", "events"),
    "activity.filter": ("events", "events"),
    "activity.summarize": ("events", "summary_draft"),
    "activity.validate": ("summary_draft", "summary"),
    "feedback.classify": ("payload", "classification"),
    "feedback.reply": ("classification", "response"),
}


@dataclass(frozen=True)
class MigrationReport:
    workflows: int = 0
    versions: int = 0
    already_migrated: bool = False


class WorkflowV2DataMigration:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def migrate(self, *, dry_run: bool = False) -> MigrationReport:
        async with self._session_factory() as session:
            workflows = list(await session.scalars(select(WorkflowRecord)))
            versions = list(await session.scalars(select(WorkflowVersionRecord)))
            changed_workflows = 0
            changed_versions = 0
            for workflow in workflows:
                converted, changed = convert_graph(workflow.draft_graph)
                if changed:
                    FlowGraph.model_validate(converted)
                    workflow.draft_graph = converted
                    changed_workflows += 1
            for version in versions:
                converted, changed = convert_graph(version.graph)
                if changed:
                    FlowGraph.model_validate(converted)
                    version.graph = converted
                    changed_versions += 1
            if dry_run:
                await session.rollback()
            else:
                await session.commit()
            return MigrationReport(
                changed_workflows,
                changed_versions,
                already_migrated=not changed_workflows and not changed_versions,
            )

    async def verify(self) -> MigrationReport:
        async with self._session_factory() as session:
            workflows = list(await session.scalars(select(WorkflowRecord)))
            versions = list(await session.scalars(select(WorkflowVersionRecord)))
            for workflow in workflows:
                FlowGraph.model_validate(workflow.draft_graph)
            for version in versions:
                FlowGraph.model_validate(version.graph)
            return MigrationReport(len(workflows), len(versions), already_migrated=True)


def convert_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    nodes = graph.get("nodes", [])
    if not nodes or all("component_version_id" in node for node in nodes):
        return graph, False
    node_types: dict[str, str] = {}
    converted_nodes = []
    for node in nodes:
        node_type = node.get("type") or node.get("kind")
        if node_type not in PORTS:
            raise ValueError(f"Unknown legacy node {node.get('id')}: {node_type}")
        node_types[str(node["id"])] = node_type
        converted_nodes.append(
            {
                "id": node["id"],
                "component_version_id": system_component_version_id(node_type),
                "name": node.get("title") or node.get("name") or node_type,
                "config": node.get("config") or {},
                "x": node.get("x", 0),
                "y": node.get("y", 0),
                "disabled": bool(node.get("disabled", False)),
            }
        )
    converted_edges = []
    for edge in graph.get("edges", []):
        if edge.get("condition"):
            raise ValueError(f"Legacy conditional edge is not migratable: {edge.get('id')}")
        source = str(edge.get("source") or edge.get("from"))
        target = str(edge.get("target") or edge.get("to"))
        if source not in node_types or target not in node_types:
            raise ValueError(f"Dangling legacy edge: {edge.get('id')}")
        source_port = PORTS[node_types[source]][1]
        target_port = PORTS[node_types[target]][0]
        if source_port is None or target_port is None:
            raise ValueError(f"Invalid legacy edge direction: {edge.get('id')}")
        converted_edges.append(
            {
                "id": edge["id"],
                "source_node_id": source,
                "source_port": source_port,
                "target_node_id": target,
                "target_port": target_port,
            }
        )
    return {"nodes": converted_nodes, "edges": converted_edges}, True
