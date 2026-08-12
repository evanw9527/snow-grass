from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from snow_grass.workflow.models import (
    ComponentDefinitionRecord,
    ComponentVersionRecord,
    WorkflowDeploymentEventRecord,
    WorkflowDeploymentRecord,
    WorkflowNodeRunRecord,
    WorkflowRecord,
    WorkflowRunRecord,
    WorkflowVersionRecord,
)
from snow_grass.workflow.registry import SystemComponentSpec
from snow_grass.workflow.schemas import ComponentManifest, FlowGraph, NodeTrace


class WorkflowRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def seed_system_components(
        self,
        specs: list[SystemComponentSpec],
        source_snapshots: dict[str, tuple[str, str, str | None]],
    ) -> None:
        async with self._session_factory() as session:
            for spec in specs:
                definition = await session.get(ComponentDefinitionRecord, spec.component_id)
                manifest = {
                    "icon": spec.icon,
                    "implementation_key": spec.implementation_key,
                    "executor_contract_version": 1,
                    "config_schema": spec.config_schema,
                    "ui_schema": spec.ui_schema,
                    "input_ports": list(spec.input_ports),
                    "output_ports": list(spec.output_ports),
                }
                if definition is None:
                    definition = ComponentDefinitionRecord(
                        id=spec.component_id,
                        key=spec.key,
                        name=spec.name,
                        description=spec.description,
                        category=spec.category,
                        icon=spec.icon,
                        status="active",
                        draft_revision=0,
                        draft_manifest=manifest,
                    )
                    session.add(definition)
                elif "icon" not in definition.draft_manifest:
                    definition.draft_manifest = {**definition.draft_manifest, "icon": spec.icon}
                version = await session.get(ComponentVersionRecord, spec.version_id)
                source_code, source_language, source_path = source_snapshots.get(
                    spec.implementation_key, ("", "python", None)
                )
                if version is None:
                    session.add(
                        ComponentVersionRecord(
                            id=spec.version_id,
                            component_id=spec.component_id,
                            version_number=1,
                            icon=spec.icon,
                            implementation_key=spec.implementation_key,
                            implementation_digest=spec.implementation_digest,
                            implementation_source=source_code,
                            source_language=source_language,
                            source_path=source_path,
                            executor_contract_version=1,
                            config_schema=spec.config_schema,
                            ui_schema=spec.ui_schema,
                            input_ports=list(spec.input_ports),
                            output_ports=list(spec.output_ports),
                            release_note="系统组件初始版本",
                        )
                    )
                elif (version.implementation_key, version.implementation_digest) != (
                    spec.implementation_key,
                    spec.implementation_digest,
                ):
                    raise RuntimeError(f"System ComponentVersion immutable mismatch: {spec.key}")
                elif not version.implementation_source and source_code:
                    version.implementation_source = source_code
                    version.source_language = source_language
                    version.source_path = source_path
                if definition.icon is None:
                    definition.icon = spec.icon
                if version is not None and version.icon == "tool" and spec.icon != "tool":
                    version.icon = spec.icon
            await session.commit()

    async def list_components(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        status: str | None = None,
    ) -> list[ComponentDefinitionRecord]:
        async with self._session_factory() as session:
            statement = select(ComponentDefinitionRecord).order_by(
                ComponentDefinitionRecord.updated_at.desc()
            )
            if query:
                statement = statement.where(ComponentDefinitionRecord.name.contains(query))
            if category:
                statement = statement.where(ComponentDefinitionRecord.category == category)
            if status:
                statement = statement.where(ComponentDefinitionRecord.status == status)
            return list(await session.scalars(statement))

    async def get_component(self, component_id: str) -> ComponentDefinitionRecord | None:
        async with self._session_factory() as session:
            return cast(
                ComponentDefinitionRecord | None,
                await session.get(ComponentDefinitionRecord, component_id),
            )

    async def create_component(
        self,
        *,
        key: str,
        name: str,
        description: str,
        category: str,
        icon: str | None,
        manifest: ComponentManifest,
    ) -> ComponentDefinitionRecord:
        async with self._session_factory() as session:
            if await session.scalar(
                select(ComponentDefinitionRecord.id).where(ComponentDefinitionRecord.key == key)
            ):
                raise RuntimeError("Component key already exists")
            record = ComponentDefinitionRecord(
                id=str(uuid4()),
                key=key,
                name=name,
                description=description,
                category=category,
                icon=icon,
                status="active",
                draft_revision=0,
                draft_manifest=manifest.model_dump(by_alias=True),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def save_component_draft(
        self, component_id: str, *, expected_revision: int, manifest: ComponentManifest
    ) -> ComponentDefinitionRecord:
        async with self._session_factory() as session:
            record = await session.get(ComponentDefinitionRecord, component_id)
            if record is None:
                raise LookupError("Component not found")
            if record.draft_revision != expected_revision:
                raise RuntimeError("Component draft revision conflict")
            record.draft_manifest = manifest.model_dump(by_alias=True)
            record.icon = manifest.icon
            record.draft_revision += 1
            record.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(record)
            return record

    async def list_component_versions(self, component_id: str) -> list[ComponentVersionRecord]:
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(ComponentVersionRecord)
                    .where(ComponentVersionRecord.component_id == component_id)
                    .order_by(ComponentVersionRecord.version_number.desc())
                )
            )

    async def get_component_version(self, version_id: str) -> ComponentVersionRecord | None:
        async with self._session_factory() as session:
            return cast(
                ComponentVersionRecord | None, await session.get(ComponentVersionRecord, version_id)
            )

    async def get_component_versions(
        self, version_ids: set[str]
    ) -> list[tuple[ComponentVersionRecord, ComponentDefinitionRecord]]:
        if not version_ids:
            return []
        async with self._session_factory() as session:
            rows = await session.execute(
                select(ComponentVersionRecord, ComponentDefinitionRecord)
                .join(
                    ComponentDefinitionRecord,
                    ComponentDefinitionRecord.id == ComponentVersionRecord.component_id,
                )
                .where(ComponentVersionRecord.id.in_(version_ids))
            )
            return list(rows.tuples())

    async def create_component_version(
        self,
        component_id: str,
        *,
        expected_revision: int,
        digest: str,
        implementation_source: str,
        source_language: str,
        source_path: str | None,
        release_note: str | None,
    ) -> ComponentVersionRecord:
        async with self._session_factory() as session:
            component = await session.get(ComponentDefinitionRecord, component_id)
            if component is None:
                raise LookupError("Component not found")
            if component.draft_revision != expected_revision:
                raise RuntimeError("Component draft revision conflict")
            manifest = ComponentManifest.model_validate(component.draft_manifest)
            current = await session.scalar(
                select(func.max(ComponentVersionRecord.version_number)).where(
                    ComponentVersionRecord.component_id == component_id
                )
            )
            version = ComponentVersionRecord(
                id=str(uuid4()),
                component_id=component_id,
                version_number=(current or 0) + 1,
                icon=manifest.icon,
                implementation_key=manifest.implementation_key,
                implementation_digest=digest,
                implementation_source=implementation_source,
                source_language=source_language,
                source_path=source_path,
                executor_contract_version=manifest.executor_contract_version,
                config_schema=manifest.config_schema,
                ui_schema=manifest.ui_schema,
                input_ports=[p.model_dump(by_alias=True) for p in manifest.input_ports],
                output_ports=[p.model_dump(by_alias=True) for p in manifest.output_ports],
                release_note=release_note,
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
            return version

    async def update_component_status(
        self, component_id: str, status: str
    ) -> ComponentDefinitionRecord:
        async with self._session_factory() as session:
            record = await session.get(ComponentDefinitionRecord, component_id)
            if record is None:
                raise LookupError("Component not found")
            record.status = status
            record.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(record)
            return record

    async def component_reference_count(self, component_id: str) -> int:
        version_ids = {v.id for v in await self.list_component_versions(component_id)}
        if not version_ids:
            return 0
        async with self._session_factory() as session:
            graphs = list(await session.scalars(select(WorkflowVersionRecord.graph)))
            count = 0
            for graph in graphs:
                nodes = cast(list[dict[str, Any]], graph.get("nodes", []))
                if any(node.get("component_version_id") in version_ids for node in nodes):
                    count += 1
            return count

    async def list_workflows(self, business_type: str | None = None) -> list[WorkflowRecord]:
        async with self._session_factory() as session:
            query = select(WorkflowRecord).order_by(WorkflowRecord.updated_at.desc())
            if business_type:
                query = query.where(WorkflowRecord.business_type == business_type)
            return list(await session.scalars(query))

    async def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        async with self._session_factory() as session:
            return cast(WorkflowRecord | None, await session.get(WorkflowRecord, workflow_id))

    async def get_by_business(self, business_type: str) -> WorkflowRecord | None:
        async with self._session_factory() as session:
            return cast(
                WorkflowRecord | None,
                await session.scalar(
                    select(WorkflowRecord)
                    .where(WorkflowRecord.business_type == business_type)
                    .order_by(WorkflowRecord.created_at)
                ),
            )

    async def create_workflow(
        self, *, business_type: str, name: str, graph: FlowGraph, description: str = ""
    ) -> WorkflowRecord:
        async with self._session_factory() as session:
            record = WorkflowRecord(
                id=str(uuid4()),
                business_type=business_type,
                name=name,
                description=description,
                status="active",
                draft_revision=0,
                draft_graph=graph.model_dump(by_alias=True),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def save_draft(
        self, workflow_id: str, *, expected_revision: int, graph: FlowGraph
    ) -> WorkflowRecord:
        async with self._session_factory() as session:
            record = await session.get(WorkflowRecord, workflow_id)
            if record is None:
                raise LookupError("Workflow not found")
            if record.draft_revision != expected_revision:
                raise RuntimeError("Workflow draft revision conflict")
            record.draft_graph = graph.model_dump(by_alias=True)
            record.draft_revision += 1
            record.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(record)
            return record

    async def latest_version(self, workflow_id: str) -> WorkflowVersionRecord | None:
        async with self._session_factory() as session:
            return cast(
                WorkflowVersionRecord | None,
                await session.scalar(
                    select(WorkflowVersionRecord)
                    .where(WorkflowVersionRecord.workflow_id == workflow_id)
                    .order_by(WorkflowVersionRecord.version_number.desc())
                ),
            )

    async def list_versions(self, workflow_id: str) -> list[WorkflowVersionRecord]:
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(WorkflowVersionRecord)
                    .where(WorkflowVersionRecord.workflow_id == workflow_id)
                    .order_by(WorkflowVersionRecord.version_number.desc())
                )
            )

    async def update_workflow_status(self, workflow_id: str, status: str) -> WorkflowRecord:
        async with self._session_factory() as session:
            record = await session.get(WorkflowRecord, workflow_id)
            if record is None:
                raise LookupError("Workflow not found")
            record.status = status
            record.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_version(self, version_id: str) -> WorkflowVersionRecord | None:
        async with self._session_factory() as session:
            return cast(
                WorkflowVersionRecord | None, await session.get(WorkflowVersionRecord, version_id)
            )

    async def create_version(
        self,
        workflow: WorkflowRecord,
        *,
        graph: FlowGraph,
        note: str | None,
        component_manifest_digest: str = "",
    ) -> WorkflowVersionRecord:
        async with self._session_factory() as session:
            current = await session.scalar(
                select(func.max(WorkflowVersionRecord.version_number)).where(
                    WorkflowVersionRecord.workflow_id == workflow.id
                )
            )
            version = WorkflowVersionRecord(
                id=str(uuid4()),
                workflow_id=workflow.id,
                version_number=(current or 0) + 1,
                graph=graph.model_dump(by_alias=True),
                note=note,
                component_manifest_digest=component_manifest_digest,
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
            return version

    async def deploy(
        self,
        workflow_id: str,
        version_id: str,
        *,
        expected_current_version_id: str | None = None,
        reason: str | None = None,
        event_type: str = "deploy",
    ) -> WorkflowDeploymentRecord:
        async with self._session_factory() as session:
            deployment = await session.scalar(
                select(WorkflowDeploymentRecord).where(
                    WorkflowDeploymentRecord.workflow_id == workflow_id
                )
            )
            current = deployment.version_id if deployment else None
            if current != expected_current_version_id:
                raise RuntimeError("Workflow deployment revision conflict")
            if deployment is None:
                deployment = WorkflowDeploymentRecord(
                    id=str(uuid4()), workflow_id=workflow_id, version_id=version_id
                )
                session.add(deployment)
            else:
                deployment.version_id = version_id
                deployment.updated_at = datetime.now(UTC)
            session.add(
                WorkflowDeploymentEventRecord(
                    id=str(uuid4()),
                    workflow_id=workflow_id,
                    from_version_id=current,
                    to_version_id=version_id,
                    event_type=event_type,
                    reason=reason,
                )
            )
            await session.commit()
            await session.refresh(deployment)
            return deployment

    async def get_deployment(
        self, workflow_id: str
    ) -> tuple[WorkflowDeploymentRecord, WorkflowVersionRecord] | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(WorkflowDeploymentRecord, WorkflowVersionRecord)
                    .join(
                        WorkflowVersionRecord,
                        WorkflowVersionRecord.id == WorkflowDeploymentRecord.version_id,
                    )
                    .where(WorkflowDeploymentRecord.workflow_id == workflow_id)
                )
            ).first()
            return cast(tuple[WorkflowDeploymentRecord, WorkflowVersionRecord] | None, row)

    async def save_run(
        self,
        *,
        run_id: str,
        workflow_id: str,
        version_id: str | None,
        business_type: str,
        preview: bool,
        status: str,
        input_summary: dict[str, object],
        output_summary: dict[str, object],
        traces: list[NodeTrace],
        started_at: datetime,
        completed_at: datetime,
        error: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                WorkflowRunRecord(
                    id=run_id,
                    workflow_id=workflow_id,
                    version_id=version_id,
                    resolved_version_id=version_id,
                    business_type=business_type,
                    status=status,
                    preview=preview,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            )
            await session.flush()
            for index, trace in enumerate(traces):
                session.add(
                    WorkflowNodeRunRecord(
                        id=str(uuid4()),
                        run_id=run_id,
                        sequence=index,
                        node_id=trace.node_id,
                        node_type=trace.node_type,
                        component_version_id=trace.component_version_id,
                        status=trace.status,
                        duration_ms=trace.duration_ms,
                        input_summary=trace.input_summary,
                        output_summary=trace.output_summary,
                        input_port_summary=trace.input_port_summary,
                        output_port_summary=trace.output_port_summary,
                        skip_reason=trace.skip_reason,
                        error=trace.error,
                    )
                )
            await session.commit()

    async def get_run(
        self, run_id: str
    ) -> tuple[WorkflowRunRecord, list[WorkflowNodeRunRecord]] | None:
        async with self._session_factory() as session:
            run = await session.get(WorkflowRunRecord, run_id)
            if run is None:
                return None
            nodes = list(
                await session.scalars(
                    select(WorkflowNodeRunRecord)
                    .where(WorkflowNodeRunRecord.run_id == run_id)
                    .order_by(WorkflowNodeRunRecord.sequence)
                )
            )
            return run, nodes

    async def list_runs(
        self,
        *,
        workflow_id: str | None = None,
        status: str | None = None,
        version_id: str | None = None,
        preview: bool | None = None,
    ) -> list[WorkflowRunRecord]:
        async with self._session_factory() as session:
            query = select(WorkflowRunRecord).order_by(WorkflowRunRecord.started_at.desc())
            if workflow_id:
                query = query.where(WorkflowRunRecord.workflow_id == workflow_id)
            if status:
                query = query.where(WorkflowRunRecord.status == status)
            if version_id:
                query = query.where(WorkflowRunRecord.resolved_version_id == version_id)
            if preview is not None:
                query = query.where(WorkflowRunRecord.preview == preview)
            return list(await session.scalars(query))
