from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, cast

from snow_grass.workflow.compiler import WorkflowCompiler
from snow_grass.workflow.data_migration import WorkflowV2DataMigration
from snow_grass.workflow.executor import WorkflowExecutor
from snow_grass.workflow.registry import BusinessPackRegistry, RuntimeFactoryCatalog
from snow_grass.workflow.repository import WorkflowRepository
from snow_grass.workflow.runtime import ComponentRuntimeRegistry, NodeExecutionContext
from snow_grass.workflow.schemas import (
    BusinessPackResponse,
    ComponentCreateRequest,
    ComponentDetail,
    ComponentManifest,
    ComponentPublishResponse,
    ComponentSummary,
    ComponentTestResponse,
    ComponentVersionResponse,
    DeploymentResponse,
    FlowGraph,
    FlowRunResponse,
    FlowValidation,
    NodeTrace,
    PortDefinition,
    PublishResponse,
    WorkflowDetail,
    WorkflowSummary,
    WorkflowVersionSummary,
)


class ComponentService:
    def __init__(
        self,
        *,
        repository: WorkflowRepository,
        factories: RuntimeFactoryCatalog,
        runtimes: ComponentRuntimeRegistry,
        packs: BusinessPackRegistry,
    ) -> None:
        self._repository = repository
        self._factories = factories
        self._runtimes = runtimes
        self._packs = packs

    async def list_components(
        self, *, query: str | None = None, category: str | None = None, status: str | None = None
    ) -> list[ComponentSummary]:
        return [
            await self._summary(item)
            for item in await self._repository.list_components(
                query=query, category=category, status=status
            )
        ]

    async def create(self, payload: ComponentCreateRequest) -> ComponentDetail:
        self._factories.get(payload.implementation_key)
        manifest = ComponentManifest(
            icon=payload.icon, implementation_key=payload.implementation_key
        )
        record = await self._repository.create_component(
            key=payload.key,
            name=payload.name,
            description=payload.description,
            category=payload.category,
            icon=payload.icon,
            manifest=manifest,
        )
        return await self.get(record.id)

    async def get(self, component_id: str) -> ComponentDetail:
        record = await self._repository.get_component(component_id)
        if record is None:
            raise LookupError("Component not found")
        summary = await self._summary(record)
        versions = [
            self._version(v, record)
            for v in await self._repository.list_component_versions(component_id)
        ]
        return ComponentDetail(
            **summary.model_dump(),
            draft=ComponentManifest.model_validate(record.draft_manifest),
            versions=versions,
        )

    async def save_draft(
        self, component_id: str, *, expected_revision: int, manifest: ComponentManifest
    ) -> ComponentDetail:
        self._factories.get(manifest.implementation_key)
        await self._repository.save_component_draft(
            component_id, expected_revision=expected_revision, manifest=manifest
        )
        return await self.get(component_id)

    async def publish(
        self, component_id: str, *, expected_revision: int, release_note: str | None
    ) -> ComponentPublishResponse:
        component = await self._repository.get_component(component_id)
        if component is None:
            raise LookupError("Component not found")
        manifest = ComponentManifest.model_validate(component.draft_manifest)
        factory = self._factories.get(manifest.implementation_key)
        source_code, source_language, source_path = self._factories.source_snapshot(
            manifest.implementation_key
        )
        version = await self._repository.create_component_version(
            component_id,
            expected_revision=expected_revision,
            digest=factory.implementation_digest,
            implementation_source=source_code,
            source_language=source_language,
            source_path=source_path,
            release_note=release_note,
        )
        response = self._version(version, component)
        await self._runtimes.resolve(response)
        return ComponentPublishResponse(
            component_version_id=version.id,
            version_number=version.version_number,
            implementation_digest=version.implementation_digest,
            published_at=version.published_at,
        )

    async def test(
        self,
        component_id: str,
        *,
        expected_revision: int,
        config: dict[str, Any],
        inputs: dict[str, Any],
    ) -> ComponentTestResponse:
        component = await self._repository.get_component(component_id)
        if component is None:
            raise LookupError("Component not found")
        if component.draft_revision != expected_revision:
            raise RuntimeError("Component draft revision conflict")
        manifest = ComponentManifest.model_validate(component.draft_manifest)
        factory = self._factories.get(manifest.implementation_key)
        started = perf_counter()
        result = await factory.create().execute(
            NodeExecutionContext(
                run_id="component-test",
                flow_version_id=None,
                node_id="component-test",
                component_version_id="draft",
                config=config,
                inputs_by_port=inputs,
                workflow_input=inputs,
            )
        )
        return ComponentTestResponse(
            output=result.outputs_by_port,
            duration_ms=max(0, int((perf_counter() - started) * 1000)),
        )

    async def set_status(self, component_id: str, status: str) -> ComponentDetail:
        await self._repository.update_component_status(component_id, status)
        return await self.get(component_id)

    async def library(self) -> list[ComponentVersionResponse]:
        result: list[ComponentVersionResponse] = []
        for component in await self._repository.list_components(status="active"):
            versions = await self._repository.list_component_versions(component.id)
            result.extend(self._version(version, component) for version in versions)
        return result

    async def _summary(self, record: Any) -> ComponentSummary:
        versions = await self._repository.list_component_versions(record.id)
        return ComponentSummary(
            id=record.id,
            key=record.key,
            name=record.name,
            description=record.description,
            category=record.category,
            icon=record.icon,
            status=record.status,
            draft_revision=record.draft_revision,
            latest_version=versions[0].version_number if versions else None,
            reference_count=await self._repository.component_reference_count(record.id),
            updated_at=record.updated_at,
        )

    @staticmethod
    def _version(version: Any, component: Any) -> ComponentVersionResponse:
        return ComponentVersionResponse(
            id=version.id,
            component_id=component.id,
            component_key=component.key,
            component_name=component.name,
            component_description=component.description,
            category=component.category,
            icon=version.icon,
            version_number=version.version_number,
            status=component.status,
            implementation_key=version.implementation_key,
            implementation_digest=version.implementation_digest,
            executor_contract_version=version.executor_contract_version,
            config_schema=version.config_schema,
            ui_schema=version.ui_schema,
            input_ports=[PortDefinition.model_validate(p) for p in version.input_ports],
            output_ports=[PortDefinition.model_validate(p) for p in version.output_ports],
            release_note=version.release_note,
            published_at=version.published_at,
            implementation_source=version.implementation_source,
            source_language=version.source_language,
            source_path=version.source_path,
        )


class WorkflowService:
    def __init__(
        self,
        *,
        repository: WorkflowRepository,
        packs: BusinessPackRegistry,
        factories: RuntimeFactoryCatalog,
        compiler: WorkflowCompiler,
        executor: WorkflowExecutor,
    ) -> None:
        self._repository = repository
        self._packs = packs
        self._factories = factories
        self._compiler = compiler
        self._executor = executor

    async def initialize(self) -> None:
        specs = {spec.key: spec for pack in self._packs.all() for spec in pack.component_specs}
        source_snapshots = {
            factory.implementation_key: (
                factory.source_code,
                factory.source_language,
                factory.source_path,
            )
            for factory in self._factories.all()
        }
        await self._repository.seed_system_components(list(specs.values()), source_snapshots)
        await WorkflowV2DataMigration(self._repository.session_factory).migrate()
        for pack in self._packs.all():
            workflow = await self._repository.get_by_business(pack.business_type)
            if workflow is None:
                workflow = await self._repository.create_workflow(
                    business_type=pack.business_type,
                    name=pack.title,
                    description=pack.description,
                    graph=pack.default_graph,
                )
            if await self._repository.get_deployment(workflow.id) is None:
                plan = await self._compiler.compile(pack.default_graph, pack)
                version = await self._repository.create_version(
                    workflow,
                    graph=pack.default_graph,
                    note="系统默认版本",
                    component_manifest_digest=plan.component_manifest_digest,
                )
                await self._repository.deploy(
                    workflow.id, version.id, expected_current_version_id=None, reason="系统初始化"
                )

    def list_business_packs(self) -> list[BusinessPackResponse]:
        return [
            BusinessPackResponse(
                business_type=p.business_type,
                title=p.title,
                description=p.description,
                allowed_component_keys=sorted(p.allowed_component_keys),
            )
            for p in self._packs.all()
        ]

    async def list_workflows(self, business_type: str | None = None) -> list[WorkflowSummary]:
        return [
            await self._summary(r) for r in await self._repository.list_workflows(business_type)
        ]

    async def create_workflow(
        self, *, business_type: str, name: str, description: str, graph: FlowGraph | None = None
    ) -> WorkflowDetail:
        pack = self._packs.get(business_type)
        record = await self._repository.create_workflow(
            business_type=business_type,
            name=name,
            description=description,
            graph=graph or pack.default_graph,
        )
        return await self.get_workflow(record.id)

    async def copy_workflow(self, workflow_id: str, *, name: str) -> WorkflowDetail:
        source = await self._require_workflow(workflow_id)
        return await self.create_workflow(
            business_type=source.business_type,
            name=name,
            description=source.description,
            graph=FlowGraph.model_validate(source.draft_graph),
        )

    async def set_workflow_status(self, workflow_id: str, status: str) -> WorkflowDetail:
        await self._repository.update_workflow_status(workflow_id, status)
        return await self.get_workflow(workflow_id)

    async def list_versions(self, workflow_id: str) -> list[WorkflowVersionSummary]:
        await self._require_workflow(workflow_id)
        return [
            WorkflowVersionSummary(
                id=version.id,
                workflow_id=version.workflow_id,
                version_number=version.version_number,
                note=version.note,
                component_manifest_digest=version.component_manifest_digest,
                published_at=version.published_at,
            )
            for version in await self._repository.list_versions(workflow_id)
        ]

    async def get_workflow(self, workflow_id: str) -> WorkflowDetail:
        record = await self._require_workflow(workflow_id)
        summary = await self._summary(record)
        graph = FlowGraph.model_validate(record.draft_graph)
        return WorkflowDetail(
            **summary.model_dump(),
            graph=graph,
            validation=await self.validate(record.business_type, graph),
        )

    async def save_draft(
        self, workflow_id: str, *, expected_revision: int, graph: FlowGraph
    ) -> WorkflowDetail:
        record = await self._require_workflow(workflow_id)
        validation = await self.validate(record.business_type, graph)
        if not validation.valid:
            raise ValueError(validation.errors[0].message)
        await self._repository.save_draft(
            workflow_id, expected_revision=expected_revision, graph=graph
        )
        return await self.get_workflow(workflow_id)

    async def validate(self, business_type: str, graph: FlowGraph) -> FlowValidation:
        return await self._compiler.validate(graph, self._packs.get(business_type))

    async def validate_workflow(self, workflow_id: str) -> FlowValidation:
        record = await self._require_workflow(workflow_id)
        return await self.validate(
            record.business_type, FlowGraph.model_validate(record.draft_graph)
        )

    async def publish(
        self, workflow_id: str, *, expected_revision: int, note: str | None, activate: bool
    ) -> PublishResponse:
        workflow = await self._require_workflow(workflow_id)
        if workflow.draft_revision != expected_revision:
            raise RuntimeError("Workflow draft revision conflict")
        graph = FlowGraph.model_validate(workflow.draft_graph)
        plan = await self._compiler.compile(graph, self._packs.get(workflow.business_type))
        version = await self._repository.create_version(
            workflow,
            graph=graph,
            note=note,
            component_manifest_digest=plan.component_manifest_digest,
        )
        deployment = None
        if activate:
            current = await self._repository.get_deployment(workflow.id)
            deployment = await self._repository.deploy(
                workflow.id,
                version.id,
                expected_current_version_id=current[1].id if current else None,
                reason="发布并部署",
            )
        return PublishResponse(
            version_id=version.id,
            version_number=version.version_number,
            deployment_id=deployment.id if deployment else None,
            active=deployment is not None,
            published_at=version.published_at,
        )

    async def deploy(
        self,
        workflow_id: str,
        *,
        version_id: str,
        expected_current_version_id: str | None,
        reason: str | None,
        event_type: Literal["deploy", "rollback"] = "deploy",
    ) -> DeploymentResponse:
        workflow = await self._require_workflow(workflow_id)
        version = await self._repository.get_version(version_id)
        if version is None or version.workflow_id != workflow_id:
            raise LookupError("Workflow version not found")
        await self._compiler.compile(
            FlowGraph.model_validate(version.graph), self._packs.get(workflow.business_type)
        )
        deployment = await self._repository.deploy(
            workflow_id,
            version_id,
            expected_current_version_id=expected_current_version_id,
            reason=reason,
            event_type=event_type,
        )
        return DeploymentResponse(
            deployment_id=deployment.id,
            workflow_id=workflow_id,
            from_version_id=expected_current_version_id,
            version_id=version_id,
            event_type=event_type,
            reason=reason,
            updated_at=deployment.updated_at,
        )

    async def preview(
        self, workflow_id: str, *, expected_revision: int, workflow_input: dict[str, Any]
    ) -> FlowRunResponse:
        workflow = await self._require_workflow(workflow_id)
        if workflow.draft_revision != expected_revision:
            raise RuntimeError("Workflow draft revision conflict")
        return await self._run(
            workflow, FlowGraph.model_validate(workflow.draft_graph), None, workflow_input, True
        )

    async def run_business(
        self, business_type: str, workflow_input: dict[str, Any]
    ) -> FlowRunResponse:
        workflow = await self._repository.get_by_business(business_type)
        if workflow is None:
            raise LookupError(f"Workflow not found for business: {business_type}")
        deployed = await self._repository.get_deployment(workflow.id)
        if deployed is None:
            raise RuntimeError(f"No deployed workflow for business: {business_type}")
        return await self._run(
            workflow,
            FlowGraph.model_validate(deployed[1].graph),
            deployed[1].id,
            workflow_input,
            False,
        )

    async def _run(
        self,
        workflow: Any,
        graph: FlowGraph,
        version_id: str | None,
        workflow_input: dict[str, Any],
        preview: bool,
    ) -> FlowRunResponse:
        pack = self._packs.get(workflow.business_type)
        prepared = (
            await pack.input_adapter(workflow_input) if pack.input_adapter else workflow_input
        )
        plan = await self._compiler.compile(graph, pack)
        result = await self._executor.execute(
            workflow_id=workflow.id,
            version_id=version_id,
            pack=pack,
            plan=plan,
            workflow_input=prepared,
            preview=preview,
        )
        await self._save_result(result, prepared)
        return result

    async def get_run(self, run_id: str) -> FlowRunResponse:
        found = await self._repository.get_run(run_id)
        if found is None:
            raise LookupError("Workflow run not found")
        run, nodes = found
        return FlowRunResponse(
            run_id=run.id,
            workflow_id=run.workflow_id,
            flow_version_id=run.version_id,
            resolved_version_id=run.resolved_version_id,
            business_type=run.business_type,
            status=cast(Any, run.status),
            preview=run.preview,
            output=run.output_summary,
            trace=[
                NodeTrace(
                    node_id=n.node_id,
                    component_version_id=n.component_version_id,
                    node_type=n.node_type,
                    status=cast(Any, n.status),
                    duration_ms=n.duration_ms,
                    input_summary=n.input_summary,
                    output_summary=n.output_summary,
                    input_port_summary=n.input_port_summary,
                    output_port_summary=n.output_port_summary,
                    skip_reason=n.skip_reason,
                    error=n.error,
                )
                for n in nodes
            ],
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    async def list_runs(
        self,
        *,
        workflow_id: str | None = None,
        status: str | None = None,
        version_id: str | None = None,
        preview: bool | None = None,
    ) -> list[FlowRunResponse]:
        records = await self._repository.list_runs(
            workflow_id=workflow_id,
            status=status,
            version_id=version_id,
            preview=preview,
        )
        return [await self.get_run(record.id) for record in records]

    async def _summary(self, record: Any) -> WorkflowSummary:
        version = await self._repository.latest_version(record.id)
        deployment = await self._repository.get_deployment(record.id)
        return WorkflowSummary(
            id=record.id,
            business_type=record.business_type,
            name=record.name,
            description=record.description,
            status=record.status,
            draft_revision=record.draft_revision,
            latest_version=version.version_number if version else None,
            deployed_version_id=deployment[1].id if deployment else None,
            updated_at=record.updated_at,
        )

    async def _require_workflow(self, workflow_id: str) -> Any:
        workflow = await self._repository.get_workflow(workflow_id)
        if workflow is None:
            raise LookupError("Workflow not found")
        return workflow

    async def _save_result(self, result: FlowRunResponse, workflow_input: dict[str, Any]) -> None:
        await self._repository.save_run(
            run_id=result.run_id,
            workflow_id=result.workflow_id,
            version_id=result.resolved_version_id,
            business_type=result.business_type,
            preview=result.preview,
            status=result.status,
            input_summary=_safe_summary(workflow_input),
            output_summary=_safe_summary(result.output),
            traces=result.trace,
            started_at=result.started_at,
            completed_at=result.completed_at or datetime.now(UTC),
            error=result.trace[-1].error if result.status == "failed" and result.trace else None,
        )


def _safe_summary(value: dict[str, Any]) -> dict[str, object]:
    return {
        key: ({"count": len(item)} if isinstance(item, list) else item)
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool, type(None), list))
    }
