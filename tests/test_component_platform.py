from __future__ import annotations

import asyncio

import pytest

from snow_grass.workflow.registry import ComponentRuntimeFactory, RuntimeFactoryCatalog
from snow_grass.workflow.runtime import (
    ComponentRuntimeRegistry,
    NodeExecutionContext,
    NodeExecutionResult,
)
from snow_grass.workflow.schemas import ComponentVersionManifest


class _EchoRuntime:
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        await asyncio.sleep(0)
        return NodeExecutionResult({"result": {"run": context.run_id, "config": context.config}})


def _version(version_id: str, digest: str = "sha256:echo-v1") -> ComponentVersionManifest:
    return ComponentVersionManifest(
        id=version_id,
        component_id="component",
        component_key="test.echo",
        version_number=1,
        implementation_key="test.echo@1",
        implementation_digest=digest,
        input_ports=[],
        output_ports=[],
    )


@pytest.mark.asyncio
async def test_component_version_has_one_singleton_under_concurrency() -> None:
    created = 0

    def build() -> _EchoRuntime:
        nonlocal created
        created += 1
        return _EchoRuntime()

    catalog = RuntimeFactoryCatalog()
    catalog.register(ComponentRuntimeFactory("test.echo@1", "sha256:echo-v1", build))
    registry = ComponentRuntimeRegistry(catalog)
    instances = await asyncio.gather(*(registry.resolve(_version("version-1")) for _ in range(100)))
    assert created == 1
    assert all(instance is instances[0] for instance in instances)


@pytest.mark.asyncio
async def test_singleton_execution_context_is_isolated_and_digest_fails_closed() -> None:
    catalog = RuntimeFactoryCatalog()
    catalog.register(ComponentRuntimeFactory("test.echo@1", "sha256:echo-v1", _EchoRuntime))
    registry = ComponentRuntimeRegistry(catalog)
    runtime = await registry.resolve(_version("version-1"))
    results = await asyncio.gather(
        *(
            runtime.execute(
                NodeExecutionContext(
                    run_id=f"run-{index}",
                    flow_version_id=None,
                    node_id="node",
                    component_version_id="version-1",
                    config={"index": index},
                    inputs_by_port={},
                    workflow_input={},
                )
            )
            for index in range(100)
        )
    )
    assert [item.outputs_by_port["result"]["config"]["index"] for item in results] == list(
        range(100)
    )
    with pytest.raises(RuntimeError, match="digest mismatch"):
        await registry.resolve(_version("version-2", "sha256:changed"))


@pytest.mark.asyncio
async def test_same_version_cannot_be_rebound() -> None:
    catalog = RuntimeFactoryCatalog()
    catalog.register(ComponentRuntimeFactory("test.echo@1", "sha256:echo-v1", _EchoRuntime))
    registry = ComponentRuntimeRegistry(catalog)
    await registry.resolve(_version("version-1"))
    changed = _version("version-1").model_copy(update={"implementation_key": "other@1"})
    with pytest.raises(RuntimeError, match="rebound"):
        await registry.resolve(changed)
