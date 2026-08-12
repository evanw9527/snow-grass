from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast

from snow_grass.workflow.schemas import ComponentVersionManifest


@dataclass(frozen=True)
class NodeExecutionContext:
    run_id: str
    flow_version_id: str | None
    node_id: str
    component_version_id: str
    config: dict[str, Any]
    inputs_by_port: dict[str, Any]
    workflow_input: dict[str, Any]
    deadline: datetime | None = None
    trace_context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeExecutionResult:
    outputs_by_port: dict[str, Any]
    active_ports: frozenset[str] | None = None

    def is_active(self, port: str) -> bool:
        return self.active_ports is None or port in self.active_ports


class ComponentRuntime(Protocol):
    async def execute(self, context: NodeExecutionContext) -> NodeExecutionResult: ...


class RuntimeFactory(Protocol):
    implementation_key: str
    implementation_digest: str

    def create(self) -> ComponentRuntime: ...


class FactoryCatalog(Protocol):
    def get(self, implementation_key: str) -> Any: ...


class ComponentRuntimeRegistry:
    """Owns exactly one immutable runtime object per ComponentVersion in this process."""

    def __init__(self, factories: FactoryCatalog) -> None:
        self._factories = factories
        self._instances: dict[str, ComponentRuntime] = {}
        self._manifests: dict[str, tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, version: ComponentVersionManifest) -> ComponentRuntime:
        identity = (version.implementation_key, version.implementation_digest)
        cached = self._instances.get(version.id)
        if cached is not None:
            if self._manifests[version.id] != identity:
                raise RuntimeError(
                    f"ComponentVersion {version.id} was rebound to a different implementation"
                )
            return cached

        async with self._lock:
            cached = self._instances.get(version.id)
            if cached is not None:
                if self._manifests[version.id] != identity:
                    raise RuntimeError(
                        f"ComponentVersion {version.id} was rebound to a different implementation"
                    )
                return cached
            factory = self._factories.get(version.implementation_key)
            if factory.implementation_digest != version.implementation_digest:
                raise RuntimeError(
                    f"Implementation digest mismatch for ComponentVersion {version.id}"
                )
            instance = cast(ComponentRuntime, factory.create())
            self._instances[version.id] = instance
            self._manifests[version.id] = identity
            return instance

    async def prepare(self, versions: list[ComponentVersionManifest]) -> None:
        for version in versions:
            await self.resolve(version)

    def cached_count(self) -> int:
        return len(self._instances)
