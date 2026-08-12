from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid5

from snow_grass.workflow.runtime import ComponentRuntime
from snow_grass.workflow.schemas import FlowGraph, ValidationIssue

ComponentRuntimeBuilder = Callable[[], ComponentRuntime]
BusinessValidator = Callable[[FlowGraph], list[ValidationIssue]]
InputAdapter = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
OutputAdapter = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
SYSTEM_COMPONENT_NAMESPACE = UUID("a9941625-fbc0-4b59-bb3b-729cb75dd58d")


@dataclass(frozen=True)
class ComponentRuntimeFactory:
    implementation_key: str
    implementation_digest: str
    builder: ComponentRuntimeBuilder
    source_code: str = ""
    source_language: str = "python"
    source_path: str | None = None

    def create(self) -> ComponentRuntime:
        return self.builder()


class RuntimeFactoryCatalog:
    def __init__(self) -> None:
        self._factories: dict[str, ComponentRuntimeFactory] = {}

    def register(self, factory: ComponentRuntimeFactory) -> None:
        existing = self._factories.get(factory.implementation_key)
        if existing is not None:
            if existing.implementation_digest != factory.implementation_digest:
                raise ValueError(
                    f"Implementation key already registered with another digest: "
                    f"{factory.implementation_key}"
                )
            raise ValueError(f"Implementation key already registered: {factory.implementation_key}")
        self._factories[factory.implementation_key] = factory

    def get(self, implementation_key: str) -> ComponentRuntimeFactory:
        try:
            return self._factories[implementation_key]
        except KeyError as exc:
            raise LookupError(f"Unknown implementation: {implementation_key}") from exc

    def all(self) -> list[ComponentRuntimeFactory]:
        return list(self._factories.values())

    def source_snapshot(self, implementation_key: str) -> tuple[str, str, str | None]:
        factory = self.get(implementation_key)
        return factory.source_code, factory.source_language, factory.source_path


def system_component_id(component_key: str) -> str:
    return str(uuid5(SYSTEM_COMPONENT_NAMESPACE, f"component:{component_key}"))


def system_component_version_id(component_key: str, version: int = 1) -> str:
    return str(uuid5(SYSTEM_COMPONENT_NAMESPACE, f"component:{component_key}:v{version}"))


@dataclass(frozen=True)
class SystemComponentSpec:
    key: str
    name: str
    description: str
    category: str
    implementation_key: str
    implementation_digest: str
    config_schema: dict[str, Any]
    ui_schema: dict[str, Any]
    input_ports: tuple[dict[str, Any], ...]
    output_ports: tuple[dict[str, Any], ...]
    icon: str = "tool"

    @property
    def component_id(self) -> str:
        return system_component_id(self.key)

    @property
    def version_id(self) -> str:
        return system_component_version_id(self.key)


@dataclass(frozen=True)
class BusinessPack:
    business_type: str
    title: str
    description: str
    allowed_component_keys: frozenset[str]
    component_specs: tuple[SystemComponentSpec, ...]
    default_graph: FlowGraph
    validator: BusinessValidator | None = None
    input_adapter: InputAdapter | None = None
    output_adapter: OutputAdapter | None = None


class BusinessPackRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, BusinessPack] = {}

    def register(self, pack: BusinessPack) -> None:
        if pack.business_type in self._packs:
            raise ValueError(f"Business pack already registered: {pack.business_type}")
        self._packs[pack.business_type] = pack

    def get(self, business_type: str) -> BusinessPack:
        try:
            return self._packs[business_type]
        except KeyError as exc:
            raise LookupError(f"Unknown business pack: {business_type}") from exc

    def all(self) -> list[BusinessPack]:
        return list(self._packs.values())
