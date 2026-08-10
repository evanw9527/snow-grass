from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    """Registry of server-reviewed tools; Skills can only narrow this set."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate tool: {definition.name}")
        self._tools[definition.name] = definition

    def list_allowed(self, allowed_names: list[str]) -> list[ToolDefinition]:
        return [self._tools[name] for name in allowed_names if name in self._tools]

    async def execute(self, name: str, arguments: dict[str, Any], allowed_names: list[str]) -> Any:
        if name not in allowed_names:
            raise PermissionError(f"Tool {name!r} is not allowed by the active Skill")
        definition = self._tools.get(name)
        if definition is None:
            raise LookupError(f"Tool {name!r} is not registered")
        return await definition.handler(arguments)
