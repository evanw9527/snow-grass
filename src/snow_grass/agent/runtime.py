from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from snow_grass.agent.events import AgentEvent
from snow_grass.memory.schema import ContextBundle


class AgentRuntimeInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool = True
    capabilities: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    model_id: str
    context: ContextBundle
    skill_id: str | None = None
    session_id: str | None = None


class AgentRuntime(Protocol):
    def info(self) -> AgentRuntimeInfo: ...

    def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]: ...
