from __future__ import annotations

from collections.abc import AsyncIterator

from snow_grass.agent.events import AgentEvent
from snow_grass.agent.runner import AgentRunner
from snow_grass.agent.runtime import AgentRunRequest, AgentRuntimeInfo


class NativeAgentRuntime:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    def info(self) -> AgentRuntimeInfo:
        return AgentRuntimeInfo(
            id="native",
            name="自研 Agent",
            description="Snow Grass 自研 AgentRunner",
            capabilities=["tool_loop", "skill", "knowledge", "session"],
        )

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        async for event in self._runner.stream(
            model_id=request.model_id,
            context=request.context,
            skill_id=request.skill_id,
            session_id=request.session_id,
        ):
            event.data.setdefault("runtime_id", "native")
            event.data.setdefault("runtime_name", "自研 Agent")
            yield event
