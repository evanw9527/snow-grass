from __future__ import annotations

from snow_grass.agent.runtime import AgentRuntime, AgentRuntimeInfo


class AgentRuntimeError(RuntimeError):
    pass


class AgentRuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, AgentRuntime] = {}

    def register(self, runtime: AgentRuntime) -> None:
        info = runtime.info()
        if info.id in self._runtimes:
            raise ValueError(f"Duplicate Agent Runtime: {info.id}")
        self._runtimes[info.id] = runtime

    def list(self) -> list[AgentRuntimeInfo]:
        return [runtime.info() for runtime in self._runtimes.values()]

    def resolve(self, runtime_id: str) -> AgentRuntime:
        runtime = self._runtimes.get(runtime_id)
        if runtime is None:
            raise AgentRuntimeError(f"Unknown Agent Runtime: {runtime_id}")
        info = runtime.info()
        if not info.available:
            reason = info.unavailable_reason or "runtime is unavailable"
            raise AgentRuntimeError(f"Agent Runtime {runtime_id} is unavailable: {reason}")
        return runtime
