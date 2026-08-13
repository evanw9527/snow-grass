from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from snow_grass.providers.base import ChatTool
from snow_grass.skills.executor import (
    RUN_SKILL_SCRIPT_TOOL,
    SkillScriptExecutionError,
    SkillScriptExecutor,
    SkillScriptResult,
)
from snow_grass.skills.schema import LoadedSkill
from snow_grass.tools.registry import ToolDefinition, ToolRegistry
from snow_grass.tools.result_cache import CachePolicy, ToolResultReuseService


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    skill: LoadedSkill | None
    session_id: str | None
    run_id: str
    runtime_id: str = "native"
    force_refresh: bool = False


BoundToolHandler = Callable[
    [dict[str, Any], ToolInvocationContext], Awaitable[tuple[str, dict[str, object]]]
]


@dataclass(frozen=True, slots=True)
class BoundTool:
    definition: ToolDefinition
    handler: BoundToolHandler

    def to_chat_tool(self) -> ChatTool:
        return self.definition.to_chat_tool()


class SkillScriptToolAdapter:
    """Binds reviewed scripts from one active Skill as a runtime-neutral Tool."""

    def __init__(
        self,
        *,
        executor: SkillScriptExecutor,
        result_cache: ToolResultReuseService,
    ) -> None:
        self._executor = executor
        self._result_cache = result_cache

    def bind(self, skill: LoadedSkill | None) -> BoundTool | None:
        if skill is None:
            return None
        scripts = self._executor.list_scripts(skill)
        if not scripts:
            return None
        definition = ToolDefinition(
            name=RUN_SKILL_SCRIPT_TOOL,
            description=(
                "Run one reviewed Python script from the active Skill package. "
                f"Available scripts: {', '.join(scripts)}. Pass arguments as separate strings."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "enum": scripts,
                        "description": "Relative path of the packaged Python script.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 32,
                    },
                    "cache_policy": {
                        "type": "string",
                        "enum": ["prefer-cache", "refresh", "no-store"],
                    },
                },
                "required": ["script"],
                "additionalProperties": False,
            },
            handler=self._unbound_handler,
            risk_level="high",
            result_type="skill-script-result",
        )
        return BoundTool(definition=definition, handler=self._execute)

    async def _execute(
        self, arguments: dict[str, Any], context: ToolInvocationContext
    ) -> tuple[str, dict[str, object]]:
        skill = context.skill
        if skill is None:
            return self._error("no_active_skill", "No active Skill is available")
        script = arguments.get("script")
        args = arguments.get("args", [])
        if (
            not isinstance(script, str)
            or not isinstance(args, list)
            or not all(isinstance(argument, str) for argument in args)
        ):
            return self._error(
                "invalid_tool_arguments",
                "script must be a string and args must be strings",
            )
        # The model may suggest a cache policy, but only explicit user refresh intent can
        # override the server policy. This keeps execution deterministic and auditable.
        policy: CachePolicy = "refresh" if context.force_refresh else "prefer-cache"
        decision = await self._result_cache.resolve(
            skill=skill,
            script=script,
            args=args,
            session_id=context.session_id,
            run_id=context.run_id,
            policy=policy,
            force_refresh=context.force_refresh,
        )
        if decision.status == "hit" and decision.result_json is not None:
            result = SkillScriptResult.model_validate(decision.result_json)
            return result.to_tool_message(), {
                **self._result_event_data(result),
                **self._result_cache.event_data(decision),
                "duration_ms": 0,
                "original_duration_ms": result.duration_ms,
            }
        try:
            result = await self._executor.execute(skill=skill, script=script, args=args)
        except SkillScriptExecutionError as exc:
            return self._error(exc.code, str(exc), script=script)
        cached_record = await self._result_cache.save(
            decision=decision,
            skill=skill,
            script=script,
            session_id=context.session_id,
            result_json=result.model_dump(mode="json"),
            ok=result.ok,
        )
        return result.to_tool_message(), {
            **self._result_event_data(result),
            **self._result_cache.event_data(decision, cached_record),
        }

    @staticmethod
    async def _unbound_handler(_: dict[str, Any]) -> Any:
        raise RuntimeError("Skill script tools must be bound to an active Skill")

    @staticmethod
    def _result_event_data(result: SkillScriptResult) -> dict[str, object]:
        return {
            "ok": result.ok,
            "script": result.script,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "duration_ms": result.duration_ms,
        }

    @staticmethod
    def _error(
        code: str, message: str, *, script: str | None = None
    ) -> tuple[str, dict[str, object]]:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {"code": code, "message": message},
        }
        event_data: dict[str, object] = {"ok": False, "error_code": code}
        if script:
            payload["script"] = script
            event_data["script"] = script
        return json.dumps(payload, ensure_ascii=False), event_data


class ToolDispatcher:
    """Resolves the active Skill's tool surface and executes every call through one boundary."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        script_adapter: SkillScriptToolAdapter,
    ) -> None:
        self._registry = registry
        self._script_adapter = script_adapter

    def resolve(self, skill: LoadedSkill | None) -> list[BoundTool]:
        allowed = skill.manifest.tools.allowed if skill else []
        bound = [self._bind_registered(tool) for tool in self._registry.list_allowed(allowed)]
        script_tool = self._script_adapter.bind(skill)
        if script_tool is not None:
            bound.append(script_tool)
        return bound

    async def execute(
        self,
        *,
        name: str,
        arguments_json: str,
        tools: Sequence[BoundTool],
        context: ToolInvocationContext,
    ) -> tuple[str, dict[str, object]]:
        tool = next((candidate for candidate in tools if candidate.definition.name == name), None)
        if tool is None:
            return self._error("unknown_tool", "The requested tool is not available")
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return self._error("invalid_tool_arguments", "Tool arguments are not valid JSON")
        if not isinstance(arguments, dict):
            return self._error("invalid_tool_arguments", "Tool arguments must be an object")
        validation_error = self._validate_arguments(tool.definition.parameters, arguments)
        if validation_error:
            return self._error("invalid_tool_arguments", validation_error)
        content, event_data = await tool.handler(arguments, context)
        return content, {"tool_name": name, **event_data}

    @staticmethod
    def chat_tools(tools: Sequence[BoundTool]) -> list[ChatTool]:
        return [tool.to_chat_tool() for tool in tools]

    @staticmethod
    def _bind_registered(definition: ToolDefinition) -> BoundTool:
        async def execute(
            arguments: dict[str, Any], _: ToolInvocationContext
        ) -> tuple[str, dict[str, object]]:
            try:
                result = await definition.handler(arguments)
            except Exception as exc:
                return ToolDispatcher._error("tool_execution_failed", str(exc))
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            return content, {"ok": True}

        return BoundTool(definition=definition, handler=execute)

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        required = schema.get("required", [])
        for name in required:
            if name not in arguments:
                return f"Missing required argument: {name}"
        if schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            unknown = set(arguments) - set(properties)
            if unknown:
                return f"Unknown arguments: {', '.join(sorted(unknown))}"
        return None

    @staticmethod
    def _error(code: str, message: str) -> tuple[str, dict[str, object]]:
        return (
            json.dumps(
                {"ok": False, "error": {"code": code, "message": message}},
                ensure_ascii=False,
            ),
            {"ok": False, "error_code": code},
        )
