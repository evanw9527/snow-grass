from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app
from snow_grass.persistence.database import Database
from snow_grass.persistence.repository import ToolResultCacheRepository
from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelInfo,
    ModelRequestOptions,
    ModelStreamChunk,
    TokenUsage,
    ToolCallDelta,
)
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.executor import SkillScriptExecutionError, SkillScriptExecutor
from snow_grass.skills.schema import LoadedSkill, SkillDraftContent
from snow_grass.tools.result_cache import ToolResultReuseService


def make_script_skill(*, script: str) -> LoadedSkill:
    manifest = {
        "schema_version": 1,
        "id": "script-test",
        "version": "1.0.0",
        "enabled": True,
        "selection": {"mode": "auto", "keywords": ["脚本测试"]},
        "models": {"allowed": []},
        "tools": {"allowed": []},
        "limits": {"max_steps": 4, "timeout_seconds": 10},
        "cache": {
            "enabled": True,
            "scope": "workspace",
            "ttl_seconds": 600,
            "error_ttl_seconds": 20,
        },
    }
    content = SkillDraftContent(
        files={
            "manifest.yaml": yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            "SKILL.md": (
                "---\n"
                "name: script-test\n"
                "description: 执行测试脚本\n"
                "---\n\n"
                "需要确定性结果时调用 scripts/example.py。\n"
            ),
            "scripts/example.py": script,
        }
    )
    return content.to_loaded_skill(skill_id="script-test", version="1.0.0", enabled=True)


@pytest.mark.asyncio
async def test_script_executor_limits_path_timeout_and_output() -> None:
    skill = make_script_skill(
        script=(
            "import sys, time\n"
            "if sys.argv[1:] == ['sleep']:\n"
            "    time.sleep(5)\n"
            "elif sys.argv[1:] == ['large']:\n"
            "    print('x' * 1000)\n"
            "else:\n"
            "    print('hello ' + ' '.join(sys.argv[1:]))\n"
        )
    )
    executor = SkillScriptExecutor(
        enabled=True,
        timeout_seconds=1,
        max_output_chars=64,
    )

    success = await executor.execute(
        skill=skill, script="scripts/example.py", args=["world"]
    )
    assert success.ok is True
    assert success.stdout == "hello world\n"

    with pytest.raises(SkillScriptExecutionError) as unsafe:
        await executor.execute(skill=skill, script="../example.py", args=[])
    assert unsafe.value.code == "script_not_allowed"

    timed_out = await executor.execute(
        skill=skill, script="scripts/example.py", args=["sleep"]
    )
    assert timed_out.ok is False
    assert timed_out.timed_out is True

    truncated = await executor.execute(
        skill=skill, script="scripts/example.py", args=["large"]
    )
    assert truncated.ok is True
    assert truncated.truncated is True
    assert len(truncated.stdout) == 64


class ScriptCallingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        assert model_id == "provider-tool-model"
        assert tools is not None
        assert options is None
        assert tools[0].function.name == "run_skill_script"
        self.calls += 1
        if messages[-1].role != "tool":
            city = "Shanghai" if "上海" in messages[-1].content else "Beijing"
            yield ModelStreamChunk(
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0,
                        id="call_script",
                        name="run_skill_script",
                        arguments='{"script":"scripts/example.py",',
                    )
                ]
            )
            yield ModelStreamChunk(
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0,
                        arguments=(
                            f'"args":["{city}"],"cache_policy":"refresh"}}'
                        ),
                    )
                ]
            )
            yield ModelStreamChunk(
                usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
            )
            return

        assert messages[-1].role == "tool"
        tool_result = json.loads(messages[-1].content)
        assert tool_result["ok"] is True
        latest_user = next(
            message.content for message in reversed(messages) if message.role == "user"
        )
        expected_city = "Shanghai" if "上海" in latest_user else "Beijing"
        assert tool_result["stdout"] == f"weather:{expected_city}\n"
        yield ModelStreamChunk(delta="北京天气查询完成")
        yield ModelStreamChunk(
            usage=TokenUsage(input_tokens=14, output_tokens=4, total_tokens=18)
        )


def test_agent_executes_selected_skill_script(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills" / "script-test"
    (skill_root / "scripts").mkdir(parents=True)
    loaded = make_script_skill(script="import sys\nprint('weather:' + sys.argv[1])\n")
    for relative_path, serialized in loaded.package_files.items():
        destination = skill_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")

    provider = ScriptCallingProvider()
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="tool-model",
            name="Tool Model",
            provider="tool-test",
            provider_name="Tool Test",
            capabilities=["chat", "streaming", "tool-calling"],
            available=True,
        ),
        provider=provider,
        provider_model_id="provider-tool-model",
    )
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'tool.db'}",
        default_model_id="tool-model",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=tmp_path / "skills",
        skill_script_execution_enabled=True,
    )

    with TestClient(create_app(settings=settings, providers=providers)) as client:
        session = client.post(
            "/api/v1/sessions",
            json={
                "title": "Script execution",
                "model_id": "tool-model",
                "skill_id": "script-test",
            },
        ).json()
        response = client.post(
            f"/api/v1/sessions/{session['id']}/messages/stream",
            json={"content": "执行脚本测试"},
        )
        assert response.status_code == 200
        assert '"step": "execute_skill_script"' in response.text
        assert '"skill_id": "script-test"' in response.text
        assert '"skill_name": "script-test"' in response.text
        assert '"skill_version": "1.0.0"' in response.text
        assert '"script": "scripts/example.py"' in response.text
        assert '"cache_status": "miss"' in response.text
        assert "北京天气查询完成" in response.text

        cached = client.post(
            f"/api/v1/sessions/{session['id']}/messages/stream",
            json={"content": "再次执行脚本测试"},
        )
        assert cached.status_code == 200
        assert '"cache_status": "hit"' in cached.text
        assert '"hit_count": 1' in cached.text

        refreshed = client.post(
            f"/api/v1/sessions/{session['id']}/messages/stream",
            json={"content": "重新查询脚本测试"},
        )
        assert refreshed.status_code == 200
        assert '"cache_status": "refresh"' in refreshed.text

        different_arguments = client.post(
            f"/api/v1/sessions/{session['id']}/messages/stream",
            json={"content": "查询上海脚本测试"},
        )
        assert different_arguments.status_code == 200
        assert '"cache_status": "miss"' in different_arguments.text

        messages = client.get(f"/api/v1/sessions/{session['id']}/messages").json()
        assert messages[-1]["content"] == "北京天气查询完成"
        assert messages[-1]["skill_id"] == "script-test"
        assert messages[-1]["input_tokens"] == 24
        assert messages[-1]["output_tokens"] == 6
        assert messages[-1]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_cache_key_normalizes_weather_city_and_isolates_skill_version(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'cache-key.db'}")
    await database.create_schema()
    service = ToolResultReuseService(
        repository=ToolResultCacheRepository(database.session_factory),
        workspace_id="test-workspace",
    )
    skill = make_script_skill(script="print('ok')\n")
    skill.manifest.id = "weather"
    city_alias = await service.resolve(
        skill=skill,
        script="scripts/example.py",
        args=["北京市"],
        session_id="session-a",
        run_id="run-a",
        policy="prefer-cache",
        force_refresh=False,
    )
    normalized_city = await service.resolve(
        skill=skill,
        script="scripts/example.py",
        args=["北京"],
        session_id="session-b",
        run_id="run-b",
        policy="prefer-cache",
        force_refresh=False,
    )
    next_version = skill.model_copy(deep=True)
    next_version.manifest.version = "1.0.1"
    changed_version = await service.resolve(
        skill=next_version,
        script="scripts/example.py",
        args=["北京"],
        session_id="session-c",
        run_id="run-c",
        policy="prefer-cache",
        force_refresh=False,
    )

    assert city_alias.cache_key == normalized_city.cache_key
    assert changed_version.cache_key != normalized_city.cache_key
    await database.dispose()
