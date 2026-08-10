from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app
from snow_grass.memory.context_builder import ContextBuilder
from snow_grass.memory.schema import MemorySearchResult
from snow_grass.memory.token_counter import ConservativeTokenCounter
from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelInfo,
    ModelStreamChunk,
    TokenUsage,
)
from snow_grass.providers.registry import ProviderRegistry


class MemoryProvider:
    def __init__(self, *, fail_summary: bool = False) -> None:
        self.fail_summary = fail_summary
        self.summary_calls = 0
        self.extraction_calls = 0
        self.main_calls: list[Sequence[ChatMessage]] = []

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        assert model_id == "provider-memory-model"
        assert tools is None
        if messages[0].content.startswith("Summarize the conversation"):
            self.summary_calls += 1
            if self.fail_summary:
                raise RuntimeError("summary unavailable")
            yield ModelStreamChunk(delta="已确认需求和关键决策的滚动摘要。")
            return
        if messages[0].content.startswith("Extract at most 3 durable memory candidates"):
            self.extraction_calls += 1
            yield ModelStreamChunk(
                delta=(
                    '[{"memory_type":"project_fact","content":"项目使用 Python 开发",'
                    '"importance":0.8}]'
                )
            )
            return
        self.main_calls.append(messages)
        yield ModelStreamChunk(delta="完成")
        yield ModelStreamChunk(usage=TokenUsage(input_tokens=20, output_tokens=3, total_tokens=23))


def make_memory_client(
    tmp_path: Path, *, fail_summary: bool = False, auto_extract: bool = False
) -> tuple[TestClient, MemoryProvider]:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}",
        default_model_id="memory-model",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
        memory_recent_message_limit=4,
        memory_compact_threshold=8,
        memory_context_token_budget=1_000,
        memory_summary_target_tokens=128,
        memory_auto_extract_enabled=auto_extract,
    )
    provider = MemoryProvider(fail_summary=fail_summary)
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="memory-model",
            name="Memory Model",
            provider="memory-test",
            provider_name="Memory Test",
            capabilities=["chat", "streaming"],
            available=True,
        ),
        provider=provider,
        provider_model_id="provider-memory-model",
    )
    return TestClient(create_app(settings=settings, providers=providers)), provider


def create_session(client: TestClient, title: str = "Memory 会话") -> str:
    response = client.post("/api/v1/sessions", json={"title": title, "model_id": "memory-model"})
    assert response.status_code == 201
    return str(response.json()["id"])


def send(client: TestClient, session_id: str, content: str) -> str:
    response = client.post(
        f"/api/v1/sessions/{session_id}/messages/stream", json={"content": content}
    )
    assert response.status_code == 200
    return cast(str, response.text)


def test_context_builder_enforces_budget_and_keeps_current_message() -> None:
    counter = ConservativeTokenCounter()
    builder = ContextBuilder(
        token_counter=counter,
        token_budget=90,
        recent_message_limit=6,
        summary_target_tokens=20,
    )
    messages = [
        ChatMessage(role="user" if index % 2 == 0 else "assistant", content="内容" * 45)
        for index in range(10)
    ]
    messages[-1] = ChatMessage(role="user", content="必须保留的当前问题")
    bundle = builder.build(
        messages=messages,
        summary="旧会话摘要" * 10,
        summary_version=2,
        memories=[
            MemorySearchResult(
                id="memory-1",
                content="项目使用 Conda 和 pip",
                memory_type="project_fact",
                importance=0.9,
                confidence=1.0,
                score=0.9,
            )
        ],
    )
    assert bundle.messages[-1].content == "必须保留的当前问题"
    assert bundle.stats.estimated_tokens <= bundle.stats.token_budget
    assert bundle.stats.trimmed_message_count > 0
    assert bundle.stats.summary_version == 2


def test_rolling_summary_logically_cools_old_messages(tmp_path: Path) -> None:
    client, provider = make_memory_client(tmp_path)
    with client:
        session_id = create_session(client)
        for index in range(5):
            send(client, session_id, f"第 {index + 1} 轮需求")

        memory = client.get(f"/api/v1/sessions/{session_id}/memory")
        assert memory.status_code == 200
        payload = memory.json()
        assert payload["summary"]["version"] == 1
        assert payload["cold_message_count"] == 5
        assert payload["hot_message_count"] == 5
        assert provider.summary_calls == 1

        history = client.get(f"/api/v1/sessions/{session_id}/messages").json()
        assert len(history) == 10


def test_memory_scope_lifecycle_and_sensitive_filter(tmp_path: Path) -> None:
    client, provider = make_memory_client(tmp_path)
    with client:
        session_a = create_session(client, "会话 A")
        session_b = create_session(client, "会话 B")

        workspace_memory = client.post(
            "/api/v1/memories",
            json={
                "scope_type": "workspace",
                "scope_id": "default",
                "memory_type": "project_fact",
                "content": "项目使用 Conda 和 pip 管理依赖",
                "importance": 0.9,
            },
        )
        assert workspace_memory.status_code == 201
        session_memory = client.post(
            "/api/v1/memories",
            json={
                "scope_type": "session",
                "scope_id": session_a,
                "memory_type": "decision",
                "content": "当前会话选择 Python 实现",
            },
        )
        assert session_memory.status_code == 201
        client.post(
            "/api/v1/memories",
            json={
                "scope_type": "session",
                "scope_id": session_b,
                "memory_type": "decision",
                "content": "另一个会话选择 Java 实现",
            },
        )
        expired = datetime.now(UTC) - timedelta(days=1)
        client.post(
            "/api/v1/memories",
            json={
                "scope_type": "session",
                "scope_id": session_a,
                "memory_type": "temporary",
                "content": "已经过期的临时事实",
                "expires_at": expired.isoformat(),
            },
        )

        send(client, session_a, "这个 Python 项目如何管理依赖？")
        system_prompt = provider.main_calls[-1][0].content
        assert "Conda 和 pip" in system_prompt
        assert "选择 Python" in system_prompt
        assert "选择 Java" not in system_prompt
        assert "已经过期" not in system_prompt

        rejected = client.post(
            "/api/v1/memories",
            json={
                "scope_type": "workspace",
                "scope_id": "default",
                "memory_type": "secret",
                "content": "api_key=sk-abcdefghijklmnopqrstuvwxyz123456",
            },
        )
        assert rejected.status_code == 422

        memory_id = session_memory.json()["id"]
        assert client.delete(f"/api/v1/memories/{memory_id}").status_code == 204
        active = client.get(
            f"/api/v1/memories?scope_type=session&scope_id={session_a}&status=active"
        ).json()
        assert all(item["id"] != memory_id for item in active)


def test_summary_failure_degrades_without_blocking_chat(tmp_path: Path) -> None:
    client, provider = make_memory_client(tmp_path, fail_summary=True)
    with client:
        session_id = create_session(client)
        response_body = ""
        for index in range(5):
            response_body = send(client, session_id, f"第 {index + 1} 轮")
        assert provider.summary_calls == 1
        assert "summary_failed" in response_body
        assert "event: message.completed" in response_body


def test_automatic_extraction_creates_inactive_candidate(tmp_path: Path) -> None:
    client, provider = make_memory_client(tmp_path, auto_extract=True)
    with client:
        session_id = create_session(client)
        send(client, session_id, "请记住这个项目使用 Python 开发")
        candidates = client.get(
            f"/api/v1/memories?scope_type=session&scope_id={session_id}&status=candidate"
        ).json()
        assert provider.extraction_calls == 1
        assert len(candidates) == 1
        assert candidates[0]["content"] == "项目使用 Python 开发"
        assert candidates[0]["status"] == "candidate"
