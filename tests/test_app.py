from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app
from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelInfo,
    ModelRequestOptions,
    ModelStreamChunk,
    TokenUsage,
)
from snow_grass.providers.registry import ProviderRegistry


class FakeProvider:
    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        assert model_id == "provider-test-model"
        assert messages[0].role == "system"
        assert "Test Model from Test" in messages[0].content
        assert "Never claim to be Claude" in messages[0].content
        assert tools is None
        assert options is None
        yield ModelStreamChunk(delta="测试")
        yield ModelStreamChunk(delta="回答")
        yield ModelStreamChunk(usage=TokenUsage(input_tokens=12, output_tokens=7, total_tokens=19))


def make_client(tmp_path: Path) -> TestClient:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0)
    rollout_path = tmp_path / "rollout.jsonl"
    rollout_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "timestamp": (now - timedelta(days=1)).isoformat(),
                    "type": "turn_context",
                    "payload": {"model": "gpt-test"},
                },
                {
                    "timestamp": (now - timedelta(days=1)).isoformat(),
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 800,
                                "cached_input_tokens": 600,
                                "output_tokens": 150,
                                "reasoning_output_tokens": 50,
                                "total_tokens": 1000,
                            }
                        },
                    },
                },
                {
                    "timestamp": now.isoformat(),
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 950,
                                "cached_input_tokens": 700,
                                "output_tokens": 190,
                                "reasoning_output_tokens": 60,
                                "total_tokens": 1200,
                            }
                        },
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    codex_db = tmp_path / "codex.db"
    connection = sqlite3.connect(codex_db)
    connection.execute(
        "CREATE TABLE threads ("
        "id TEXT, tokens_used INTEGER, model TEXT, cwd TEXT, updated_at INTEGER, "
        "rollout_path TEXT)"
    )
    connection.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
        (
            "thread-test",
            1200,
            "gpt-test",
            "/work/demo",
            int(now.timestamp()),
            str(rollout_path),
        ),
    )
    connection.commit()
    connection.close()
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        default_model_id="test-model",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
        codex_state_db_path=codex_db,
    )
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            provider_name="Test",
            capabilities=["chat", "streaming"],
            available=True,
        ),
        provider=FakeProvider(),
        provider_model_id="provider-test-model",
    )
    return TestClient(create_app(settings=settings, providers=providers))


def test_health_models_and_skills(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        models = client.get("/api/v1/models").json()
        assert models == [
            {
                "id": "test-model",
                "name": "Test Model",
                "provider": "test",
                "provider_name": "Test",
                "capabilities": ["chat", "streaming"],
                "available": True,
            }
        ]

        skills = client.get("/api/v1/skills").json()
        assert skills[0]["id"] == "requirement-analysis"
        disabled = client.post("/api/v1/skills/requirement_analysis/disable").json()
        assert disabled["enabled"] is False
        enabled = client.post("/api/v1/skills/requirement_analysis/enable").json()
        assert enabled["enabled"] is True


def test_session_stream_and_history(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/v1/sessions",
            json={"title": "需求会话", "model_id": "test-model"},
        )
        assert created.status_code == 201
        session_id = created.json()["id"]

        response = client.post(
            f"/api/v1/sessions/{session_id}/messages/stream",
            json={"content": "请分析这个需求并给出方案"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "event: run.started" in body
        assert "event: step.completed" in body
        assert '"skill_id": "requirement-analysis"' in body
        assert body.count("event: message.delta") == 2
        assert "event: message.completed" in body
        assert "event: run.completed" in body

        messages = client.get(f"/api/v1/sessions/{session_id}/messages").json()
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "测试回答"
        assert messages[1]["skill_id"] == "requirement-analysis"
        assert messages[1]["input_tokens"] == 12
        assert messages[1]["output_tokens"] == 7
        assert messages[1]["total_tokens"] == 19

        usage = client.get("/api/v1/usage/summary?days=30").json()
        assert usage == {
            "period_days": 30,
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
            "requests": 1,
            "sessions": 1,
            "by_model": [
                {
                    "model_id": "test-model",
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "total_tokens": 19,
                    "requests": 1,
                }
            ],
        }

        local_usage = client.get("/api/v1/usage/local?days=30").json()
        assert local_usage["total_tokens"] == 1219
        assert local_usage["sessions"] == 2
        assert local_usage["sources"][0] == {
            "id": "codex",
            "name": "Codex",
            "available": True,
            "total_tokens": 1200,
            "sessions": 1,
        }
        assert local_usage["projects"][0]["name"] == "demo"

        connection = sqlite3.connect(tmp_path / "test.db")
        yesterday = (
            (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(days=1))
            .replace(hour=12, minute=0, second=0, microsecond=0)
            .astimezone(UTC)
            .replace(tzinfo=None)
        )
        connection.execute(
            "INSERT INTO chat_messages ("
            "id, session_id, role, content, model_id, skill_id, created_at, "
            "input_tokens, output_tokens, total_tokens"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "historical-assistant-message",
                session_id,
                "assistant",
                "昨日回答",
                "test-model",
                None,
                yesterday.isoformat(sep=" "),
                80,
                20,
                100,
            ),
        )
        connection.commit()
        connection.close()

        today_usage = client.get("/api/v1/usage/local?days=1").json()
        assert today_usage["total_tokens"] == 219
        assert today_usage["sources"][0]["total_tokens"] == 200
        assert today_usage["sources"][1]["total_tokens"] == 19
        assert today_usage["daily"] == [
            {
                "date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
                "total_tokens": 219,
            }
        ]

        rollout_path = tmp_path / "rollout.jsonl"
        with rollout_path.open("a", encoding="utf-8") as rollout:
            rollout.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 990,
                                    "cached_input_tokens": 720,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 60,
                                    "total_tokens": 1250,
                                }
                            },
                        },
                    }
                )
                + "\n"
            )
        refreshed_usage = client.get("/api/v1/usage/local?days=1").json()
        assert refreshed_usage["total_tokens"] == 269
        assert refreshed_usage["sources"][0]["total_tokens"] == 250


def test_catalog_marks_models_unavailable_without_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'catalog.db'}",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
    )
    registry = ProviderRegistry.from_catalog(settings.load_model_catalog())
    assert {model.id for model in registry.list_models()} == {
        "deepseek-chat",
        "glm-4-plus",
    }
    assert all(not model.available for model in registry.list_models())
