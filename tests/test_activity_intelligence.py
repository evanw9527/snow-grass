from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app
from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelInfo,
    ModelRequestOptions,
    ModelStreamChunk,
)
from snow_grass.providers.registry import ProviderRegistry


class FakeProvider:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls = 0

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        self.calls += 1
        yield ModelStreamChunk(delta=self.responses.pop(0))


def make_client(tmp_path: Path, provider: FakeProvider) -> TestClient:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'intelligence.db'}",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
    )
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="deepseek-chat",
            name="DeepSeek",
            provider="test",
            provider_name="Test",
            available=True,
        ),
        provider=provider,
    )
    return TestClient(create_app(settings=settings, providers=providers))


def activity(event_id: str, text: str, occurred_at: datetime | None = None) -> dict[str, str]:
    return {
        "client_event_id": event_id,
        "app_name": "备忘录",
        "bundle_id": "com.apple.Notes",
        "text": text,
        "occurred_at": (occurred_at or datetime.now(UTC)).isoformat(),
    }


def analyze(client: TestClient, event_id: str) -> dict[str, object]:
    response = client.post(
        "/api/v1/activity-events/analyze",
        json={
            "client_event_id": event_id,
            "available_actions": ["idle", "wave", "comfort"],
        },
    )
    assert response.status_code == 200
    return response.json()


def test_clear_local_results_do_not_call_deepseek(tmp_path: Path) -> None:
    provider = FakeProvider()
    with make_client(tmp_path, provider) as client:
        client.post(
            "/api/v1/activity-events/batch",
            json={
                "events": [
                    activity("neutral", "修复登录失败的单元测试"),
                    activity("negative", "我真的很沮丧，这个问题卡了很久"),
                ]
            },
        )

        neutral = analyze(client, "neutral")
        negative = analyze(client, "negative")

        assert neutral["should_respond"] is False
        assert neutral["analysis_source"] == "local"
        assert negative["should_respond"] is True
        assert negative["category"] == "frustrated"
        assert negative["analysis_source"] == "local"
        assert provider.calls == 0


def test_ambiguous_text_uses_deepseek_and_high_risk_uses_fixed_reply(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"should_respond":true,"category":"sad","severity":1,'
            '"reply":"听起来这件事让你不好受。","action":"idle"}'
        ]
    )
    with make_client(tmp_path, provider) as client:
        client.post(
            "/api/v1/activity-events/batch",
            json={
                "events": [
                    activity("ambiguous", "这件事情令人沮丧"),
                    activity("risk", "我不想活了"),
                ]
            },
        )

        ambiguous = analyze(client, "ambiguous")
        risk = analyze(client, "risk")

        assert ambiguous["analysis_source"] == "deepseek"
        assert ambiguous["should_respond"] is True
        assert ambiguous["category"] == "sad"
        assert ambiguous["action"] == "comfort"
        assert risk["category"] == "high_risk"
        assert risk["is_fixed_safety_reply"] is True
        assert risk["analysis_source"] == "rules"
        assert provider.calls == 1

        stats = client.get("/api/v1/activity-events/analysis-stats").json()
        assert stats["analyzed_count"] == 2
        assert stats["deepseek_count"] == 1
        assert stats["rules_count"] == 1
        assert stats["local_count"] == 0
        assert {item["category"] for item in stats["categories"]} == {"sad", "high_risk"}


def test_hourly_summary_is_persisted_and_idempotent(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"title":"分页功能与测试整理","keywords":["分页","测试"],'
            '"completed":["完成分页"],"in_progress":["整理测试"],'
            '"blockers":[],"next_steps":["运行回归测试"]}'
        ]
    )
    start = datetime(2026, 8, 11, 2, 0, tzinfo=UTC)
    with make_client(tmp_path, provider) as client:
        client.post(
            "/api/v1/activity-events/batch",
            json={
                "events": [
                    activity("one", "完成活动记录分页功能", start + timedelta(minutes=5)),
                    activity("two", "正在整理分页测试", start + timedelta(minutes=35)),
                ]
            },
        )
        payload = {"period_start": start.isoformat(), "timezone_offset_minutes": 480}
        first = client.post("/api/v1/activity-events/hourly-summary", json=payload)
        second = client.post("/api/v1/activity-events/hourly-summary", json=payload)

        assert first.status_code == 200
        assert first.json()["status"] == "generated"
        assert first.json()["title"] == "分页功能与测试整理"
        assert first.json()["keywords"] == ["分页", "测试"]
        assert first.json()["completed"] == ["完成分页"]
        assert first.json()["flow_version_id"]
        assert first.json()["flow_run_id"]
        assert first.json()["quality_status"] == "validated"
        assert second.json()["status"] == "already_generated"
        assert provider.calls == 1

        stored = client.get("/api/v1/activity-events/hourly-summaries").json()
        assert len(stored) == 1
        assert stored[0]["status"] == "generated"
        assert stored[0]["title"] == "分页功能与测试整理"
        assert stored[0]["keywords"] == ["分页", "测试"]
        assert stored[0]["period_start"].endswith("Z")
        assert stored[0]["period_end"].endswith("Z")

        knowledge = client.get(
            "/api/v1/knowledge/documents", params={"source_type": "hourly_summary"}
        ).json()
        assert knowledge["total"] == 1
        assert knowledge["items"][0]["title"] == "分页功能与测试整理"
        assert knowledge["items"][0]["keywords"] == ["分页", "测试"]


def test_short_empty_hour_does_not_call_model(tmp_path: Path) -> None:
    provider = FakeProvider()
    start = datetime(2026, 8, 11, 3, 0, tzinfo=UTC)
    with make_client(tmp_path, provider) as client:
        client.post(
            "/api/v1/activity-events/batch",
            json={"events": [activity("short", "好", start + timedelta(minutes=2))]},
        )
        response = client.post(
            "/api/v1/activity-events/hourly-summary",
            json={"period_start": start.isoformat(), "timezone_offset_minutes": 480},
        )
        assert response.json()["status"] == "insufficient_activity"
        assert provider.calls == 0
