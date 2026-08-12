from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'activity.db'}",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
        activity_retention_days=30,
    )
    return TestClient(create_app(settings=settings))


def event(event_id: str, text: str, occurred_at: datetime | None = None) -> dict[str, str]:
    return {
        "client_event_id": event_id,
        "app_name": "备忘录",
        "bundle_id": "com.apple.Notes",
        "text": text,
        "occurred_at": (occurred_at or datetime.now(UTC)).isoformat(),
    }


def test_batch_is_idempotent_and_does_not_accept_window_title(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        first = client.post("/api/v1/activity-events/batch", json={"events": [event("e1", "你好")]})
        assert first.status_code == 202
        assert first.json() == {"accepted": 1, "duplicate": 0}
        second = client.post(
            "/api/v1/activity-events/batch", json={"events": [event("e1", "你好")]}
        )
        assert second.json() == {"accepted": 0, "duplicate": 1}
        stored = client.get("/api/v1/activity-events").json()
        assert stored[0]["text"] == "你好"
        assert "window_title" not in stored[0]


def test_ingest_purges_events_older_than_retention(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        old = datetime.now(UTC) - timedelta(days=31)
        client.post("/api/v1/activity-events/batch", json={"events": [event("old", "过期", old)]})
        client.post("/api/v1/activity-events/batch", json={"events": [event("new", "保留")]})
        stored = client.get("/api/v1/activity-events").json()
        assert [item["client_event_id"] for item in stored] == ["new"]


def test_credentials_are_redacted_and_events_can_be_deleted(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        client.post(
            "/api/v1/activity-events/batch",
            json={"events": [event("secret", "api_key=sk-abcdefghijklmnop1234")]},
        )
        stored = client.get("/api/v1/activity-events").json()
        assert stored[0]["text"] == "[REDACTED]"
        response = client.delete(
            "/api/v1/activity-events",
            params={"before": (datetime.now(UTC) + timedelta(seconds=1)).isoformat()},
        )
        assert response.json() == {"deleted": 1}
        assert client.get("/api/v1/activity-events").json() == []


def test_activity_page_filters_and_paginates_without_breaking_recent(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        now = datetime.now(UTC)
        payload = {
            "events": [
                event("notes-1", "项目计划", now - timedelta(minutes=3)),
                {
                    **event("code-1", "实现分页", now - timedelta(minutes=2)),
                    "app_name": "Visual Studio Code",
                    "bundle_id": "com.microsoft.VSCode",
                },
                {
                    **event("code-2", "修复分页", now - timedelta(minutes=1)),
                    "app_name": "Visual Studio Code",
                    "bundle_id": "com.microsoft.VSCode",
                },
            ]
        }
        assert client.post("/api/v1/activity-events/batch", json=payload).status_code == 202

        first_page = client.get(
            "/api/v1/activity-events/page",
            params={"query": "分页", "page": 1, "page_size": 1},
        )
        assert first_page.status_code == 200
        assert first_page.json()["total"] == 2
        assert first_page.json()["items"][0]["client_event_id"] == "code-2"
        assert first_page.json()["items"][0]["occurred_at"].endswith("Z")

        second_page = client.get(
            "/api/v1/activity-events/page",
            params={
                "bundle_id": "com.microsoft.VSCode",
                "page": 2,
                "page_size": 1,
            },
        ).json()
        assert second_page["total"] == 2
        assert second_page["items"][0]["client_event_id"] == "code-1"
        assert len(client.get("/api/v1/activity-events", params={"limit": 2}).json()) == 2


def test_activity_single_and_range_delete_are_scoped(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        now = datetime.now(UTC)
        client.post(
            "/api/v1/activity-events/batch",
            json={
                "events": [
                    event("early", "较早", now - timedelta(hours=2)),
                    event("middle", "中间", now - timedelta(hours=1)),
                    event("late", "最近", now),
                ]
            },
        )
        items = client.get("/api/v1/activity-events/page").json()["items"]
        middle_id = next(item["id"] for item in items if item["client_event_id"] == "middle")
        assert client.delete(f"/api/v1/activity-events/{middle_id}").json() == {"deleted": 1}
        assert client.delete(f"/api/v1/activity-events/{middle_id}").json() == {"deleted": 0}

        response = client.delete(
            "/api/v1/activity-events/range",
            params={
                "start": (now - timedelta(hours=3)).isoformat(),
                "end": (now - timedelta(hours=1, minutes=30)).isoformat(),
            },
        )
        assert response.json() == {"deleted": 1}
        remaining = client.get("/api/v1/activity-events").json()
        assert [item["client_event_id"] for item in remaining] == ["late"]

        invalid = client.get(
            "/api/v1/activity-events/page",
            params={"start": now.isoformat(), "end": (now - timedelta(days=1)).isoformat()},
        )
        assert invalid.status_code == 422


def test_activity_summary_deduplicates_exact_text_without_dropping_english(
    tmp_path: Path,
) -> None:
    with make_client(tmp_path) as client:
        base = datetime(2026, 8, 10, 15, 59, 58, tzinfo=UTC)
        payload = {
            "events": [
                event("same-1", "  Hello   world  ", base),
                event("same-2", "Hello world", base + timedelta(seconds=3)),
                event("case", "hello world", base + timedelta(seconds=4)),
                event("later", "Hello world", base + timedelta(seconds=8)),
                {
                    **event("other-app", "Hello world", base + timedelta(seconds=2)),
                    "app_name": "文本编辑",
                    "bundle_id": "com.apple.TextEdit",
                },
                event("next-day", "中文输入", base + timedelta(seconds=5)),
            ]
        }
        assert client.post("/api/v1/activity-events/batch", json=payload).status_code == 202

        summary = client.get(
            "/api/v1/activity-events/summary",
            params={"timezone_offset_minutes": 480},
        )
        assert summary.status_code == 200
        body = summary.json()
        assert body["raw_event_count"] == 6
        assert body["unique_event_count"] == 5
        assert body["duplicate_event_count"] == 1
        assert body["active_app_count"] == 2
        assert [item["date"] for item in body["daily"]] == ["2026-08-10", "2026-08-11"]
        assert sum(item["event_count"] for item in body["daily"]) == 5
        assert body["apps"][0]["bundle_id"] == "com.apple.Notes"

        deduplicated = client.get(
            "/api/v1/activity-events/page",
            params={"deduplicate": True, "page_size": 20},
        ).json()
        assert deduplicated["total"] == 5
        assert deduplicated["raw_total"] == 6
        assert deduplicated["duplicate_total"] == 1
        assert deduplicated["deduplicated"] is True
        assert {item["client_event_id"] for item in deduplicated["items"]} == {
            "same-1",
            "case",
            "later",
            "other-app",
            "next-day",
        }

        raw = client.get(
            "/api/v1/activity-events/page",
            params={"deduplicate": False, "page_size": 20},
        ).json()
        assert raw["total"] == 6
        assert raw["raw_total"] == 6
        assert raw["duplicate_total"] == 1


def test_activity_summary_respects_filters(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        now = datetime.now(UTC)
        client.post(
            "/api/v1/activity-events/batch",
            json={
                "events": [
                    event("notes", "项目总结", now - timedelta(minutes=2)),
                    {
                        **event("code", "项目代码", now - timedelta(minutes=1)),
                        "app_name": "Visual Studio Code",
                        "bundle_id": "com.microsoft.VSCode",
                    },
                    event("other", "购物清单", now),
                ]
            },
        )

        body = client.get(
            "/api/v1/activity-events/summary",
            params={
                "query": "项目",
                "bundle_id": "com.apple.Notes",
                "start": (now - timedelta(minutes=3)).isoformat(),
                "end": now.isoformat(),
            },
        ).json()
        assert body["raw_event_count"] == 1
        assert body["unique_event_count"] == 1
        assert body["apps"][0]["bundle_id"] == "com.apple.Notes"


def test_activity_domain_has_no_chat_memory_agent_or_pet_imports() -> None:
    domain = PROJECT_ROOT / "src/snow_grass/activity"
    forbidden = {"snow_grass.agent", "snow_grass.api", "snow_grass.memory", "snow_grass.pet"}
    for path in domain.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(any(module.startswith(prefix) for prefix in forbidden) for module in imports)
