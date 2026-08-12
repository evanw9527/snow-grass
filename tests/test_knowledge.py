from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from snow_grass.core.config import PROJECT_ROOT, Settings
from snow_grass.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'knowledge.db'}",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
        knowledge_enabled=True,
    )
    return TestClient(create_app(settings=settings))


def ingest(client: TestClient, event_id: str, text: str) -> None:
    response = client.post(
        "/api/v1/activity-events/batch",
        json={
            "events": [
                {
                    "client_event_id": event_id,
                    "app_name": "Visual Studio Code",
                    "bundle_id": "com.microsoft.VSCode",
                    "text": text,
                    "occurred_at": datetime.now(UTC).isoformat(),
                }
            ]
        },
    )
    assert response.status_code == 202


def test_activity_is_indexed_searchable_and_reported_in_stats(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "login-fix", "修复登录问题并补充了回归测试")

        page = client.get("/api/v1/knowledge/documents").json()
        assert page["total"] == 1
        assert page["items"][0]["source_id"] == "login-fix"
        assert page["items"][0]["source_type"] == "activity_event"
        assert page["items"][0]["occurred_at"].endswith("Z")
        assert page["items"][0]["created_at"].endswith("Z")
        assert page["items"][0]["updated_at"].endswith("Z")

        search = client.post("/api/v1/knowledge/search", json={"query": "登录问题", "limit": 5})
        assert search.status_code == 200
        assert search.json()["items"][0]["source_id"] == "login-fix"
        assert search.json()["items"][0]["occurred_at"].endswith("Z")

        stats = client.get("/api/v1/knowledge/stats").json()
        assert stats["globally_enabled"] is True
        assert stats["total"] == 1
        assert stats["active"] == 1
        assert stats["activity_events"] == 1
        assert stats["last_indexed_at"].endswith("Z")


def test_disabled_and_deleted_documents_are_not_retrieved(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "permission-boundary", "知识库权限边界验证")
        document = client.get("/api/v1/knowledge/documents").json()["items"][0]

        disabled = client.patch(
            f"/api/v1/knowledge/documents/{document['id']}",
            json={"status": "disabled"},
        )
        assert disabled.status_code == 200
        assert disabled.json()["status"] == "disabled"
        assert (
            client.post("/api/v1/knowledge/search", json={"query": "权限边界"}).json()["items"]
            == []
        )

        enabled = client.post(
            "/api/v1/knowledge/documents/batch",
            json={"document_ids": [document["id"]], "action": "enable"},
        )
        assert enabled.json()["updated"] == 1
        assert client.delete(f"/api/v1/knowledge/documents/{document['id']}").status_code == 204
        assert (
            client.post("/api/v1/knowledge/search", json={"query": "权限边界"}).json()["items"]
            == []
        )


def test_reindex_preserves_tombstones_unless_restore_is_explicit(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "tombstone", "软删除重建索引验证")
        document = client.get("/api/v1/knowledge/documents").json()["items"][0]
        client.delete(f"/api/v1/knowledge/documents/{document['id']}")

        ordinary = client.post("/api/v1/knowledge/reindex", json={})
        assert ordinary.status_code == 200
        assert ordinary.json()["skipped"] == 1
        deleted = client.get("/api/v1/knowledge/documents", params={"status": "deleted"}).json()
        assert deleted["total"] == 1

        restored = client.post("/api/v1/knowledge/reindex", json={"restore_deleted": True})
        assert restored.json()["updated"] == 1
        active = client.get("/api/v1/knowledge/documents", params={"status": "active"}).json()
        assert active["total"] == 1


def test_knowledge_base_switch_controls_all_documents_in_source(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "base-switch", "知识库级别停用验证")
        bases = client.get("/api/v1/knowledge/bases")
        assert bases.status_code == 200
        activity_base = next(
            item for item in bases.json() if item["source_type"] == "activity_event"
        )
        assert activity_base["enabled"] is True
        assert activity_base["total"] == 1
        assert activity_base["last_indexed_at"].endswith("Z")

        disabled = client.patch("/api/v1/knowledge/bases/activity_event", json={"enabled": False})
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        search = client.post(
            "/api/v1/knowledge/search",
            json={"query": "级别停用", "source_types": ["activity_event"]},
        ).json()
        assert search["items"] == []
        assert search["degraded_reason"] == "knowledge_bases_disabled"

        client.patch("/api/v1/knowledge/bases/activity_event", json={"enabled": True})
        assert client.post("/api/v1/knowledge/search", json={"query": "级别停用"}).json()["items"]


def test_daily_work_query_uses_time_window_and_filters_question_echo(
    tmp_path: Path,
) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "real-work", "完成 Agent 对话滚动与知识库管理优化")
        ingest(client, "question-echo", "今天我做了什么")
        ingest(client, "meta-question", "帮我确认下知识库能不能用")

        response = client.post(
            "/api/v1/knowledge/search", json={"query": "我今天做了什么", "limit": 8}
        )

        assert response.status_code == 200
        source_ids = [item["source_id"] for item in response.json()["items"]]
        assert "real-work" in source_ids
        assert "question-echo" not in source_ids
        assert "meta-question" not in source_ids


def test_daily_work_without_explicit_day_searches_recent_work(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        ingest(client, "recent-work", "调整 FatPet 与 Snow Grass 的知识库检索链路")

        response = client.post("/api/v1/knowledge/search", json={"query": "我的日常工作是什么"})

        assert response.status_code == 200
        assert response.json()["items"][0]["source_id"] == "recent-work"
