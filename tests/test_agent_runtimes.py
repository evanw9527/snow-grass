from __future__ import annotations

import sqlite3
from pathlib import Path

from test_app import make_client

from snow_grass.persistence.database import Database


def test_runtime_catalog_and_native_session_default(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        runtimes = client.get("/api/v1/agent-runtimes")
        assert runtimes.status_code == 200
        assert [runtime["id"] for runtime in runtimes.json()] == [
            "native",
            "openai-agents",
        ]
        assert runtimes.json()[0]["available"] is True

        created = client.post(
            "/api/v1/sessions",
            json={"title": "Native", "model_id": "test-model"},
        )
        assert created.status_code == 201
        assert created.json()["runtime_id"] == "native"
        assert client.get("/api/v1/sessions").json()[0]["runtime_id"] == "native"


def test_framework_failure_is_attributed_without_native_fallback(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/v1/sessions",
            json={
                "title": "Framework",
                "model_id": "test-model",
                "skill_id": "requirement-analysis",
                "runtime_id": "openai-agents",
            },
        )
        assert created.status_code == 201
        session_id = created.json()["id"]
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages/stream",
            json={"content": "请分析这个需求"},
        )
        assert response.status_code == 200
        assert '"runtime_id": "openai-agents"' in response.text
        assert '"code": "framework_runtime_error"' in response.text
        assert "测试回答" not in response.text

    connection = sqlite3.connect(tmp_path / "test.db")
    run = connection.execute(
        "SELECT runtime_id, status, error_code FROM agent_runs"
    ).fetchone()
    connection.close()
    assert run == ("openai-agents", "failed", "framework_runtime_error")


def test_sqlite_migration_defaults_existing_sessions_to_native(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE chat_sessions ("
        "id VARCHAR(36) PRIMARY KEY, title VARCHAR(200) NOT NULL, "
        "model_id VARCHAR(120) NOT NULL, skill_id VARCHAR(120), "
        "knowledge_enabled BOOLEAN NOT NULL DEFAULT 0, "
        "created_at DATETIME, updated_at DATETIME)"
    )
    connection.execute(
        "INSERT INTO chat_sessions "
        "(id,title,model_id,skill_id,knowledge_enabled) VALUES "
        "('legacy','Legacy','test-model',NULL,0)"
    )
    connection.commit()
    connection.close()

    database = Database(f"sqlite+aiosqlite:///{path}")

    async def migrate() -> None:
        await database.create_schema()
        await database.dispose()

    import asyncio

    asyncio.run(migrate())
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT runtime_id FROM chat_sessions WHERE id='legacy'"
    ).fetchone()
    connection.close()
    assert row == ("native",)
