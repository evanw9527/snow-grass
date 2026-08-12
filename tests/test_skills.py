from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from fastapi import FastAPI
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


class SkillTestProvider:
    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        if options == ModelRequestOptions(response_format="json_object"):
            assert tools is None
            selected = "weather-query" if "天气" in messages[-1].content else None
            yield ModelStreamChunk(
                delta=json.dumps(
                    {
                        "selected_skill_id": selected,
                        "confidence": 0.95 if selected else 0.1,
                        "reason": "测试路由",
                    },
                    ensure_ascii=False,
                )
            )
            return
        assert tools is None
        assert options is None
        yield ModelStreamChunk(delta="ok")


def make_skill_app(tmp_path: Path) -> FastAPI:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'skills.db'}",
        default_model_id="test-model",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
    )
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            provider_name="Test",
            capabilities=["chat"],
            available=True,
        ),
        provider=SkillTestProvider(),
    )
    return create_app(settings=settings, providers=providers)


def draft_content(
    *,
    instructions: str = "你负责把需求整理成清晰的验收标准。",
    version: str = "1.0.0",
    tools: list[str] | None = None,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "id": "acceptance-helper",
        "version": version,
        "enabled": True,
        "selection": {"mode": "auto", "keywords": ["验收标准"]},
        "models": {"allowed": ["test-model"]},
        "tools": {"allowed": tools or []},
        "limits": {"max_steps": 6, "timeout_seconds": 90},
    }
    tests = [
        {
            "input": "请帮我编写验收标准",
            "model_id": "test-model",
            "expected_selected": True,
        }
    ]
    return {
        "files": {
            "manifest.yaml": yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            "SKILL.md": (
                "---\n"
                "name: acceptance-helper\n"
                "description: 整理验收标准\n"
                "---\n\n"
                f"# 验收标准助手\n\n{instructions}\n"
            ),
            "agents/openai.yaml": (
                "interface:\n"
                "  display_name: 验收标准助手\n"
                "  short_description: 整理验收标准\n"
                "  default_prompt: 请整理验收标准。\n"
            ),
            "references/format.md": "使用 Given / When / Then 组织验收标准。\n",
            "scripts/check_example.py": "print('example')\n",
            "tests/selection.yaml": yaml.safe_dump(tests, allow_unicode=True, sort_keys=False),
        }
    }


def create_saved_skill(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/v1/admin/skills",
        json={
            "id": "acceptance-helper",
            "name": "验收标准助手",
            "description": "整理验收标准",
        },
    )
    assert response.status_code == 201
    response = client.put(
        "/api/v1/admin/skills/acceptance-helper/draft",
        json={"expected_revision": 1, "content": draft_content()},
    )
    assert response.status_code == 200
    return cast(dict[str, Any], response.json())


def test_skill_draft_validate_publish_and_runtime(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        catalog = client.get("/api/v1/admin/skills").json()
        assert any(item["source"] == "builtin" for item in catalog)

        draft = create_saved_skill(client)
        assert draft["revision"] == 2
        assert {"manifest.yaml", "SKILL.md", "scripts/check_example.py"} <= set(
            draft["content"]["files"]
        )

        stale = client.put(
            "/api/v1/admin/skills/acceptance-helper/draft",
            json={"expected_revision": 1, "content": draft_content()},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["current_revision"] == 2

        validation = client.post(
            "/api/v1/admin/skills/acceptance-helper/validate",
            json={"expected_revision": 2},
        )
        assert validation.status_code == 200
        assert validation.json()["valid"] is True

        published = client.post(
            "/api/v1/admin/skills/acceptance-helper/publish",
            json={"version": "1.0.0", "expected_revision": 2, "activate": True},
        )
        assert published.status_code == 200
        assert published.json()["active"] is True
        runtime = client.get("/api/v1/skills").json()
        assert any(item["id"] == "acceptance-helper" for item in runtime)
        session_id = client.post(
            "/api/v1/sessions",
            json={"title": "Skill selection", "model_id": "test-model"},
        ).json()["id"]
        streamed = client.post(
            f"/api/v1/sessions/{session_id}/messages/stream",
            json={"content": "请帮我编写验收标准"},
        )
        assert '"skill_id": "acceptance-helper"' in streamed.text


def test_skill_release_rollback_clone_and_archive(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        create_saved_skill(client)
        client.post(
            "/api/v1/admin/skills/acceptance-helper/validate",
            json={"expected_revision": 2},
        )
        first = client.post(
            "/api/v1/admin/skills/acceptance-helper/publish",
            json={"version": "1.0.0", "expected_revision": 2, "activate": True},
        ).json()

        saved = client.put(
            "/api/v1/admin/skills/acceptance-helper/draft",
            json={
                "expected_revision": 2,
                "content": draft_content(
                    instructions="你负责输出可测试的 Given/When/Then 验收标准。",
                    version="1.1.0",
                ),
            },
        ).json()
        client.post(
            "/api/v1/admin/skills/acceptance-helper/validate",
            json={"expected_revision": saved["revision"]},
        )
        client.post(
            "/api/v1/admin/skills/acceptance-helper/publish",
            json={"version": "1.1.0", "expected_revision": 3, "activate": True},
        )
        rollback = client.post(
            f"/api/v1/admin/skills/acceptance-helper/releases/{first['id']}/activate"
        )
        assert rollback.status_code == 200
        assert rollback.json()["active_version"] == "1.0.0"
        detail = client.get("/api/v1/admin/skills/acceptance-helper").json()
        assert any(item["action"] == "rollback" for item in detail["audits"])

        cloned = client.post(
            "/api/v1/admin/skills/requirement-analysis/clone",
            json={"new_id": "requirements-copy", "new_name": "需求分析副本"},
        )
        assert cloned.status_code == 201
        assert cloned.json()["source"] == "managed"

        readonly = client.put(
            "/api/v1/admin/skills/requirement-analysis/draft",
            json={"expected_revision": 1, "content": draft_content()},
        )
        assert readonly.status_code == 422

        archived = client.post(
            "/api/v1/admin/skills/acceptance-helper/archive",
            json={"expected_revision": 3},
        )
        assert archived.status_code == 200
        assert archived.json()["lifecycle_status"] == "archived"
        assert not any(
            item["id"] == "acceptance-helper" for item in client.get("/api/v1/skills").json()
        )


def test_skill_enabled_state_survives_restart(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        disabled = client.post("/api/v1/skills/requirement_analysis/disable").json()
        assert disabled["enabled"] is False

    with TestClient(make_skill_app(tmp_path)) as client:
        skills = client.get("/api/v1/skills").json()
        requirement = next(item for item in skills if item["id"] == "requirement-analysis")
        assert requirement["enabled"] is False


def test_skill_validation_rejects_unsafe_or_unknown_capabilities(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        create_saved_skill(client)
        unsafe = draft_content(
            instructions="API_KEY=abcdefghijklmnopqrstuvwxyz123456", tools=["shell"]
        )
        saved = client.put(
            "/api/v1/admin/skills/acceptance-helper/draft",
            json={"expected_revision": 2, "content": unsafe},
        ).json()
        report = client.post(
            "/api/v1/admin/skills/acceptance-helper/validate",
            json={"expected_revision": saved["revision"]},
        ).json()
        assert report["valid"] is False
        assert {issue["code"] for issue in report["issues"]} >= {
            "sensitive_content",
            "unknown_tool",
        }


def test_skill_package_rejects_unsafe_paths(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        created = client.post(
            "/api/v1/admin/skills",
            json={
                "id": "safe-package",
                "name": "安全技能包",
                "description": "验证技能包路径",
            },
        ).json()
        content = created["draft"]["content"]
        content["files"]["../outside.py"] = "print('unsafe')"
        response = client.put(
            "/api/v1/admin/skills/safe-package/draft",
            json={"expected_revision": 1, "content": content},
        )
        assert response.status_code == 422
        assert "Unsafe package path" in response.json()["detail"]


def test_auto_selection_uses_metadata_without_keywords(tmp_path: Path) -> None:
    with TestClient(make_skill_app(tmp_path)) as client:
        created = client.post(
            "/api/v1/admin/skills",
            json={
                "id": "weather-query",
                "name": "天气查询",
                "description": "查询城市天气",
            },
        ).json()
        content = created["draft"]["content"]
        manifest = yaml.safe_load(content["files"]["manifest.yaml"])
        manifest["version"] = "1.0.0"
        manifest["selection"]["keywords"] = []
        content["files"]["manifest.yaml"] = yaml.safe_dump(
            manifest, allow_unicode=True, sort_keys=False
        )
        saved = client.put(
            "/api/v1/admin/skills/weather-query/draft",
            json={"expected_revision": 1, "content": content},
        ).json()
        report = client.post(
            "/api/v1/admin/skills/weather-query/validate",
            json={"expected_revision": saved["revision"]},
        ).json()
        assert report["valid"] is True
        published = client.post(
            "/api/v1/admin/skills/weather-query/publish",
            json={"version": "1.0.0", "expected_revision": 2, "activate": True},
        )
        assert published.status_code == 200
        session_id = client.post(
            "/api/v1/sessions",
            json={"title": "Metadata selection", "model_id": "test-model"},
        ).json()["id"]
        streamed = client.post(
            f"/api/v1/sessions/{session_id}/messages/stream",
            json={"content": "北京明天天气怎么样"},
        )
        assert '"skill_id": "weather-query"' in streamed.text
