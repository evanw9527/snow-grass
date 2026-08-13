from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
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


class _Provider:
    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        yield ModelStreamChunk(
            delta='{"completed":[],"in_progress":[],"blockers":[],"next_steps":[]}'
        )


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'workflow.db'}",
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
        provider=_Provider(),
    )
    return TestClient(create_app(settings=settings, providers=providers))


def test_flow_lifecycle_is_persisted_and_versioned(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        packs = client.get("/api/v1/business-packs")
        assert packs.status_code == 200
        assert {item["business_type"] for item in packs.json()} == {
            "activity_summary",
            "negative_feedback",
        }

        library = client.get("/api/v1/component-library")
        assert library.status_code == 200
        component_versions = library.json()
        assert {item["component_key"] for item in component_versions} == {
            "core.input",
            "core.output",
            "activity.normalize",
            "activity.filter",
            "activity.summarize",
            "activity.validate",
            "feedback.classify",
            "feedback.reply",
        }
        assert all(item["implementation_source"] for item in component_versions)
        assert all(item["source_language"] == "python" for item in component_versions)
        assert all(item["icon"] for item in component_versions)

        workflows = client.get("/api/v1/workflows?business_type=activity_summary").json()
        detail = client.get(f"/api/v1/workflows/{workflows[0]['id']}").json()
        assert detail["validation"]["valid"] is True
        assert all(
            "component_version_id" in node and "type" not in node
            for node in detail["graph"]["nodes"]
        )
        assert all(
            "source_port" in edge and edge["condition"] is None
            for edge in detail["graph"]["edges"]
        )

        saved = client.put(
            f"/api/v1/workflows/{detail['id']}/draft",
            json={"expected_revision": detail["draft_revision"], "graph": detail["graph"]},
        )
        assert saved.status_code == 200
        revision = saved.json()["draft_revision"]
        assert revision == detail["draft_revision"] + 1

        conflict = client.put(
            f"/api/v1/workflows/{detail['id']}/draft",
            json={"expected_revision": detail["draft_revision"], "graph": detail["graph"]},
        )
        assert conflict.status_code == 409

        published = client.post(
            f"/api/v1/workflows/{detail['id']}/publish",
            json={"expected_revision": revision, "activate": True},
        )
        assert published.status_code == 200
        assert published.json()["active"] is True
        assert published.json()["version_number"] == 2


def test_second_business_reuses_kernel_and_trace(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        workflow = client.get("/api/v1/workflows?business_type=negative_feedback").json()[0]
        detail = client.get(f"/api/v1/workflows/{workflow['id']}").json()
        preview = client.post(
            f"/api/v1/workflows/{workflow['id']}/preview",
            json={
                "draft_revision": detail["draft_revision"],
                "input": {"text": "今天这件事让我很烦"},
            },
        )
        assert preview.status_code == 200
        payload = preview.json()
        assert payload["status"] == "succeeded"
        assert payload["output"]["should_respond"] is True
        assert [item["status"] for item in payload["trace"]] == ["succeeded"] * 4

        stored = client.get(f"/api/v1/workflow-runs/{payload['run_id']}")
        assert stored.status_code == 200
        assert len(stored.json()["trace"]) == 4

        page = client.get(
            f"/api/v1/workflow-runs?workflow_id={workflow['id']}&limit=1&offset=0"
        )
        assert page.status_code == 200
        assert page.json()["total"] == 1
        assert len(page.json()["items"]) == 1
        assert page.json()["aggregate"]["node_executions"] == 4
        assert len(page.json()["node_stats"]) == 4
        assert page.json()["items"][0]["started_at"].endswith("Z")


def test_cycle_is_rejected_by_common_validator(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        workflow = client.get("/api/v1/workflows?business_type=negative_feedback").json()[0]
        detail = client.get(f"/api/v1/workflows/{workflow['id']}").json()
        graph = detail["graph"]
        graph["edges"].append({"id": "cycle", "from": "output", "to": "input", "label": "循环"})
        response = client.put(
            f"/api/v1/workflows/{workflow['id']}/draft",
            json={"expected_revision": detail["draft_revision"], "graph": graph},
        )
        assert response.status_code == 422
