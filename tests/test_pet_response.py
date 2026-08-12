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
    TokenUsage,
)
from snow_grass.providers.registry import ProviderRegistry


class PetProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.options: ModelRequestOptions | None = None

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        assert model_id == "provider-pet-model"
        assert tools is None
        assert "只输出 JSON" in messages[0].content
        self.options = options
        yield ModelStreamChunk(delta=self.content)
        yield ModelStreamChunk(usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12))


def make_client(tmp_path: Path, provider: PetProvider | None) -> TestClient:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'pet.db'}",
        default_model_id="pet-model",
        models_config_path=PROJECT_ROOT / "config/models.yaml",
        skills_path=PROJECT_ROOT / "skills",
        codex_state_db_path=tmp_path / "missing-codex.db",
    )
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="pet-model",
            name="Pet Model",
            provider="test",
            provider_name="Test",
            available=provider is not None,
        ),
        provider=provider,
        provider_model_id="provider-pet-model",
    )
    return TestClient(create_app(settings=settings, providers=providers))


def create_session(client: TestClient) -> str:
    response = client.post(
        "/api/v1/sessions",
        json={"title": "桌宠", "model_id": "pet-model"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def test_pet_response_uses_json_output_and_allowed_action(tmp_path: Path) -> None:
    provider = PetProvider('{"reply":"你做得真棒！","action":"like"}')
    with make_client(tmp_path, provider) as client:
        session_id = create_session(client)
        response = client.post(
            "/api/v1/pet/respond",
            json={
                "session_id": session_id,
                "content": "夸夸我",
                "available_actions": ["idle", "like"],
            },
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "你做得真棒！"
    assert response.json()["action"] == "like"
    assert response.json()["usage"]["total_tokens"] == 12
    assert provider.options == ModelRequestOptions(response_format="json_object")


def test_pet_response_replaces_unavailable_action(tmp_path: Path) -> None:
    provider = PetProvider('{"reply":"跳起来！","action":"dance"}')
    with make_client(tmp_path, provider) as client:
        session_id = create_session(client)
        response = client.post(
            "/api/v1/pet/respond",
            json={
                "session_id": session_id,
                "content": "跳舞",
                "available_actions": ["idle", "wave"],
            },
        )

    assert response.status_code == 200
    assert response.json()["action"] == "idle"


def test_pet_response_falls_back_for_invalid_json(tmp_path: Path) -> None:
    provider = PetProvider("not-json")
    with make_client(tmp_path, provider) as client:
        session_id = create_session(client)
        response = client.post(
            "/api/v1/pet/respond",
            json={
                "session_id": session_id,
                "content": "你好",
                "available_actions": ["idle"],
            },
        )

    assert response.status_code == 200
    assert response.json()["action"] == "idle"
    assert response.json()["reply"]


def test_pet_response_reports_missing_provider_key(tmp_path: Path) -> None:
    with make_client(tmp_path, None) as client:
        session_id = create_session(client)
        response = client.post(
            "/api/v1/pet/respond",
            json={
                "session_id": session_id,
                "content": "你好",
                "available_actions": ["idle"],
            },
        )

    assert response.status_code == 503
    assert "provider key is missing" in response.json()["detail"]
