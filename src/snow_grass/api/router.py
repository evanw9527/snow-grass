from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import NoReturn, cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from snow_grass.agent.runner import AgentRunner
from snow_grass.api.schemas import (
    AdminSkillSummaryResponse,
    ArchiveSkillRequest,
    ArchiveSkillResponse,
    CloneSkillRequest,
    CompactMemoryRequest,
    CompactMemoryResponse,
    CreateMemoryRequest,
    CreateSessionRequest,
    CreateSkillRequest,
    LocalUsageResponse,
    MemoryItemResponse,
    MessageResponse,
    PublishSkillRequest,
    SaveSkillDraftRequest,
    SessionMemoryResponse,
    SessionResponse,
    SetSkillEnabledRequest,
    SkillActivationResponse,
    SkillDetailResponse,
    SkillDraftResponse,
    SkillReleaseResponse,
    StreamMessageRequest,
    UpdateMemoryRequest,
    UpdateSessionKnowledgeRequest,
    UsageSummaryResponse,
    ValidateSkillRequest,
)
from snow_grass.core.config import Settings
from snow_grass.local_usage import CodexUsageService
from snow_grass.memory.security import SensitiveMemoryError
from snow_grass.memory.service import MemoryService
from snow_grass.persistence.repository import ChatRepository
from snow_grass.providers.base import ModelInfo
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.registry import SkillError, SkillRegistry
from snow_grass.skills.schema import SkillInfo
from snow_grass.skills.service import (
    SkillConflictError,
    SkillNotFoundError,
    SkillService,
    SkillServiceError,
)

api_router = APIRouter(prefix="/api/v1")


def _state(request: Request, name: str) -> object:
    return getattr(request.app.state, name)


@api_router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    registry = cast(ProviderRegistry, _state(request, "providers"))
    return registry.list_models()


@api_router.get("/skills", response_model=list[SkillInfo])
async def list_skills(request: Request) -> list[SkillInfo]:
    registry = cast(SkillRegistry, _state(request, "skills"))
    return registry.list_skills()


def _skill_service(request: Request) -> SkillService:
    settings = cast(Settings, _state(request, "settings"))
    if not settings.skill_admin_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Skill management is disabled"
        )
    return cast(SkillService, _state(request, "skill_service"))


def _raise_skill_error(exc: SkillServiceError) -> NoReturn:
    if isinstance(exc, SkillNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, SkillConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "current_revision": exc.current_revision},
        ) from exc
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


async def _set_skill_enabled(request: Request, skill_id: str, enabled: bool) -> SkillInfo:
    service = cast(SkillService, _state(request, "skill_service"))
    try:
        return SkillInfo.model_validate(await service.set_enabled(skill_id, enabled))
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post("/skills/{skill_id}/enable", response_model=SkillInfo)
async def enable_skill(request: Request, skill_id: str) -> SkillInfo:
    return await _set_skill_enabled(request, skill_id, True)


@api_router.post("/skills/{skill_id}/disable", response_model=SkillInfo)
async def disable_skill(request: Request, skill_id: str) -> SkillInfo:
    return await _set_skill_enabled(request, skill_id, False)


@api_router.get("/admin/skills", response_model=list[AdminSkillSummaryResponse])
async def list_admin_skills(
    request: Request,
    query: str = Query(default="", max_length=200),
    source: str | None = Query(default=None, pattern="^(builtin|managed)$"),
    lifecycle_status: str | None = Query(default=None, alias="status"),
) -> list[dict[str, object]]:
    return await _skill_service(request).list_admin(
        query=query, source=source, status=lifecycle_status
    )


@api_router.post(
    "/admin/skills",
    response_model=SkillDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_admin_skill(request: Request, payload: CreateSkillRequest) -> dict[str, object]:
    try:
        return await _skill_service(request).create(
            skill_id=payload.id, name=payload.name, description=payload.description
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.get("/admin/skills/{skill_id}", response_model=SkillDetailResponse)
async def get_admin_skill(request: Request, skill_id: str) -> dict[str, object]:
    try:
        return await _skill_service(request).get_detail(skill_id)
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.put("/admin/skills/{skill_id}/draft", response_model=SkillDraftResponse)
async def save_admin_skill_draft(
    request: Request, skill_id: str, payload: SaveSkillDraftRequest
) -> dict[str, object]:
    try:
        return await _skill_service(request).save_draft(
            skill_id=skill_id,
            expected_revision=payload.expected_revision,
            content=payload.content,
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post("/admin/skills/{skill_id}/validate")
async def validate_admin_skill(
    request: Request, skill_id: str, payload: ValidateSkillRequest
) -> object:
    try:
        return await _skill_service(request).validate_draft(
            skill_id=skill_id, expected_revision=payload.expected_revision
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post("/admin/skills/{skill_id}/publish", response_model=SkillReleaseResponse)
async def publish_admin_skill(
    request: Request, skill_id: str, payload: PublishSkillRequest
) -> dict[str, object]:
    try:
        return await _skill_service(request).publish(
            skill_id=skill_id,
            version=payload.version,
            expected_revision=payload.expected_revision,
            activate=payload.activate,
            release_notes=payload.release_notes,
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.get("/admin/skills/{skill_id}/releases", response_model=list[SkillReleaseResponse])
async def list_admin_skill_releases(request: Request, skill_id: str) -> list[object]:
    try:
        detail = await _skill_service(request).get_detail(skill_id)
        return cast(list[object], detail["releases"])
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post(
    "/admin/skills/{skill_id}/releases/{release_id}/activate",
    response_model=SkillActivationResponse,
)
async def activate_admin_skill_release(
    request: Request, skill_id: str, release_id: str
) -> dict[str, object]:
    try:
        return await _skill_service(request).activate_release(
            skill_id=skill_id, release_id=release_id
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.put("/admin/skills/{skill_id}/enabled", response_model=SkillInfo)
async def set_admin_skill_enabled(
    request: Request, skill_id: str, payload: SetSkillEnabledRequest
) -> SkillInfo:
    return await _set_skill_enabled(request, skill_id, payload.enabled)


@api_router.post(
    "/admin/skills/{skill_id}/clone",
    response_model=SkillDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def clone_admin_skill(
    request: Request, skill_id: str, payload: CloneSkillRequest
) -> dict[str, object]:
    try:
        return await _skill_service(request).clone(
            skill_id=skill_id,
            new_id=payload.new_id,
            new_name=payload.new_name,
            release_id=payload.release_id,
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post("/admin/skills/{skill_id}/archive", response_model=ArchiveSkillResponse)
async def archive_admin_skill(
    request: Request, skill_id: str, payload: ArchiveSkillRequest
) -> dict[str, object]:
    try:
        return await _skill_service(request).archive(
            skill_id=skill_id, expected_revision=payload.expected_revision
        )
    except SkillServiceError as exc:
        _raise_skill_error(exc)


@api_router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: Request, payload: CreateSessionRequest) -> object:
    settings = cast(Settings, _state(request, "settings"))
    providers = cast(ProviderRegistry, _state(request, "providers"))
    repository = cast(ChatRepository, _state(request, "repository"))
    model_id = payload.model_id or settings.default_model_id
    if not providers.has_model(model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown model: {model_id}"
        )
    if payload.skill_id:
        skills = cast(SkillRegistry, _state(request, "skills"))
        try:
            skills.select(content="", model_id=model_id, explicit_skill_id=payload.skill_id)
        except SkillError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return await repository.create_session(
        title=payload.title,
        model_id=model_id,
        skill_id=payload.skill_id,
        knowledge_enabled=payload.knowledge_enabled,
    )


@api_router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(request: Request, limit: int = 50) -> list[object]:
    repository = cast(ChatRepository, _state(request, "repository"))
    return list(await repository.list_sessions(limit=max(1, min(limit, 100))))


@api_router.patch(
    "/sessions/{session_id}/knowledge", response_model=SessionResponse
)
async def update_session_knowledge(
    request: Request, session_id: str, payload: UpdateSessionKnowledgeRequest
) -> object:
    repository = cast(ChatRepository, _state(request, "repository"))
    record = await repository.update_session_knowledge(session_id, payload.enabled)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return record


@api_router.get("/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(request: Request, days: int = Query(default=30, ge=1, le=365)) -> object:
    repository = cast(ChatRepository, _state(request, "repository"))
    return await repository.usage_summary(days=days)


@api_router.get("/usage/local", response_model=LocalUsageResponse)
async def local_usage(request: Request, days: int = Query(default=30, ge=1, le=365)) -> object:
    repository = cast(ChatRepository, _state(request, "repository"))
    codex_usage = cast(CodexUsageService, _state(request, "codex_usage"))
    snow_grass = await repository.usage_summary(days=days)
    codex = await codex_usage.summary(days=days)
    codex_tokens = codex.total_tokens if codex else 0
    codex_sessions = codex.sessions if codex else 0

    sources: list[dict[str, object]] = [
        {
            "id": "codex",
            "name": "Codex",
            "available": codex is not None,
            "total_tokens": codex_tokens,
            "sessions": codex_sessions,
        },
        {
            "id": "snow-grass",
            "name": "Snow Grass",
            "available": True,
            "total_tokens": snow_grass.total_tokens,
            "sessions": snow_grass.sessions,
        },
    ]
    models = [] if codex is None else [{"source": "Codex", **item} for item in codex.models]
    models.extend(
        {
            "source": "Snow Grass",
            "name": item.model_id,
            "total_tokens": item.total_tokens,
            "sessions": item.requests,
        }
        for item in snow_grass.by_model
    )
    daily: dict[str, int] = defaultdict(int)
    if codex is not None:
        for item in codex.daily:
            daily[str(item["date"])] += int(item["total_tokens"])
    for item in snow_grass.daily:
        daily[str(item["date"])] += int(item["total_tokens"])
    return {
        "period_days": days,
        "total_tokens": codex_tokens + snow_grass.total_tokens,
        "sessions": codex_sessions + snow_grass.sessions,
        "sources": sources,
        "daily": [{"date": day, "total_tokens": tokens} for day, tokens in sorted(daily.items())],
        "models": sorted(models, key=lambda item: int(item["total_tokens"]), reverse=True),
        "projects": codex.projects if codex else [],
        "updated_at": datetime.now(UTC),
    }


@api_router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def list_messages(request: Request, session_id: str) -> list[object]:
    repository = cast(ChatRepository, _state(request, "repository"))
    if await repository.get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return list(await repository.list_messages(session_id))


@api_router.get("/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
async def get_session_memory(request: Request, session_id: str) -> dict[str, object]:
    repository = cast(ChatRepository, _state(request, "repository"))
    if await repository.get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    summary = await repository.get_current_summary(session_id)
    hot_count, cold_count = await repository.memory_counts(session_id)
    last_context = await repository.get_last_context_run(session_id)
    return {
        "summary": summary,
        "hot_message_count": hot_count,
        "cold_message_count": cold_count,
        "last_context_stats": last_context,
    }


@api_router.post("/sessions/{session_id}/memory/compact", response_model=CompactMemoryResponse)
async def compact_session_memory(
    request: Request, session_id: str, payload: CompactMemoryRequest
) -> object:
    repository = cast(ChatRepository, _state(request, "repository"))
    memory = cast(MemoryService, _state(request, "memory"))
    chat_session = await repository.get_session(session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return await memory.compact(
        session_id=session_id, model_id=chat_session.model_id, force=payload.force
    )


@api_router.get("/memories", response_model=list[MemoryItemResponse])
async def list_memories(
    request: Request,
    scope_type: str,
    scope_id: str,
    memory_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=200),
) -> list[object]:
    if scope_type not in {"session", "workspace"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="scope_type must be session or workspace",
        )
    repository = cast(ChatRepository, _state(request, "repository"))
    return list(
        await repository.list_memories(
            scope_type=scope_type,
            scope_id=scope_id,
            status=memory_status,
            limit=limit,
        )
    )


@api_router.post(
    "/memories", response_model=MemoryItemResponse, status_code=status.HTTP_201_CREATED
)
async def create_memory(request: Request, payload: CreateMemoryRequest) -> object:
    memory = cast(MemoryService, _state(request, "memory"))
    repository = cast(ChatRepository, _state(request, "repository"))
    if payload.scope_type == "session" and await repository.get_session(payload.scope_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    try:
        return await memory.create_memory(
            scope_type=payload.scope_type,
            scope_id=payload.scope_id,
            memory_type=payload.memory_type,
            content=payload.content,
            data=payload.data,
            importance=payload.importance,
            expires_at=payload.expires_at,
        )
    except (SensitiveMemoryError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc


@api_router.patch("/memories/{memory_id}", response_model=MemoryItemResponse)
async def update_memory(request: Request, memory_id: str, payload: UpdateMemoryRequest) -> object:
    memory = cast(MemoryService, _state(request, "memory"))
    try:
        record = await memory.update_memory(memory_id, payload.model_dump(exclude_unset=True))
    except SensitiveMemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return record


@api_router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(request: Request, memory_id: str) -> None:
    repository = cast(ChatRepository, _state(request, "repository"))
    if not await repository.delete_memory(memory_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")


@api_router.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    request: Request, session_id: str, payload: StreamMessageRequest
) -> StreamingResponse:
    settings = cast(Settings, _state(request, "settings"))
    repository = cast(ChatRepository, _state(request, "repository"))
    providers = cast(ProviderRegistry, _state(request, "providers"))
    runner = cast(AgentRunner, _state(request, "runner"))
    chat_session = await repository.get_session(session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    model_id = payload.model_id or chat_session.model_id
    if not providers.has_model(model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown model: {model_id}"
        )
    skill_id = payload.skill_id if payload.skill_id is not None else chat_session.skill_id
    await repository.add_message(
        session_id=session_id,
        role="user",
        content=payload.content,
        model_id=model_id,
        skill_id=skill_id,
    )
    memory = cast(MemoryService, _state(request, "memory"))
    effective_knowledge = settings.knowledge_enabled and (
        payload.knowledge_enabled
        if payload.knowledge_enabled is not None
        else chat_session.knowledge_enabled
    )
    context = await memory.prepare_context(
        session_id=session_id,
        model_id=model_id,
        use_knowledge=effective_knowledge,
        knowledge_degraded_reason=(
            None if settings.knowledge_enabled else "knowledge_globally_disabled"
        ),
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in runner.stream(
            model_id=model_id,
            context=context,
            skill_id=skill_id,
            session_id=session_id,
        ):
            completed_content: str | None = None
            if event.type == "message.completed":
                usage = cast(dict[str, int], event.data.get("usage") or {})
                completed_content = str(event.data["content"])
                await repository.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=completed_content,
                    model_id=model_id,
                    skill_id=cast(str | None, event.data.get("skill_id")),
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                )
            yield event.to_sse()
            if completed_content is not None:
                await memory.extract_candidates(
                    session_id=session_id,
                    model_id=model_id,
                    user_content=payload.content,
                    assistant_content=completed_content,
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
