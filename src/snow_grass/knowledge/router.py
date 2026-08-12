from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, HTTPException, Query, Request, status

from snow_grass.core.config import Settings
from snow_grass.knowledge.repository import KnowledgeRepository
from snow_grass.knowledge.schemas import (
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
    KnowledgeBatchAction,
    KnowledgeBatchRequest,
    KnowledgeBatchResponse,
    KnowledgeDocumentResponse,
    KnowledgeDocumentStatus,
    KnowledgePageResponse,
    KnowledgeReindexRequest,
    KnowledgeReindexResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSourceType,
    KnowledgeStatsResponse,
    KnowledgeStatusUpdateRequest,
)
from snow_grass.knowledge.service import KnowledgeService

knowledge_router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def _service(request: Request) -> KnowledgeService:
    return cast(KnowledgeService, request.app.state.knowledge_service)


def _repository(request: Request) -> KnowledgeRepository:
    return cast(KnowledgeRepository, request.app.state.knowledge_repository)


@knowledge_router.get("/bases", response_model=list[KnowledgeBaseResponse])
async def bases(request: Request) -> list[KnowledgeBaseResponse]:
    return await _repository(request).list_bases()


@knowledge_router.patch(
    "/bases/{source_type}", response_model=KnowledgeBaseResponse
)
async def update_base(
    request: Request,
    source_type: KnowledgeSourceType,
    payload: KnowledgeBaseUpdateRequest,
) -> KnowledgeBaseResponse:
    return await _repository(request).set_base_enabled(source_type, payload.enabled)


@knowledge_router.post("/search", response_model=KnowledgeSearchResponse)
async def search(request: Request, payload: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    return await _service(request).search(
        query=payload.query,
        limit=payload.limit,
        source_types=payload.source_types,
        start=payload.start,
        end=payload.end,
    )


@knowledge_router.get("/stats", response_model=KnowledgeStatsResponse)
async def stats(request: Request) -> KnowledgeStatsResponse:
    settings = cast(Settings, request.app.state.settings)
    return await _repository(request).stats(globally_enabled=settings.knowledge_enabled)


@knowledge_router.get("/documents", response_model=KnowledgePageResponse)
async def documents(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
    query: Annotated[str | None, Query(max_length=200)] = None,
    source_type: KnowledgeSourceType | None = None,
    document_status: Annotated[
        KnowledgeDocumentStatus | None, Query(alias="status")
    ] = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> KnowledgePageResponse:
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="start must not be later than end")
    return await _service(request).page(
        page=page,
        page_size=page_size,
        query=query,
        source_type=source_type,
        status=document_status,
        start=start,
        end=end,
    )


@knowledge_router.post("/documents/batch", response_model=KnowledgeBatchResponse)
async def batch(
    request: Request, payload: KnowledgeBatchRequest
) -> KnowledgeBatchResponse:
    return await _repository(request).batch_action(payload.document_ids, payload.action)


@knowledge_router.post("/reindex", response_model=KnowledgeReindexResponse)
async def reindex(
    request: Request, payload: KnowledgeReindexRequest
) -> KnowledgeReindexResponse:
    return await _service(request).reindex(
        source_types=payload.source_types,
        start=payload.start,
        restore_deleted=payload.restore_deleted,
    )


@knowledge_router.patch(
    "/documents/{document_id}", response_model=KnowledgeDocumentResponse
)
async def update_document(
    request: Request, document_id: str, payload: KnowledgeStatusUpdateRequest
) -> KnowledgeDocumentResponse:
    record = await _repository(request).update_status(document_id, payload.status)
    if record is None:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    return _service(request).response(record)


@knowledge_router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(request: Request, document_id: str) -> None:
    result = await _repository(request).batch_action(
        [document_id], action=KnowledgeBatchAction.delete
    )
    if result.matched == 0:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
