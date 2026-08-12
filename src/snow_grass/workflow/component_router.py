from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, TypeVar, cast

from fastapi import APIRouter, HTTPException, Query, Request

from snow_grass.workflow.schemas import (
    ComponentCreateRequest,
    ComponentDetail,
    ComponentDraftRequest,
    ComponentManifest,
    ComponentPublishRequest,
    ComponentPublishResponse,
    ComponentStatusRequest,
    ComponentSummary,
    ComponentTestRequest,
    ComponentTestResponse,
    ComponentVersionResponse,
)
from snow_grass.workflow.service import ComponentService

component_router = APIRouter(prefix="/api/v1", tags=["components"])
T = TypeVar("T")


def _service(request: Request) -> ComponentService:
    return cast(ComponentService, request.app.state.component_service)


@component_router.get("/components", response_model=list[ComponentSummary])
async def components(
    request: Request,
    query: Annotated[str | None, Query(max_length=200)] = None,
    category: Annotated[str | None, Query(max_length=120)] = None,
    status: Annotated[str | None, Query(max_length=20)] = None,
) -> list[ComponentSummary]:
    return await _service(request).list_components(query=query, category=category, status=status)


@component_router.post("/components", response_model=ComponentDetail, status_code=201)
async def create_component(request: Request, payload: ComponentCreateRequest) -> ComponentDetail:
    return await _call(_service(request).create(payload))


@component_router.get("/components/{component_id}", response_model=ComponentDetail)
async def component_detail(request: Request, component_id: str) -> ComponentDetail:
    return await _call(_service(request).get(component_id))


@component_router.put("/components/{component_id}/draft", response_model=ComponentDetail)
async def save_component_draft(
    request: Request, component_id: str, payload: ComponentDraftRequest
) -> ComponentDetail:
    return await _call(
        _service(request).save_draft(
            component_id,
            expected_revision=payload.expected_revision,
            manifest=ComponentManifest.model_validate(
                payload.model_dump(exclude={"expected_revision"})
            ),
        )
    )


@component_router.post("/components/{component_id}/test", response_model=ComponentTestResponse)
async def test_component(
    request: Request, component_id: str, payload: ComponentTestRequest
) -> ComponentTestResponse:
    return await _call(
        _service(request).test(
            component_id,
            expected_revision=payload.expected_revision,
            config=payload.config,
            inputs=payload.inputs,
        )
    )


@component_router.post(
    "/components/{component_id}/publish", response_model=ComponentPublishResponse
)
async def publish_component(
    request: Request, component_id: str, payload: ComponentPublishRequest
) -> ComponentPublishResponse:
    return await _call(
        _service(request).publish(
            component_id,
            expected_revision=payload.expected_revision,
            release_note=payload.release_note,
        )
    )


@component_router.put("/components/{component_id}/status", response_model=ComponentDetail)
async def component_status(
    request: Request, component_id: str, payload: ComponentStatusRequest
) -> ComponentDetail:
    return await _call(_service(request).set_status(component_id, payload.status))


@component_router.get("/component-library", response_model=list[ComponentVersionResponse])
async def component_library(request: Request) -> list[ComponentVersionResponse]:
    return await _call(_service(request).library())


async def _call(awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
