from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, HTTPException, Query, Request

from snow_grass.workflow.schemas import (
    BusinessPackResponse,
    DeploymentRequest,
    DeploymentResponse,
    FlowRunPageResponse,
    FlowRunResponse,
    FlowValidation,
    PreviewRequest,
    PublishRequest,
    PublishResponse,
    RollbackRequest,
    SaveDraftRequest,
    WorkflowCopyRequest,
    WorkflowCreateRequest,
    WorkflowDetail,
    WorkflowStatusRequest,
    WorkflowSummary,
    WorkflowVersionSummary,
)
from snow_grass.workflow.service import WorkflowService

workflow_router = APIRouter(prefix="/api/v1", tags=["workflows"])


def _service(request: Request) -> WorkflowService:
    return cast(WorkflowService, request.app.state.workflow_service)


@workflow_router.get("/business-packs", response_model=list[BusinessPackResponse])
async def business_packs(request: Request) -> list[BusinessPackResponse]:
    return _service(request).list_business_packs()


@workflow_router.get("/workflows", response_model=list[WorkflowSummary])
async def workflows(
    request: Request,
    business_type: Annotated[str | None, Query(max_length=120)] = None,
) -> list[WorkflowSummary]:
    return await _service(request).list_workflows(business_type)


@workflow_router.post("/workflows", response_model=WorkflowDetail, status_code=201)
async def create_workflow(request: Request, payload: WorkflowCreateRequest) -> WorkflowDetail:
    try:
        return await _service(request).create_workflow(
            business_type=payload.business_type,
            name=payload.name,
            description=payload.description,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
async def workflow_detail(request: Request, workflow_id: str) -> WorkflowDetail:
    try:
        return await _service(request).get_workflow(workflow_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.get(
    "/workflows/{workflow_id}/versions", response_model=list[WorkflowVersionSummary]
)
async def workflow_versions(request: Request, workflow_id: str) -> list[WorkflowVersionSummary]:
    try:
        return await _service(request).list_versions(workflow_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/copy", response_model=WorkflowDetail)
async def copy_workflow(
    request: Request, workflow_id: str, payload: WorkflowCopyRequest
) -> WorkflowDetail:
    try:
        return await _service(request).copy_workflow(workflow_id, name=payload.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.put("/workflows/{workflow_id}/status", response_model=WorkflowDetail)
async def workflow_status(
    request: Request, workflow_id: str, payload: WorkflowStatusRequest
) -> WorkflowDetail:
    try:
        return await _service(request).set_workflow_status(workflow_id, payload.status)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.put("/workflows/{workflow_id}/draft", response_model=WorkflowDetail)
async def save_draft(
    request: Request, workflow_id: str, payload: SaveDraftRequest
) -> WorkflowDetail:
    try:
        return await _service(request).save_draft(
            workflow_id,
            expected_revision=payload.expected_revision,
            graph=payload.graph,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/validate", response_model=FlowValidation)
async def validate_workflow(request: Request, workflow_id: str) -> FlowValidation:
    try:
        return await _service(request).validate_workflow(workflow_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/preview", response_model=FlowRunResponse)
async def preview(request: Request, workflow_id: str, payload: PreviewRequest) -> FlowRunResponse:
    try:
        return await _service(request).preview(
            workflow_id,
            expected_revision=payload.draft_revision,
            workflow_input=payload.input,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/publish", response_model=PublishResponse)
async def publish(request: Request, workflow_id: str, payload: PublishRequest) -> PublishResponse:
    try:
        return await _service(request).publish(
            workflow_id,
            expected_revision=payload.expected_revision,
            note=payload.version_note,
            activate=payload.activate,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/deployments", response_model=DeploymentResponse)
async def deploy(
    request: Request, workflow_id: str, payload: DeploymentRequest
) -> DeploymentResponse:
    try:
        return await _service(request).deploy(
            workflow_id,
            version_id=payload.version_id,
            expected_current_version_id=payload.expected_current_version_id,
            reason=payload.reason,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@workflow_router.post("/workflows/{workflow_id}/rollback", response_model=DeploymentResponse)
async def rollback(
    request: Request, workflow_id: str, payload: RollbackRequest
) -> DeploymentResponse:
    try:
        return await _service(request).deploy(
            workflow_id,
            version_id=payload.target_version_id,
            expected_current_version_id=payload.expected_current_version_id,
            reason=payload.reason,
            event_type="rollback",
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@workflow_router.get("/workflow-runs", response_model=FlowRunPageResponse)
async def workflow_runs(
    request: Request,
    workflow_id: Annotated[str | None, Query(max_length=36)] = None,
    status: Annotated[str | None, Query(max_length=20)] = None,
    version_id: Annotated[str | None, Query(max_length=36)] = None,
    preview: bool | None = None,
    started_after: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FlowRunPageResponse:
    return await _service(request).list_runs(
        workflow_id=workflow_id,
        status=status,
        version_id=version_id,
        preview=preview,
        started_after=started_after,
        limit=limit,
        offset=offset,
    )


@workflow_router.get("/workflow-runs/{run_id}", response_model=FlowRunResponse)
async def workflow_run(request: Request, run_id: str) -> FlowRunResponse:
    try:
        return await _service(request).get_run(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
