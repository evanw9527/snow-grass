from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, HTTPException, Query, Request, status

from snow_grass.activity.schemas import (
    ActivityAnalysisStatsResponse,
    ActivityAnalyzeRequest,
    ActivityAnalyzeResponse,
    ActivityBatchRequest,
    ActivityBatchResponse,
    ActivityDeleteResponse,
    ActivityEventResponse,
    ActivityPageResponse,
    ActivitySummaryResponse,
    HourlySummaryRequest,
    HourlySummaryResponse,
    HourlySummaryStatus,
)
from snow_grass.activity.service import ActivityService

activity_router = APIRouter(prefix="/api/v1/activity-events", tags=["activity"])


@activity_router.post(
    "/batch", response_model=ActivityBatchResponse, status_code=status.HTTP_202_ACCEPTED
)
async def ingest(request: Request, payload: ActivityBatchRequest) -> ActivityBatchResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    return await service.ingest(payload)


@activity_router.get("", response_model=list[ActivityEventResponse])
async def recent(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ActivityEventResponse]:
    service = cast(ActivityService, request.app.state.activity_service)
    return await service.recent(limit=limit)


@activity_router.get("/page", response_model=ActivityPageResponse)
async def page(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    query: Annotated[str | None, Query(max_length=200)] = None,
    bundle_id: Annotated[str | None, Query(max_length=240)] = None,
    start: datetime | None = None,
    end: datetime | None = None,
    deduplicate: bool = False,
    dedupe_window_seconds: Annotated[int, Query(ge=1, le=300)] = 5,
) -> ActivityPageResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.page(
            page=page,
            page_size=page_size,
            query=query,
            bundle_id=bundle_id,
            start=start,
            end=end,
            deduplicate=deduplicate,
            dedupe_window_seconds=dedupe_window_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@activity_router.get("/summary", response_model=ActivitySummaryResponse)
async def summary(
    request: Request,
    query: Annotated[str | None, Query(max_length=200)] = None,
    bundle_id: Annotated[str | None, Query(max_length=240)] = None,
    start: datetime | None = None,
    end: datetime | None = None,
    dedupe_window_seconds: Annotated[int, Query(ge=1, le=300)] = 5,
    timezone_offset_minutes: Annotated[int, Query(ge=-720, le=840)] = 480,
) -> ActivitySummaryResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.summary(
            query=query,
            bundle_id=bundle_id,
            start=start,
            end=end,
            dedupe_window_seconds=dedupe_window_seconds,
            timezone_offset_minutes=timezone_offset_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@activity_router.post("/analyze", response_model=ActivityAnalyzeResponse)
async def analyze(
    request: Request, payload: ActivityAnalyzeRequest
) -> ActivityAnalyzeResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.analyze(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@activity_router.post("/hourly-summary", response_model=HourlySummaryResponse)
async def hourly_summary(
    request: Request, payload: HourlySummaryRequest
) -> HourlySummaryResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.hourly_summary(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@activity_router.get("/analysis-stats", response_model=ActivityAnalysisStatsResponse)
async def analysis_stats(
    request: Request,
    query: Annotated[str | None, Query(max_length=200)] = None,
    bundle_id: Annotated[str | None, Query(max_length=240)] = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ActivityAnalysisStatsResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.analysis_stats(
            query=query, bundle_id=bundle_id, start=start, end=end
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@activity_router.get("/hourly-summaries", response_model=list[HourlySummaryResponse])
async def hourly_summaries(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=48)] = 24,
    start: datetime | None = None,
    end: datetime | None = None,
    summary_status: Annotated[
        HourlySummaryStatus | None, Query(alias="status")
    ] = None,
) -> list[HourlySummaryResponse]:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.list_hourly_summaries(
            limit=limit, start=start, end=end, status=summary_status
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@activity_router.delete("/range", response_model=ActivityDeleteResponse)
async def delete_range(
    request: Request, start: datetime, end: datetime
) -> ActivityDeleteResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    try:
        return await service.delete_range(start=start, end=end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@activity_router.delete("/{event_id}", response_model=ActivityDeleteResponse)
async def delete_one(request: Request, event_id: str) -> ActivityDeleteResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    return await service.delete_by_id(event_id)


@activity_router.delete("", response_model=ActivityDeleteResponse)
async def delete_before(request: Request, before: datetime) -> ActivityDeleteResponse:
    service = cast(ActivityService, request.app.state.activity_service)
    return await service.delete_before(before)
