from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import cast
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from snow_grass.activity.models import ActivityEventRecord, HourlyActivitySummaryRecord
from snow_grass.activity.schemas import (
    ActivityAnalyzeResponse,
    ActivityEventInput,
    HourlySummaryResponse,
)


class ActivityRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def insert_batch(
        self, events: Sequence[ActivityEventInput]
    ) -> tuple[list[ActivityEventRecord], int]:
        async with self._session_factory() as session:
            ids = [event.client_event_id for event in events]
            existing = set(
                await session.scalars(
                    select(ActivityEventRecord.client_event_id).where(
                        ActivityEventRecord.client_event_id.in_(ids)
                    )
                )
            )
            fresh = [event for event in events if event.client_event_id not in existing]
            records = [
                ActivityEventRecord(id=str(uuid4()), **event.model_dump()) for event in fresh
            ]
            session.add_all(records)
            await session.commit()
            return records, len(events) - len(fresh)

    async def list_recent(self, *, limit: int) -> list[ActivityEventRecord]:
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(ActivityEventRecord)
                    .order_by(ActivityEventRecord.occurred_at.desc())
                    .limit(limit)
                )
            )

    async def get_by_client_event_id(self, client_event_id: str) -> ActivityEventRecord | None:
        async with self._session_factory() as session:
            return cast(
                ActivityEventRecord | None,
                await session.scalar(
                    select(ActivityEventRecord).where(
                        ActivityEventRecord.client_event_id == client_event_id
                    )
                ),
            )

    async def save_analysis(
        self, client_event_id: str, analysis: ActivityAnalyzeResponse, analyzed_at: datetime
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(ActivityEventRecord)
                .where(ActivityEventRecord.client_event_id == client_event_id)
                .values(
                    analysis_source=analysis.analysis_source.value,
                    analysis_category=analysis.category.value,
                    analysis_action=analysis.action.value,
                    analysis_severity=analysis.severity,
                    analysis_should_respond=analysis.should_respond,
                    analyzed_at=analyzed_at,
                )
            )
            await session.commit()

    async def list_window(
        self, *, start: datetime, end: datetime
    ) -> list[ActivityEventRecord]:
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(ActivityEventRecord)
                    .where(
                        ActivityEventRecord.occurred_at >= start,
                        ActivityEventRecord.occurred_at < end,
                    )
                    .order_by(ActivityEventRecord.occurred_at, ActivityEventRecord.id)
                )
            )

    async def get_hourly_summary(
        self, period_start: datetime
    ) -> HourlyActivitySummaryRecord | None:
        async with self._session_factory() as session:
            return cast(
                HourlyActivitySummaryRecord | None,
                await session.scalar(
                    select(HourlyActivitySummaryRecord).where(
                        HourlyActivitySummaryRecord.period_start == period_start
                    )
                ),
            )

    async def save_hourly_summary(
        self, summary: HourlySummaryResponse
    ) -> HourlyActivitySummaryRecord:
        async with self._session_factory() as session:
            record = HourlyActivitySummaryRecord(
                id=summary.id,
                period_start=summary.period_start,
                period_end=summary.period_end,
                status=summary.status.value,
                title=summary.title,
                keywords_json=json.dumps(summary.keywords, ensure_ascii=False),
                completed_json=json.dumps(summary.completed, ensure_ascii=False),
                in_progress_json=json.dumps(summary.in_progress, ensure_ascii=False),
                blockers_json=json.dumps(summary.blockers, ensure_ascii=False),
                next_steps_json=json.dumps(summary.next_steps, ensure_ascii=False),
                event_count=summary.event_count,
                flow_version_id=summary.flow_version_id,
                flow_run_id=summary.flow_run_id,
                quality_status=summary.quality_status,
                generated_at=summary.generated_at,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def list_hourly_summaries(
        self,
        *,
        limit: int,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
    ) -> list[HourlyActivitySummaryRecord]:
        filters: list[ColumnElement[bool]] = []
        if start:
            filters.append(HourlyActivitySummaryRecord.period_start >= start)
        if end:
            filters.append(HourlyActivitySummaryRecord.period_start < end)
        if status:
            filters.append(HourlyActivitySummaryRecord.status == status)
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(HourlyActivitySummaryRecord)
                    .where(*filters)
                    .order_by(HourlyActivitySummaryRecord.period_start.desc())
                    .limit(limit)
                )
            )

    async def list_page(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        bundle_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[list[ActivityEventRecord], int]:
        filters = self._filters(
            query=query, bundle_id=bundle_id, start=start, end=end
        )

        async with self._session_factory() as session:
            count_result = await session.execute(
                select(func.count(ActivityEventRecord.id)).where(*filters)
            )
            total = count_result.scalar_one()
            items = list(
                await session.scalars(
                    select(ActivityEventRecord)
                    .where(*filters)
                    .order_by(
                        ActivityEventRecord.occurred_at.desc(),
                        ActivityEventRecord.id.desc(),
                    )
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            )
            return items, total

    async def list_filtered(
        self,
        *,
        query: str | None = None,
        bundle_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[ActivityEventRecord]:
        filters = self._filters(
            query=query, bundle_id=bundle_id, start=start, end=end
        )
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(ActivityEventRecord)
                    .where(*filters)
                    .order_by(
                        ActivityEventRecord.occurred_at.asc(),
                        ActivityEventRecord.id.asc(),
                    )
                )
            )

    @staticmethod
    def _filters(
        *,
        query: str | None,
        bundle_id: str | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if query:
            filters.append(
                or_(
                    ActivityEventRecord.text.contains(query, autoescape=True),
                    ActivityEventRecord.app_name.contains(query, autoescape=True),
                    ActivityEventRecord.bundle_id.contains(query, autoescape=True),
                )
            )
        if bundle_id:
            filters.append(ActivityEventRecord.bundle_id == bundle_id)
        if start:
            filters.append(ActivityEventRecord.occurred_at >= start)
        if end:
            filters.append(ActivityEventRecord.occurred_at <= end)
        return filters

    async def delete_before(self, cutoff: datetime) -> list[str]:
        async with self._session_factory() as session:
            expired_ids = list(
                await session.scalars(
                    select(ActivityEventRecord.client_event_id).where(
                        ActivityEventRecord.occurred_at < cutoff
                    )
                )
            )
            await session.execute(
                delete(ActivityEventRecord).where(ActivityEventRecord.occurred_at < cutoff)
            )
            await session.commit()
            return expired_ids

    async def delete_by_id(self, event_id: str) -> list[str]:
        async with self._session_factory() as session:
            matched_ids = list(
                await session.scalars(
                    select(ActivityEventRecord.client_event_id).where(
                        ActivityEventRecord.id == event_id
                    )
                )
            )
            await session.execute(
                delete(ActivityEventRecord).where(ActivityEventRecord.id == event_id)
            )
            await session.commit()
            return matched_ids

    async def delete_range(self, *, start: datetime, end: datetime) -> list[str]:
        async with self._session_factory() as session:
            matched_ids = list(
                await session.scalars(
                    select(ActivityEventRecord.client_event_id).where(
                        ActivityEventRecord.occurred_at >= start,
                        ActivityEventRecord.occurred_at <= end,
                    )
                )
            )
            await session.execute(
                delete(ActivityEventRecord).where(
                    ActivityEventRecord.occurred_at >= start,
                    ActivityEventRecord.occurred_at <= end,
                )
            )
            await session.commit()
            return matched_ids
