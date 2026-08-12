from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unicodedata import normalize
from uuid import uuid4

from snow_grass.activity.intelligence import ActivityIntelligence
from snow_grass.activity.models import ActivityEventRecord, HourlyActivitySummaryRecord
from snow_grass.activity.repository import ActivityRepository
from snow_grass.activity.schemas import (
    ActivityAnalysisCategory,
    ActivityAnalysisCategoryStats,
    ActivityAnalysisSource,
    ActivityAnalysisStatsResponse,
    ActivityAnalyzeRequest,
    ActivityAnalyzeResponse,
    ActivityAppSummary,
    ActivityBatchRequest,
    ActivityBatchResponse,
    ActivityDailySummary,
    ActivityDeleteResponse,
    ActivityEventResponse,
    ActivityPageResponse,
    ActivitySummaryResponse,
    HourlySummaryRequest,
    HourlySummaryResponse,
    HourlySummaryStatus,
)
from snow_grass.activity.security import ActivitySecurity
from snow_grass.knowledge.service import KnowledgeService

if TYPE_CHECKING:
    from snow_grass.workflow.service import WorkflowService


@dataclass
class _AppAccumulator:
    app_name: str
    event_count: int = 0
    character_count: int = 0


class ActivityService:
    def __init__(
        self,
        repository: ActivityRepository,
        *,
        retention_days: int = 30,
        security: ActivitySecurity | None = None,
        intelligence: ActivityIntelligence | None = None,
        workflows: WorkflowService | None = None,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._repository = repository
        self._retention_days = retention_days
        self._security = security or ActivitySecurity()
        self._intelligence = intelligence
        self._workflows = workflows
        self._knowledge = knowledge

    async def ingest(self, payload: ActivityBatchRequest) -> ActivityBatchResponse:
        await self.purge_expired()
        sanitized = [
            event.model_copy(update={"text": self._security.sanitize(event.text)})
            for event in payload.events
        ]
        accepted, duplicate = await self._repository.insert_batch(sanitized)
        if self._knowledge is not None and accepted:
            try:
                await self._knowledge.index_activity_events(accepted)
            except Exception:
                pass
        return ActivityBatchResponse(accepted=len(accepted), duplicate=duplicate)

    async def recent(self, *, limit: int) -> list[ActivityEventResponse]:
        return [
            ActivityEventResponse.model_validate(event)
            for event in await self._repository.list_recent(limit=limit)
        ]

    async def page(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        bundle_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        deduplicate: bool = False,
        dedupe_window_seconds: int = 5,
    ) -> ActivityPageResponse:
        self._validate_range(start=start, end=end)
        raw_items = await self._repository.list_filtered(
            query=query,
            bundle_id=bundle_id,
            start=start,
            end=end,
        )
        unique_items = self._deduplicate(raw_items, window_seconds=dedupe_window_seconds)
        selected = unique_items if deduplicate else raw_items
        total = len(selected)
        start_index = (page - 1) * page_size
        items = list(reversed(selected))[start_index : start_index + page_size]
        return ActivityPageResponse(
            items=[ActivityEventResponse.model_validate(item) for item in items],
            total=total,
            raw_total=len(raw_items),
            duplicate_total=len(raw_items) - len(unique_items),
            deduplicated=deduplicate,
            page=page,
            page_size=page_size,
        )

    async def summary(
        self,
        *,
        query: str | None = None,
        bundle_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        dedupe_window_seconds: int = 5,
        timezone_offset_minutes: int = 480,
    ) -> ActivitySummaryResponse:
        self._validate_range(start=start, end=end)
        raw_items = await self._repository.list_filtered(
            query=query,
            bundle_id=bundle_id,
            start=start,
            end=end,
        )
        unique_items = self._deduplicate(raw_items, window_seconds=dedupe_window_seconds)
        daily_counts: dict[date, list[int]] = defaultdict(lambda: [0, 0])
        app_counts: dict[str, _AppAccumulator] = {}
        offset = timedelta(minutes=timezone_offset_minutes)

        for event in unique_items:
            character_count = len(event.text)
            local_date = (self._utc_datetime(event.occurred_at) + offset).date()
            daily_counts[local_date][0] += 1
            daily_counts[local_date][1] += character_count

            app = app_counts.setdefault(
                event.bundle_id,
                _AppAccumulator(app_name=event.app_name),
            )
            app.app_name = event.app_name
            app.event_count += 1
            app.character_count += character_count

        daily = [
            ActivityDailySummary(
                date=day,
                event_count=counts[0],
                character_count=counts[1],
            )
            for day, counts in sorted(daily_counts.items())
        ]
        apps = sorted(
            (
                ActivityAppSummary(
                    app_name=values.app_name,
                    bundle_id=bundle,
                    event_count=values.event_count,
                    character_count=values.character_count,
                )
                for bundle, values in app_counts.items()
            ),
            key=lambda item: (-item.event_count, -item.character_count, item.bundle_id),
        )
        return ActivitySummaryResponse(
            raw_event_count=len(raw_items),
            unique_event_count=len(unique_items),
            duplicate_event_count=len(raw_items) - len(unique_items),
            character_count=sum(len(event.text) for event in unique_items),
            active_app_count=len(apps),
            daily=daily,
            apps=apps,
            generated_at=datetime.now(UTC),
        )

    async def purge_expired(self) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
        source_ids = await self._repository.delete_before(cutoff)
        if self._knowledge is not None:
            try:
                await self._knowledge.remove_activity_sources(source_ids)
                await self._knowledge.purge_expired_summaries()
            except Exception:
                pass
        return len(source_ids)

    async def analyze(self, payload: ActivityAnalyzeRequest) -> ActivityAnalyzeResponse:
        if self._intelligence is None:
            raise RuntimeError("Activity intelligence is unavailable")
        event = await self._repository.get_by_client_event_id(payload.client_event_id)
        if event is None:
            raise LookupError("Activity event not found")
        result = await self._intelligence.analyze(event.text, payload.available_actions)
        await self._repository.save_analysis(
            payload.client_event_id, result, analyzed_at=datetime.now(UTC)
        )
        return result

    async def analysis_stats(
        self,
        *,
        query: str | None = None,
        bundle_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> ActivityAnalysisStatsResponse:
        self._validate_range(start=start, end=end)
        events = await self._repository.list_filtered(
            query=query, bundle_id=bundle_id, start=start, end=end
        )
        analyzed = [event for event in events if event.analysis_source]
        source_counts = {
            source.value: sum(event.analysis_source == source.value for event in analyzed)
            for source in ActivityAnalysisSource
        }
        categories: list[ActivityAnalysisCategoryStats] = []
        for category in ActivityAnalysisCategory:
            matching = [event for event in analyzed if event.analysis_category == category.value]
            if not matching:
                continue
            categories.append(
                ActivityAnalysisCategoryStats(
                    category=category,
                    total=len(matching),
                    local=sum(
                        event.analysis_source == ActivityAnalysisSource.local.value
                        for event in matching
                    ),
                    deepseek=sum(
                        event.analysis_source == ActivityAnalysisSource.deepseek.value
                        for event in matching
                    ),
                    rules=sum(
                        event.analysis_source == ActivityAnalysisSource.rules.value
                        for event in matching
                    ),
                )
            )
        categories.sort(key=lambda item: (-item.total, item.category.value))
        return ActivityAnalysisStatsResponse(
            total_event_count=len(events),
            analyzed_count=len(analyzed),
            pending_count=len(events) - len(analyzed),
            local_count=source_counts[ActivityAnalysisSource.local.value],
            deepseek_count=source_counts[ActivityAnalysisSource.deepseek.value],
            rules_count=source_counts[ActivityAnalysisSource.rules.value],
            responded_count=sum(event.analysis_should_respond is True for event in analyzed),
            categories=categories,
            generated_at=datetime.now(UTC),
        )

    async def hourly_summary(self, payload: HourlySummaryRequest) -> HourlySummaryResponse:
        if self._intelligence is None:
            raise RuntimeError("Activity intelligence is unavailable")
        start = self._utc_datetime(payload.period_start).replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        existing = await self._repository.get_hourly_summary(start)
        if existing is not None:
            if self._knowledge is not None:
                try:
                    await self._knowledge.index_hourly_summary(existing)
                except Exception:
                    pass
            return self._summary_response(existing, status=HourlySummaryStatus.already_generated)

        raw_items = await self._repository.list_window(start=start, end=end)
        items = self._deduplicate(raw_items, window_seconds=5)
        effective_characters = sum(len("".join(event.text.split())) for event in items)
        now = datetime.now(UTC)
        if len(items) < 2 and effective_characters < 20:
            response = HourlySummaryResponse(
                id=str(uuid4()),
                period_start=start,
                period_end=end,
                status=HourlySummaryStatus.insufficient_activity,
                event_count=len(items),
                generated_at=now,
            )
        else:
            flow_version_id = None
            flow_run_id = None
            quality_status = None
            if self._workflows is not None:
                run = await self._workflows.run_business(
                    "activity_summary", {"events": items, "period_start": start.isoformat()}
                )
                if run.status != "succeeded":
                    raise RuntimeError("Activity summary workflow failed")
                content_payload = self._validate_flow_summary_output(run.output)
                flow_version_id = run.resolved_version_id or run.flow_version_id
                flow_run_id = run.run_id
                quality_status = "validated"
            else:
                internal = await self._intelligence.summarize(items)
                content_payload = {
                    key: [item.text for item in getattr(internal, key)]
                    for key in ("completed", "in_progress", "blockers", "next_steps")
                }
                content_payload["title"] = internal.title
                content_payload["keywords"] = internal.keywords
            section_values = [
                *content_payload.get("completed", []),
                *content_payload.get("in_progress", []),
                *content_payload.get("blockers", []),
                *content_payload.get("next_steps", []),
            ]
            response = HourlySummaryResponse(
                id=str(uuid4()),
                period_start=start,
                period_end=end,
                status=HourlySummaryStatus.generated,
                event_count=len(items),
                generated_at=now,
                title=str(
                    content_payload.get("title") or (section_values[0] if section_values else "")
                )[:240],
                keywords=[str(value)[:80] for value in content_payload.get("keywords", [])[:30]],
                completed=list(content_payload.get("completed", [])),
                in_progress=list(content_payload.get("in_progress", [])),
                blockers=list(content_payload.get("blockers", [])),
                next_steps=list(content_payload.get("next_steps", [])),
                flow_version_id=flow_version_id,
                flow_run_id=flow_run_id,
                quality_status=quality_status,
            )
        saved = await self._repository.save_hourly_summary(response)
        if self._knowledge is not None:
            try:
                await self._knowledge.index_hourly_summary(saved)
            except Exception:
                pass
        return self._summary_response(saved)

    @staticmethod
    def _validate_flow_summary_output(output: dict[str, Any]) -> dict[str, Any]:
        for key in ("completed", "in_progress", "blockers", "next_steps"):
            value = output.get(key)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise RuntimeError(f"Activity summary workflow returned invalid {key}")
        if not isinstance(output.get("keywords", []), list):
            raise RuntimeError("Activity summary workflow returned invalid keywords")
        return output

    async def list_hourly_summaries(
        self,
        *,
        limit: int,
        start: datetime | None = None,
        end: datetime | None = None,
        status: HourlySummaryStatus | None = None,
    ) -> list[HourlySummaryResponse]:
        self._validate_range(start=start, end=end)
        records = await self._repository.list_hourly_summaries(
            limit=limit,
            start=start,
            end=end,
            status=status.value if status else None,
        )
        return [self._summary_response(record) for record in records]

    @staticmethod
    def _summary_response(
        record: HourlyActivitySummaryRecord,
        status: HourlySummaryStatus | None = None,
    ) -> HourlySummaryResponse:
        return HourlySummaryResponse(
            id=record.id,
            period_start=record.period_start,
            period_end=record.period_end,
            status=status or HourlySummaryStatus(record.status),
            title=record.title,
            keywords=json.loads(record.keywords_json or "[]"),
            completed=json.loads(record.completed_json),
            in_progress=json.loads(record.in_progress_json),
            blockers=json.loads(record.blockers_json),
            next_steps=json.loads(record.next_steps_json),
            event_count=record.event_count,
            flow_version_id=record.flow_version_id,
            flow_run_id=record.flow_run_id,
            quality_status=record.quality_status,
            generated_at=record.generated_at,
        )

    async def delete_before(self, cutoff: datetime) -> ActivityDeleteResponse:
        source_ids = await self._repository.delete_before(cutoff)
        await self._remove_knowledge_sources(source_ids)
        return ActivityDeleteResponse(deleted=len(source_ids))

    async def delete_by_id(self, event_id: str) -> ActivityDeleteResponse:
        source_ids = await self._repository.delete_by_id(event_id)
        await self._remove_knowledge_sources(source_ids)
        return ActivityDeleteResponse(deleted=len(source_ids))

    async def delete_range(self, *, start: datetime, end: datetime) -> ActivityDeleteResponse:
        if start > end:
            raise ValueError("start must not be later than end")
        source_ids = await self._repository.delete_range(start=start, end=end)
        await self._remove_knowledge_sources(source_ids)
        return ActivityDeleteResponse(deleted=len(source_ids))

    async def _remove_knowledge_sources(self, source_ids: list[str]) -> None:
        if self._knowledge is None or not source_ids:
            return
        try:
            await self._knowledge.remove_activity_sources(source_ids)
        except Exception:
            pass

    @staticmethod
    def _validate_range(*, start: datetime | None, end: datetime | None) -> None:
        if start is not None and end is not None and start > end:
            raise ValueError("start must not be later than end")

    @classmethod
    def _deduplicate(
        cls,
        events: list[ActivityEventRecord],
        *,
        window_seconds: int,
    ) -> list[ActivityEventRecord]:
        kept: list[ActivityEventRecord] = []
        last_kept_at: dict[tuple[str, str], datetime] = {}
        window = timedelta(seconds=window_seconds)
        for event in events:
            key = (event.bundle_id, cls._normalized_text(event.text))
            occurred_at = cls._utc_datetime(event.occurred_at)
            previous = last_kept_at.get(key)
            if previous is not None and occurred_at - previous <= window:
                continue
            kept.append(event)
            last_kept_at[key] = occurred_at
        return kept

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(normalize("NFKC", value).split())

    @staticmethod
    def _utc_datetime(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
