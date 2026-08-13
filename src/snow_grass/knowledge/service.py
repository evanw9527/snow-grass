from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

from snow_grass.activity.models import ActivityEventRecord, HourlyActivitySummaryRecord
from snow_grass.activity.repository import ActivityRepository
from snow_grass.knowledge.models import KnowledgeDocumentRecord
from snow_grass.knowledge.repository import KnowledgeRepository
from snow_grass.knowledge.schemas import (
    KnowledgeDocumentInput,
    KnowledgeDocumentResponse,
    KnowledgeDocumentStatus,
    KnowledgePageResponse,
    KnowledgeReindexResponse,
    KnowledgeSearchItem,
    KnowledgeSearchResponse,
    KnowledgeSourceType,
)


class KnowledgeService:
    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        activity_repository: ActivityRepository,
        search_timeout_ms: int,
        summary_retention_days: int,
    ) -> None:
        self._repository = repository
        self._activity_repository = activity_repository
        self._search_timeout = search_timeout_ms / 1000
        self._summary_retention_days = summary_retention_days

    async def index_activity_events(
        self,
        events: Sequence[ActivityEventRecord],
        *,
        restore_deleted: bool = False,
    ) -> tuple[int, int, int]:
        documents = []
        for event in events:
            occurred_at = _utc(event.occurred_at)
            item = KnowledgeDocumentInput(
                source_type=KnowledgeSourceType.activity_event,
                source_id=event.client_event_id,
                title=f"{event.app_name} 输入",
                content=event.text,
                source_app=event.app_name,
                occurred_at=occurred_at,
            )
            documents.append((item, _hash(item)))
        return await self._repository.upsert_documents(documents, restore_deleted=restore_deleted)

    async def index_hourly_summary(
        self, summary: HourlyActivitySummaryRecord, *, restore_deleted: bool = False
    ) -> tuple[int, int, int]:
        sections = [
            ("已完成", json.loads(summary.completed_json)),
            ("进行中", json.loads(summary.in_progress_json)),
            ("阻塞", json.loads(summary.blockers_json)),
            ("下一步", json.loads(summary.next_steps_json)),
        ]
        content = (
            "\n".join(
                f"{label}：" + "；".join(str(value) for value in values)
                for label, values in sections
                if values
            )
            or "该时段没有足够内容生成总结。"
        )
        occurred_at = _utc(summary.period_start)
        local_occurred_at = occurred_at.astimezone(_SHANGHAI)
        title = summary.title or f"工作总结 · {local_occurred_at:%Y-%m-%d %H:00}"
        keywords = json.loads(summary.keywords_json or "[]")
        item = KnowledgeDocumentInput(
            source_type=KnowledgeSourceType.hourly_summary,
            source_id=summary.id,
            title=title,
            content=content,
            keywords=keywords,
            occurred_at=occurred_at,
        )
        return await self._repository.upsert_documents(
            [(item, _hash(item))], restore_deleted=restore_deleted
        )

    async def search(
        self,
        *,
        query: str,
        limit: int,
        source_types: list[KnowledgeSourceType] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> KnowledgeSearchResponse:
        if not query.strip():
            return KnowledgeSearchResponse(items=[])
        enabled_sources = await self._repository.enabled_source_types(source_types)
        if not enabled_sources:
            return KnowledgeSearchResponse(items=[], degraded_reason="knowledge_bases_disabled")
        overview_range = _work_overview_range(query)
        if overview_range is not None:
            range_start, range_end = overview_range
            return await self._search_work_overview(
                query=query,
                limit=limit,
                source_types=enabled_sources,
                start=start or range_start,
                end=end or range_end,
            )
        try:
            candidates, degraded = await asyncio.wait_for(
                self._repository.search(
                    query=query,
                    limit=limit,
                    source_types=enabled_sources,
                    start=start,
                    end=end,
                ),
                timeout=self._search_timeout,
            )
        except TimeoutError:
            return KnowledgeSearchResponse(items=[], degraded_reason="search_timeout")
        now = datetime.now(UTC)
        ranked: list[KnowledgeSearchItem] = []
        seen: set[str] = set()
        for record, lexical_score in candidates:
            normalized = " ".join(record.content.casefold().split())
            if (
                normalized in seen
                or _is_empty_summary(record)
                or _is_query_echo(query, record.content)
            ):
                continue
            seen.add(normalized)
            age_days = max(0.0, (now - _utc(record.occurred_at)).total_seconds() / 86_400)
            recency = 1.0 / (1.0 + age_days / 30)
            source_weight = 1.25 if record.source_type == "hourly_summary" else 1.0
            score = (1.0 + lexical_score) * source_weight * (0.75 + 0.25 * recency)
            ranked.append(
                KnowledgeSearchItem(
                    id=record.id,
                    content=record.content,
                    title=record.title,
                    source_type=KnowledgeSourceType(record.source_type),
                    source_id=record.source_id,
                    source_app=record.source_app,
                    occurred_at=_utc(record.occurred_at),
                    score=round(score, 6),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return KnowledgeSearchResponse(items=ranked[:limit], degraded_reason=degraded)

    async def _search_work_overview(
        self,
        *,
        query: str,
        limit: int,
        source_types: list[KnowledgeSourceType],
        start: datetime,
        end: datetime,
    ) -> KnowledgeSearchResponse:
        """Retrieve a time window for work-overview questions, not lexical matches."""
        try:
            summary_records: list[KnowledgeDocumentRecord] = []
            activity_records: list[KnowledgeDocumentRecord] = []
            if KnowledgeSourceType.hourly_summary in source_types:
                summary_records, _ = await asyncio.wait_for(
                    self._repository.list_page(
                        page=1,
                        page_size=max(48, limit * 6),
                        source_type=KnowledgeSourceType.hourly_summary,
                        status=KnowledgeDocumentStatus.active,
                        start=start,
                        end=end,
                    ),
                    timeout=self._search_timeout,
                )
            if KnowledgeSourceType.activity_event in source_types:
                activity_records, _ = await asyncio.wait_for(
                    self._repository.list_page(
                        page=1,
                        page_size=max(96, limit * 12),
                        source_type=KnowledgeSourceType.activity_event,
                        status=KnowledgeDocumentStatus.active,
                        start=start,
                        end=end,
                    ),
                    timeout=self._search_timeout,
                )
        except TimeoutError:
            return KnowledgeSearchResponse(items=[], degraded_reason="search_timeout")

        summaries = _unique_records(
            record for record in summary_records if not _is_empty_summary(record)
        )
        activities = _unique_records(
            record
            for record in activity_records
            if not _is_query_echo(query, record.content)
            and _activity_signal_score(record) is not None
        )
        activities.sort(key=lambda record: _activity_signal_score(record) or 0.0, reverse=True)

        # Summaries provide structure while raw activities preserve concrete evidence.
        summary_quota = max(1, limit - 2) if activities else limit
        selected = summaries[:summary_quota]
        selected_summary_count = len(selected)
        selected.extend(activities[: max(0, limit - len(selected))])
        if len(selected) < limit:
            remaining = limit - len(selected)
            selected.extend(summaries[selected_summary_count : selected_summary_count + remaining])

        now = datetime.now(UTC)
        items = []
        for record in selected[:limit]:
            age_days = max(0.0, (now - _utc(record.occurred_at)).total_seconds() / 86_400)
            source_weight = 2.0 if record.source_type == "hourly_summary" else 1.0
            items.append(
                KnowledgeSearchItem(
                    id=record.id,
                    content=record.content,
                    title=record.title,
                    source_type=KnowledgeSourceType(record.source_type),
                    source_id=record.source_id,
                    source_app=record.source_app,
                    occurred_at=_utc(record.occurred_at),
                    score=round(source_weight / (1.0 + age_days / 30), 6),
                )
            )
        return KnowledgeSearchResponse(items=items)

    async def page(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        source_type: KnowledgeSourceType | None = None,
        status: KnowledgeDocumentStatus | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> KnowledgePageResponse:
        records, total = await self._repository.list_page(
            page=page,
            page_size=page_size,
            query=query,
            source_type=source_type,
            status=status,
            start=start,
            end=end,
        )
        return KnowledgePageResponse(
            items=[self.response(item) for item in records],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def reindex(
        self,
        *,
        source_types: list[KnowledgeSourceType] | None,
        start: datetime | None,
        restore_deleted: bool,
    ) -> KnowledgeReindexResponse:
        selected = set(source_types or list(KnowledgeSourceType))
        indexed = updated = skipped = failed = 0
        if KnowledgeSourceType.activity_event in selected:
            events = await self._activity_repository.list_filtered(start=start)
            for event in events:
                try:
                    values = await self.index_activity_events(
                        [event], restore_deleted=restore_deleted
                    )
                    indexed += values[0]
                    updated += values[1]
                    skipped += values[2]
                except Exception:
                    failed += 1
        if KnowledgeSourceType.hourly_summary in selected:
            summaries = await self._activity_repository.list_hourly_summaries(
                limit=100_000, start=start
            )
            for summary in summaries:
                try:
                    values = await self.index_hourly_summary(
                        summary, restore_deleted=restore_deleted
                    )
                    indexed += values[0]
                    updated += values[1]
                    skipped += values[2]
                except Exception:
                    failed += 1
        return KnowledgeReindexResponse(
            indexed=indexed, updated=updated, skipped=skipped, failed=failed
        )

    async def purge_expired_summaries(self) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=self._summary_retention_days)
        return await self._repository.purge_summary_before(cutoff)

    async def remove_activity_sources(self, source_ids: list[str]) -> int:
        return await self._repository.remove_by_sources(
            KnowledgeSourceType.activity_event, source_ids
        )

    @staticmethod
    def response(record: KnowledgeDocumentRecord) -> KnowledgeDocumentResponse:
        return KnowledgeDocumentResponse(
            id=record.id,
            source_type=KnowledgeSourceType(record.source_type),
            source_id=record.source_id,
            title=record.title,
            content=record.content,
            keywords=json.loads(record.keywords_json or "[]"),
            source_app=record.source_app,
            occurred_at=_utc(record.occurred_at),
            status=KnowledgeDocumentStatus(record.status),
            created_at=_utc(record.created_at),
            updated_at=_utc(record.updated_at),
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _hash(item: KnowledgeDocumentInput) -> str:
    payload = item.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


_SHANGHAI = ZoneInfo("Asia/Shanghai")
_WORK_OVERVIEW_MARKERS = (
    "工作",
    "日常工作",
    "工作内容",
    "工作总结",
    "工作进展",
    "我的工作",
    "做了什么",
    "干了什么",
    "忙了什么",
)
_EMPTY_SUMMARY_MARKER = "没有足够内容生成总结"
_ACTIVITY_META_MARKERS = (
    "做了什么",
    "干了什么",
    "每天的总结",
    "帮我确认",
    "能不能用",
    "你查询了吗",
)
_TERMINAL_APPS = {"iterm2", "terminal", "终端"}


def _work_overview_range(
    query: str, *, now: datetime | None = None
) -> tuple[datetime, datetime] | None:
    normalized = "".join(query.casefold().split())
    if not any(marker in normalized for marker in _WORK_OVERVIEW_MARKERS):
        return None

    local_now = now.astimezone(_SHANGHAI) if now is not None else datetime.now(_SHANGHAI)
    today = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "昨天" in normalized or "昨日" in normalized:
        local_start = today - timedelta(days=1)
        local_end = today
    elif "今天" in normalized or "今日" in normalized or "当天" in normalized:
        local_start = today
        local_end = today + timedelta(days=1)
    else:
        local_start = today - timedelta(days=6)
        local_end = today + timedelta(days=1)
    return (
        local_start.astimezone(UTC),
        (local_end - timedelta(microseconds=1)).astimezone(UTC),
    )


def _is_empty_summary(record: KnowledgeDocumentRecord) -> bool:
    return (
        record.source_type == KnowledgeSourceType.hourly_summary.value
        and _EMPTY_SUMMARY_MARKER in record.content
    )


def _is_query_echo(query: str, content: str) -> bool:
    query_text = re.sub(r"[^\w\u4e00-\u9fff]", "", query.casefold())
    content_text = re.sub(r"[^\w\u4e00-\u9fff]", "", content.casefold())
    if not query_text or not content_text:
        return False
    if query_text == content_text:
        return True
    length_ratio = min(len(query_text), len(content_text)) / max(len(query_text), len(content_text))
    return length_ratio >= 0.65 and SequenceMatcher(None, query_text, content_text).ratio() >= 0.72


def _activity_signal_score(record: KnowledgeDocumentRecord) -> float | None:
    content = " ".join(record.content.split())
    normalized = content.casefold()
    if (
        len(content) < 10
        or (record.source_app or "").casefold() in _TERMINAL_APPS
        or any(marker in normalized for marker in _ACTIVITY_META_MARKERS)
    ):
        return None
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", content))
    if chinese_count < 6:
        return None
    age_hours = max(
        0.0,
        (datetime.now(UTC) - _utc(record.occurred_at)).total_seconds() / 3_600,
    )
    return min(len(content), 80) + 8.0 / (1.0 + age_hours / 24)


def _unique_records(
    records: Iterable[KnowledgeDocumentRecord],
) -> list[KnowledgeDocumentRecord]:
    unique: list[KnowledgeDocumentRecord] = []
    seen: set[str] = set()
    for record in records:
        normalized = " ".join(record.content.casefold().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(record)
    return unique
