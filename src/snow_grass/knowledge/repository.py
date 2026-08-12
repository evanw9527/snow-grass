from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from snow_grass.knowledge.models import KnowledgeBaseRecord, KnowledgeDocumentRecord
from snow_grass.knowledge.schemas import (
    KnowledgeBaseResponse,
    KnowledgeBatchAction,
    KnowledgeBatchResponse,
    KnowledgeDocumentInput,
    KnowledgeDocumentStatus,
    KnowledgeSourceType,
    KnowledgeStatsResponse,
)
from snow_grass.persistence.models import utc_now


class KnowledgeRepository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        workspace_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._workspace_id = workspace_id

    async def upsert_documents(
        self,
        documents: Sequence[tuple[KnowledgeDocumentInput, str]],
        *,
        restore_deleted: bool = False,
    ) -> tuple[int, int, int]:
        indexed = updated = skipped = 0
        async with self._session_factory() as session:
            for item, content_hash in documents:
                record = cast(
                    KnowledgeDocumentRecord | None,
                    await session.scalar(
                        select(KnowledgeDocumentRecord).where(
                            KnowledgeDocumentRecord.workspace_id == self._workspace_id,
                            KnowledgeDocumentRecord.source_type == item.source_type.value,
                            KnowledgeDocumentRecord.source_id == item.source_id,
                        )
                    ),
                )
                if record is None:
                    session.add(
                        KnowledgeDocumentRecord(
                            id=str(uuid4()),
                            workspace_id=self._workspace_id,
                            source_type=item.source_type.value,
                            source_id=item.source_id,
                            title=item.title,
                            content=item.content,
                            keywords_json=_json_list(item.keywords),
                            source_app=item.source_app,
                            occurred_at=item.occurred_at,
                            content_hash=content_hash,
                        )
                    )
                    indexed += 1
                    continue
                target_status = record.status
                if restore_deleted and record.status == KnowledgeDocumentStatus.deleted.value:
                    target_status = KnowledgeDocumentStatus.active.value
                unchanged = (
                    record.content_hash == content_hash
                    and record.title == item.title
                    and record.status == target_status
                )
                if unchanged:
                    skipped += 1
                    continue
                record.title = item.title
                record.content = item.content
                record.keywords_json = _json_list(item.keywords)
                record.source_app = item.source_app
                record.occurred_at = item.occurred_at
                record.content_hash = content_hash
                record.status = target_status
                record.updated_at = utc_now()
                updated += 1
            await session.commit()
        return indexed, updated, skipped

    async def list_page(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        source_type: KnowledgeSourceType | None = None,
        status: KnowledgeDocumentStatus | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[list[KnowledgeDocumentRecord], int]:
        filters = self._filters(
            query=query,
            source_types=[source_type] if source_type else None,
            status=status,
            start=start,
            end=end,
        )
        async with self._session_factory() as session:
            total = int(
                await session.scalar(
                    select(func.count(KnowledgeDocumentRecord.id)).where(*filters)
                )
                or 0
            )
            items = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord)
                        .where(*filters)
                        .order_by(
                            KnowledgeDocumentRecord.occurred_at.desc(),
                            KnowledgeDocumentRecord.id.desc(),
                        )
                        .offset((page - 1) * page_size)
                        .limit(page_size)
                    )
                ).all()
            )
        return items, total

    async def search(
        self,
        *,
        query: str,
        limit: int,
        source_types: list[KnowledgeSourceType] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[list[tuple[KnowledgeDocumentRecord, float]], str | None]:
        try:
            matches = await self._fts_search(
                query=query,
                limit=limit,
                source_types=source_types,
                start=start,
                end=end,
            )
            if matches:
                return matches, None
            return await self._fallback_search(
                query=query,
                limit=limit,
                source_types=source_types,
                start=start,
                end=end,
            ), "fts_no_match"
        except Exception:
            return await self._fallback_search(
                query=query,
                limit=limit,
                source_types=source_types,
                start=start,
                end=end,
            ), "fts_unavailable"

    async def _fts_search(
        self,
        *,
        query: str,
        limit: int,
        source_types: list[KnowledgeSourceType] | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[tuple[KnowledgeDocumentRecord, float]]:
        conditions = ["d.workspace_id = :workspace_id", "d.status = 'active'"]
        params: dict[str, object] = {
            "workspace_id": self._workspace_id,
            "query": _fts_query(query),
            "limit": max(limit * 4, limit),
        }
        if source_types:
            placeholders = []
            for index, source_type in enumerate(source_types):
                key = f"source_{index}"
                placeholders.append(f":{key}")
                params[key] = source_type.value
            conditions.append(f"d.source_type IN ({', '.join(placeholders)})")
        if start:
            conditions.append("d.occurred_at >= :start")
            params["start"] = start
        if end:
            conditions.append("d.occurred_at <= :end")
            params["end"] = end
        statement = text(
            "SELECT d.id, bm25(knowledge_documents_fts) AS rank "
            "FROM knowledge_documents_fts "
            "JOIN knowledge_documents d ON d.rowid = knowledge_documents_fts.rowid "
            f"WHERE knowledge_documents_fts MATCH :query AND {' AND '.join(conditions)} "
            "ORDER BY rank LIMIT :limit"
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement, params)).all()
            if not rows:
                return []
            ids = [str(row[0]) for row in rows]
            records = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord).where(
                            KnowledgeDocumentRecord.id.in_(ids)
                        )
                    )
                ).all()
            )
        by_id = {record.id: record for record in records}
        return [
            (by_id[str(row[0])], max(0.0, -float(row[1])))
            for row in rows
            if str(row[0]) in by_id
        ]

    async def _fallback_search(
        self,
        *,
        query: str,
        limit: int,
        source_types: list[KnowledgeSourceType] | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[tuple[KnowledgeDocumentRecord, float]]:
        filters = self._filters(
            query=query,
            source_types=source_types,
            status=KnowledgeDocumentStatus.active,
            start=start,
            end=end,
        )
        async with self._session_factory() as session:
            records = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord)
                        .where(*filters)
                        .order_by(KnowledgeDocumentRecord.occurred_at.desc())
                        .limit(max(limit * 4, limit))
                    )
                ).all()
            )
        normalized = query.casefold()
        return [
            (
                record,
                1.0
                + record.content.casefold().count(normalized)
                + record.title.casefold().count(normalized),
            )
            for record in records
        ]

    async def stats(self, *, globally_enabled: bool) -> KnowledgeStatsResponse:
        async with self._session_factory() as session:
            records = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord).where(
                            KnowledgeDocumentRecord.workspace_id == self._workspace_id
                        )
                    )
                ).all()
            )
            index_mode = "contains_fallback"
            try:
                exists = await session.scalar(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type='table' AND name='knowledge_documents_fts'"
                    )
                )
                if exists:
                    index_mode = "fts5_trigram"
            except Exception:
                pass
        return KnowledgeStatsResponse(
            globally_enabled=globally_enabled,
            index_mode=index_mode,
            total=len(records),
            active=sum(item.status == "active" for item in records),
            disabled=sum(item.status == "disabled" for item in records),
            deleted=sum(item.status == "deleted" for item in records),
            activity_events=sum(item.source_type == "activity_event" for item in records),
            hourly_summaries=sum(item.source_type == "hourly_summary" for item in records),
            last_indexed_at=_utc_or_none(
                max((item.updated_at for item in records), default=None)
            ),
        )

    async def list_bases(self) -> list[KnowledgeBaseResponse]:
        async with self._session_factory() as session:
            policies = list(
                (
                    await session.scalars(
                        select(KnowledgeBaseRecord).where(
                            KnowledgeBaseRecord.workspace_id == self._workspace_id
                        )
                    )
                ).all()
            )
            documents = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord).where(
                            KnowledgeDocumentRecord.workspace_id == self._workspace_id
                        )
                    )
                ).all()
            )
        enabled_by_source = {item.source_type: item.enabled for item in policies}
        id_by_source = {item.source_type: item.id for item in policies}
        definitions = {
            KnowledgeSourceType.activity_event: (
                "本地活动输入",
                "FatPet 采集并同步的本地文字活动。",
            ),
            KnowledgeSourceType.hourly_summary: (
                "DeepSeek 小时总结",
                "基于每小时活动生成的结构化工作总结。",
            ),
        }
        responses = []
        for source_type, (name, description) in definitions.items():
            items = [item for item in documents if item.source_type == source_type.value]
            responses.append(
                KnowledgeBaseResponse(
                    id=id_by_source.get(source_type.value, source_type.value),
                    source_type=source_type,
                    name=name,
                    description=description,
                    enabled=enabled_by_source.get(source_type.value, True),
                    total=len(items),
                    active=sum(item.status == "active" for item in items),
                    deleted=sum(item.status == "deleted" for item in items),
                    last_indexed_at=_utc_or_none(
                        max((item.updated_at for item in items), default=None)
                    ),
                )
            )
        return responses

    async def set_base_enabled(
        self, source_type: KnowledgeSourceType, enabled: bool
    ) -> KnowledgeBaseResponse:
        async with self._session_factory() as session:
            record = cast(
                KnowledgeBaseRecord | None,
                await session.scalar(
                    select(KnowledgeBaseRecord).where(
                        KnowledgeBaseRecord.workspace_id == self._workspace_id,
                        KnowledgeBaseRecord.source_type == source_type.value,
                    )
                ),
            )
            if record is None:
                record = KnowledgeBaseRecord(
                    id=str(uuid4()),
                    workspace_id=self._workspace_id,
                    source_type=source_type.value,
                    enabled=enabled,
                )
                session.add(record)
            else:
                record.enabled = enabled
                record.updated_at = utc_now()
            await session.commit()
        return next(
            item for item in await self.list_bases() if item.source_type == source_type
        )

    async def enabled_source_types(
        self, requested: list[KnowledgeSourceType] | None = None
    ) -> list[KnowledgeSourceType]:
        bases = await self.list_bases()
        allowed = {item.source_type for item in bases if item.enabled}
        candidates = requested or list(KnowledgeSourceType)
        return [item for item in candidates if item in allowed]

    async def get(self, document_id: str) -> KnowledgeDocumentRecord | None:
        async with self._session_factory() as session:
            record = await session.get(KnowledgeDocumentRecord, document_id)
            if record is None or record.workspace_id != self._workspace_id:
                return None
            return record

    async def update_status(
        self, document_id: str, status: KnowledgeDocumentStatus
    ) -> KnowledgeDocumentRecord | None:
        async with self._session_factory() as session:
            record = await session.get(KnowledgeDocumentRecord, document_id)
            if record is None or record.workspace_id != self._workspace_id:
                return None
            if record.status == KnowledgeDocumentStatus.deleted.value:
                return None
            record.status = status.value
            record.updated_at = utc_now()
            await session.commit()
            await session.refresh(record)
            return record

    async def batch_action(
        self, document_ids: list[str], action: KnowledgeBatchAction
    ) -> KnowledgeBatchResponse:
        target = {
            KnowledgeBatchAction.enable: KnowledgeDocumentStatus.active.value,
            KnowledgeBatchAction.disable: KnowledgeDocumentStatus.disabled.value,
            KnowledgeBatchAction.delete: KnowledgeDocumentStatus.deleted.value,
        }[action]
        async with self._session_factory() as session:
            records = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord).where(
                            KnowledgeDocumentRecord.workspace_id == self._workspace_id,
                            KnowledgeDocumentRecord.id.in_(document_ids),
                        )
                    )
                ).all()
            )
            updated = 0
            for record in records:
                if record.status == target or (
                    record.status == KnowledgeDocumentStatus.deleted.value
                    and action is not KnowledgeBatchAction.delete
                ):
                    continue
                record.status = target
                record.updated_at = utc_now()
                updated += 1
            await session.commit()
        return KnowledgeBatchResponse(
            matched=len(records), updated=updated, skipped=len(records) - updated
        )

    async def remove_by_sources(
        self, source_type: KnowledgeSourceType, source_ids: list[str]
    ) -> int:
        if not source_ids:
            return 0
        async with self._session_factory() as session:
            matched = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord.id).where(
                            KnowledgeDocumentRecord.workspace_id == self._workspace_id,
                            KnowledgeDocumentRecord.source_type == source_type.value,
                            KnowledgeDocumentRecord.source_id.in_(source_ids),
                        )
                    )
                ).all()
            )
            await session.execute(
                delete(KnowledgeDocumentRecord).where(
                    KnowledgeDocumentRecord.workspace_id == self._workspace_id,
                    KnowledgeDocumentRecord.source_type == source_type.value,
                    KnowledgeDocumentRecord.source_id.in_(source_ids),
                )
            )
            await session.commit()
            return len(matched)

    async def purge_summary_before(self, cutoff: datetime) -> int:
        async with self._session_factory() as session:
            filters = (
                KnowledgeDocumentRecord.workspace_id == self._workspace_id,
                KnowledgeDocumentRecord.source_type
                == KnowledgeSourceType.hourly_summary.value,
                KnowledgeDocumentRecord.occurred_at < cutoff,
            )
            matched = list(
                (
                    await session.scalars(
                        select(KnowledgeDocumentRecord.id).where(*filters)
                    )
                ).all()
            )
            await session.execute(
                delete(KnowledgeDocumentRecord).where(
                    *filters,
                )
            )
            await session.commit()
            return len(matched)

    def _filters(
        self,
        *,
        query: str | None,
        source_types: list[KnowledgeSourceType] | None,
        status: KnowledgeDocumentStatus | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = [
            KnowledgeDocumentRecord.workspace_id == self._workspace_id
        ]
        if query:
            filters.append(
                or_(
                    KnowledgeDocumentRecord.title.contains(query, autoescape=True),
                    KnowledgeDocumentRecord.content.contains(query, autoescape=True),
                    KnowledgeDocumentRecord.keywords_json.contains(query, autoescape=True),
                    KnowledgeDocumentRecord.source_app.contains(query, autoescape=True),
                )
            )
        if source_types:
            filters.append(
                KnowledgeDocumentRecord.source_type.in_([item.value for item in source_types])
            )
        if status:
            filters.append(KnowledgeDocumentRecord.status == status.value)
        if start:
            filters.append(KnowledgeDocumentRecord.occurred_at >= start)
        if end:
            filters.append(KnowledgeDocumentRecord.occurred_at <= end)
        return filters


def _json_list(values: Sequence[str]) -> str:
    import json

    return json.dumps(list(values), ensure_ascii=False)


def _utc_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _fts_query(query: str) -> str:
    normalized = " ".join(query.strip().split()).replace('"', '""')
    if len(normalized) < 3:
        return f'"{normalized}"'
    trigrams = list(
        dict.fromkeys(
            normalized[index : index + 3] for index in range(len(normalized) - 2)
        )
    )
    return " OR ".join(f'"{part}"' for part in trigrams[:32])
