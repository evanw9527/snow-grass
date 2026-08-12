from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import and_, distinct, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from snow_grass.memory.schema import ContextStats
from snow_grass.persistence.models import (
    ContextRunRecord,
    MemoryItemRecord,
    MessageRecord,
    SessionRecord,
    SessionSummaryRecord,
    ToolResultCacheRecord,
    utc_now,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class ModelUsageSummary:
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int


@dataclass(frozen=True, slots=True)
class UsageSummary:
    period_days: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    sessions: int
    by_model: list[ModelUsageSummary]
    daily: list[dict[str, int | str]]


class ToolResultCacheRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_by_key(self, cache_key: str) -> ToolResultCacheRecord | None:
        async with self._session_factory() as session:
            return cast(
                ToolResultCacheRecord | None,
                await session.scalar(
                    select(ToolResultCacheRecord).where(
                        ToolResultCacheRecord.cache_key == cache_key
                    )
                ),
            )

    async def mark_hit(self, record_id: str) -> ToolResultCacheRecord | None:
        async with self._session_factory() as session:
            record = await session.get(ToolResultCacheRecord, record_id)
            if record is None:
                return None
            record.hit_count += 1
            record.last_hit_at = utc_now()
            await session.commit()
            await session.refresh(record)
            return record

    async def upsert(
        self,
        *,
        workspace_id: str,
        session_id: str | None,
        scope_type: str,
        scope_id: str,
        skill_id: str,
        skill_version: str,
        script: str,
        arguments_json: list[str],
        cache_key: str,
        result_json: dict[str, object],
        status: str,
        expires_at: datetime,
    ) -> ToolResultCacheRecord:
        async with self._session_factory() as session:
            record = await session.scalar(
                select(ToolResultCacheRecord).where(
                    ToolResultCacheRecord.cache_key == cache_key
                )
            )
            if record is None:
                record = ToolResultCacheRecord(
                    id=str(uuid4()),
                    workspace_id=workspace_id,
                    session_id=session_id,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    skill_id=skill_id,
                    skill_version=skill_version,
                    script=script,
                    arguments_json=arguments_json,
                    cache_key=cache_key,
                    result_json=result_json,
                    status=status,
                    expires_at=expires_at,
                )
                session.add(record)
            else:
                record.session_id = session_id
                record.arguments_json = arguments_json
                record.result_json = result_json
                record.status = status
                record.expires_at = expires_at
                record.hit_count = 0
                record.last_hit_at = None
                record.created_at = utc_now()
                record.updated_at = utc_now()
            await session.commit()
            await session.refresh(record)
            return record


class ChatRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_session(
        self,
        *,
        title: str,
        model_id: str,
        skill_id: str | None = None,
        knowledge_enabled: bool = False,
    ) -> SessionRecord:
        record = SessionRecord(
            id=str(uuid4()),
            title=title,
            model_id=model_id,
            skill_id=skill_id,
            knowledge_enabled=knowledge_enabled,
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self._session_factory() as session:
            return await session.get(SessionRecord, session_id)

    async def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        statement = select(SessionRecord).order_by(SessionRecord.updated_at.desc()).limit(limit)
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def update_session_knowledge(
        self, session_id: str, enabled: bool
    ) -> SessionRecord | None:
        async with self._session_factory() as session:
            record = await session.get(SessionRecord, session_id)
            if record is None:
                return None
            record.knowledge_enabled = enabled
            record.updated_at = utc_now()
            await session.commit()
            await session.refresh(record)
            return record

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        model_id: str | None = None,
        skill_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> MessageRecord:
        record = MessageRecord(
            id=str(uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            model_id=model_id,
            skill_id=skill_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        async with self._session_factory() as session:
            session.add(record)
            parent = await session.get(SessionRecord, session_id)
            if parent is None:
                raise LookupError(f"Session {session_id!r} does not exist")
            parent.updated_at = utc_now()
            await session.commit()
            await session.refresh(record)
        return record

    async def list_messages(self, session_id: str) -> list[MessageRecord]:
        statement = (
            select(MessageRecord)
            .where(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.created_at.asc())
        )
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def count_messages(self, session_id: str) -> int:
        statement = select(func.count(MessageRecord.id)).where(
            MessageRecord.session_id == session_id
        )
        async with self._session_factory() as session:
            return int((await session.scalar(statement)) or 0)

    async def get_current_summary(self, session_id: str) -> SessionSummaryRecord | None:
        statement = select(SessionSummaryRecord).where(
            SessionSummaryRecord.session_id == session_id,
            SessionSummaryRecord.is_current.is_(True),
        )
        async with self._session_factory() as session:
            return (await session.scalars(statement)).first()

    async def list_messages_after(
        self, session_id: str, after_message_id: str | None
    ) -> list[MessageRecord]:
        statement = select(MessageRecord).where(MessageRecord.session_id == session_id)
        async with self._session_factory() as session:
            if after_message_id:
                boundary = await session.get(MessageRecord, after_message_id)
                if boundary is not None and boundary.session_id == session_id:
                    statement = statement.where(
                        or_(
                            MessageRecord.created_at > boundary.created_at,
                            and_(
                                MessageRecord.created_at == boundary.created_at,
                                MessageRecord.id > boundary.id,
                            ),
                        )
                    )
            statement = statement.order_by(MessageRecord.created_at.asc(), MessageRecord.id.asc())
            return list((await session.scalars(statement)).all())

    async def create_summary(
        self,
        *,
        session_id: str,
        content: str,
        covered_until_message_id: str,
        covered_message_count: int,
        estimated_tokens: int,
    ) -> SessionSummaryRecord:
        async with self._session_factory() as session:
            current_statement = select(SessionSummaryRecord).where(
                SessionSummaryRecord.session_id == session_id,
                SessionSummaryRecord.is_current.is_(True),
            )
            current = await session.scalar(current_statement)
            if current is not None:
                current.is_current = False
            version_statement = select(func.max(SessionSummaryRecord.version)).where(
                SessionSummaryRecord.session_id == session_id
            )
            version = int((await session.scalar(version_statement)) or 0) + 1
            record = SessionSummaryRecord(
                id=str(uuid4()),
                session_id=session_id,
                version=version,
                content=content,
                covered_until_message_id=covered_until_message_id,
                covered_message_count=covered_message_count,
                estimated_tokens=estimated_tokens,
                is_current=True,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def memory_counts(self, session_id: str) -> tuple[int, int]:
        total = await self.count_messages(session_id)
        summary = await self.get_current_summary(session_id)
        cold = min(total, summary.covered_message_count if summary else 0)
        return total - cold, cold

    async def create_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        memory_type: str,
        content: str,
        data: dict[str, object] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        status: str = "active",
        expires_at: datetime | None = None,
    ) -> MemoryItemRecord:
        record = MemoryItemRecord(
            id=str(uuid4()),
            scope_type=scope_type,
            scope_id=scope_id,
            memory_type=memory_type,
            content=content,
            data=data,
            importance=importance,
            confidence=confidence,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            status=status,
            expires_at=expires_at,
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_memory(self, memory_id: str) -> MemoryItemRecord | None:
        async with self._session_factory() as session:
            return await session.get(MemoryItemRecord, memory_id)

    async def list_memories(
        self,
        *,
        scope_type: str,
        scope_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItemRecord]:
        statement = select(MemoryItemRecord).where(
            MemoryItemRecord.scope_type == scope_type,
            MemoryItemRecord.scope_id == scope_id,
        )
        if status:
            statement = statement.where(MemoryItemRecord.status == status)
        else:
            statement = statement.where(MemoryItemRecord.status != "deleted")
        statement = statement.order_by(MemoryItemRecord.updated_at.desc()).limit(limit)
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def update_memory(
        self, memory_id: str, updates: dict[str, Any]
    ) -> MemoryItemRecord | None:
        async with self._session_factory() as session:
            record = await session.get(MemoryItemRecord, memory_id)
            if record is None:
                return None
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = utc_now()
            await session.commit()
            await session.refresh(record)
            return record

    async def delete_memory(self, memory_id: str) -> bool:
        record = await self.update_memory(memory_id, {"status": "deleted", "deleted_at": utc_now()})
        return record is not None

    async def search_active_memories(
        self,
        *,
        session_id: str,
        workspace_id: str,
        limit: int,
    ) -> list[MemoryItemRecord]:
        now = utc_now()
        statement = (
            select(MemoryItemRecord)
            .where(
                or_(
                    and_(
                        MemoryItemRecord.scope_type == "session",
                        MemoryItemRecord.scope_id == session_id,
                    ),
                    and_(
                        MemoryItemRecord.scope_type == "workspace",
                        MemoryItemRecord.scope_id == workspace_id,
                    ),
                ),
                MemoryItemRecord.status == "active",
                MemoryItemRecord.deleted_at.is_(None),
                or_(MemoryItemRecord.expires_at.is_(None), MemoryItemRecord.expires_at > now),
            )
            .order_by(
                MemoryItemRecord.importance.desc(),
                MemoryItemRecord.confidence.desc(),
                MemoryItemRecord.updated_at.desc(),
            )
            .limit(max(limit * 4, limit))
        )
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def mark_memories_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        async with self._session_factory() as session:
            records = list(
                (
                    await session.scalars(
                        select(MemoryItemRecord).where(MemoryItemRecord.id.in_(memory_ids))
                    )
                ).all()
            )
            now = utc_now()
            for record in records:
                record.access_count += 1
                record.last_accessed_at = now
            await session.commit()

    async def record_context_run(
        self, *, session_id: str, model_id: str, stats: ContextStats
    ) -> ContextRunRecord:
        record = ContextRunRecord(
            id=str(uuid4()),
            session_id=session_id,
            model_id=model_id,
            **stats.to_dict(),
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_last_context_run(self, session_id: str) -> ContextRunRecord | None:
        statement = (
            select(ContextRunRecord)
            .where(ContextRunRecord.session_id == session_id)
            .order_by(ContextRunRecord.created_at.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            return (await session.scalars(statement)).first()

    async def usage_summary(self, days: int = 30) -> UsageSummary:
        first_day = datetime.now(SHANGHAI).date() - timedelta(days=days - 1)
        cutoff = datetime.combine(first_day, datetime.min.time(), tzinfo=SHANGHAI).astimezone(UTC)
        filters = (
            MessageRecord.role == "assistant",
            MessageRecord.total_tokens > 0,
            MessageRecord.created_at >= cutoff,
        )
        totals_statement = select(
            func.coalesce(func.sum(MessageRecord.input_tokens), 0),
            func.coalesce(func.sum(MessageRecord.output_tokens), 0),
            func.coalesce(func.sum(MessageRecord.total_tokens), 0),
            func.count(MessageRecord.id),
            func.count(distinct(MessageRecord.session_id)),
        ).where(*filters)
        model_name = func.coalesce(MessageRecord.model_id, "unknown")
        by_model_statement = (
            select(
                model_name,
                func.coalesce(func.sum(MessageRecord.input_tokens), 0),
                func.coalesce(func.sum(MessageRecord.output_tokens), 0),
                func.coalesce(func.sum(MessageRecord.total_tokens), 0),
                func.count(MessageRecord.id),
            )
            .where(*filters)
            .group_by(model_name)
            .order_by(func.sum(MessageRecord.total_tokens).desc())
        )
        daily_statement = select(MessageRecord.created_at, MessageRecord.total_tokens).where(
            *filters
        )
        async with self._session_factory() as session:
            totals = (await session.execute(totals_statement)).one()
            model_rows = (await session.execute(by_model_statement)).all()
            daily_rows = (await session.execute(daily_statement)).all()
        by_day: dict[str, int] = defaultdict(int)
        for created_at, total_tokens in daily_rows:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            day = created_at.astimezone(SHANGHAI).date().isoformat()
            by_day[day] += int(total_tokens or 0)
        return UsageSummary(
            period_days=days,
            input_tokens=int(totals[0] or 0),
            output_tokens=int(totals[1] or 0),
            total_tokens=int(totals[2] or 0),
            requests=int(totals[3] or 0),
            sessions=int(totals[4] or 0),
            by_model=[
                ModelUsageSummary(
                    model_id=str(row[0]),
                    input_tokens=int(row[1] or 0),
                    output_tokens=int(row[2] or 0),
                    total_tokens=int(row[3] or 0),
                    requests=int(row[4] or 0),
                )
                for row in model_rows
            ],
            daily=[{"date": day, "total_tokens": tokens} for day, tokens in sorted(by_day.items())],
        )
