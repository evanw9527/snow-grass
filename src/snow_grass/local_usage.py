from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from snow_grass.persistence.models import CodexUsageCursorRecord, CodexUsageEventRecord

SHANGHAI = ZoneInfo("Asia/Shanghai")
EVENT_INSERT_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class CodexUsage:
    total_tokens: int
    sessions: int
    daily: list[dict[str, int | str]]
    models: list[dict[str, int | str]]
    projects: list[dict[str, int | str]]


@dataclass(frozen=True, slots=True)
class CodexUsageCursor:
    source_path: str
    thread_id: str
    byte_offset: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    current_model: str | None
    project_path: str


@dataclass(frozen=True, slots=True)
class CodexUsageEvent:
    id: str
    thread_id: str
    source_path: str
    occurred_at: datetime
    model_id: str
    project_path: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class CodexImportBatch:
    available: bool
    events: list[CodexUsageEvent]
    cursors: list[CodexUsageCursor]


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _nonnegative_delta(current: int, previous: int) -> int:
    return current - previous if current >= previous else current


def _read_thread_sources(database_path: Path) -> list[sqlite3.Row] | None:
    if not database_path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, model, cwd, rollout_path
            FROM threads
            WHERE rollout_path IS NOT NULL AND rollout_path != ''
            """
        ).fetchall()
        connection.close()
    except (sqlite3.Error, OSError):
        return None
    return rows


def scan_codex_usage(database_path: Path, cursors: dict[str, CodexUsageCursor]) -> CodexImportBatch:
    """Read only complete, newly appended Codex JSONL events."""
    sources = _read_thread_sources(database_path)
    if sources is None:
        return CodexImportBatch(available=False, events=[], cursors=[])

    events: list[CodexUsageEvent] = []
    updated_cursors: list[CodexUsageCursor] = []
    for source in sources:
        source_path = str(source["rollout_path"])
        rollout = Path(source_path)
        if not rollout.is_file():
            continue

        thread_id = str(source["id"] or "")
        project_path = str(source["cwd"] or "")
        fallback_model = str(source["model"] or "未记录模型")
        prior = cursors.get(source_path)
        file_size = rollout.stat().st_size
        offset = prior.byte_offset if prior and prior.byte_offset <= file_size else 0
        previous = {
            "input_tokens": prior.input_tokens if prior and offset else 0,
            "cached_input_tokens": prior.cached_input_tokens if prior and offset else 0,
            "output_tokens": prior.output_tokens if prior and offset else 0,
            "reasoning_output_tokens": prior.reasoning_output_tokens if prior and offset else 0,
            "total_tokens": prior.total_tokens if prior and offset else 0,
        }
        current_model = prior.current_model if prior and offset else fallback_model
        safe_offset = offset

        try:
            with rollout.open("rb") as handle:
                handle.seek(offset)
                while True:
                    line_start = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        break
                    safe_offset = handle.tell()
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if item.get("type") == "turn_context":
                        model = payload.get("model")
                        if isinstance(model, str) and model:
                            current_model = model
                        continue
                    if item.get("type") != "event_msg" or payload.get("type") != "token_count":
                        continue
                    info = payload.get("info")
                    totals = info.get("total_token_usage") if isinstance(info, dict) else None
                    if not isinstance(totals, dict):
                        continue
                    current = {key: max(0, int(totals.get(key) or 0)) for key in previous}
                    delta = {
                        key: _nonnegative_delta(current[key], previous[key]) for key in previous
                    }
                    previous = current
                    occurred_at = _parse_timestamp(item.get("timestamp"))
                    if occurred_at is None or delta["total_tokens"] <= 0:
                        continue
                    event_key = (
                        f"{source_path}\0{line_start}\0{item.get('timestamp')}\0"
                        f"{current['total_tokens']}"
                    )
                    events.append(
                        CodexUsageEvent(
                            id=hashlib.sha256(event_key.encode()).hexdigest(),
                            thread_id=thread_id,
                            source_path=source_path,
                            occurred_at=occurred_at,
                            model_id=current_model or fallback_model,
                            project_path=project_path,
                            input_tokens=delta["input_tokens"],
                            cached_input_tokens=delta["cached_input_tokens"],
                            output_tokens=delta["output_tokens"],
                            reasoning_output_tokens=delta["reasoning_output_tokens"],
                            total_tokens=delta["total_tokens"],
                        )
                    )
        except OSError:
            continue

        updated_cursors.append(
            CodexUsageCursor(
                source_path=source_path,
                thread_id=thread_id,
                byte_offset=safe_offset,
                input_tokens=previous["input_tokens"],
                cached_input_tokens=previous["cached_input_tokens"],
                output_tokens=previous["output_tokens"],
                reasoning_output_tokens=previous["reasoning_output_tokens"],
                total_tokens=previous["total_tokens"],
                current_model=current_model,
                project_path=project_path,
            )
        )
    return CodexImportBatch(available=True, events=events, cursors=updated_cursors)


class CodexUsageService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        database_path: Path,
    ) -> None:
        self._session_factory = session_factory
        self._database_path = database_path
        self._sync_lock = asyncio.Lock()

    async def _sync(self) -> bool:
        async with self._sync_lock:
            async with self._session_factory() as session:
                records = (await session.scalars(select(CodexUsageCursorRecord))).all()
                cursors = {
                    record.source_path: CodexUsageCursor(
                        source_path=record.source_path,
                        thread_id=record.thread_id,
                        byte_offset=record.byte_offset,
                        input_tokens=record.input_tokens,
                        cached_input_tokens=record.cached_input_tokens,
                        output_tokens=record.output_tokens,
                        reasoning_output_tokens=record.reasoning_output_tokens,
                        total_tokens=record.total_tokens,
                        current_model=record.current_model,
                        project_path=record.project_path,
                    )
                    for record in records
                }
            batch = await asyncio.to_thread(scan_codex_usage, self._database_path, cursors)
            if not batch.available:
                return False

            async with self._session_factory() as session, session.begin():
                if batch.events:
                    for offset in range(0, len(batch.events), EVENT_INSERT_BATCH_SIZE):
                        event_batch = batch.events[offset : offset + EVENT_INSERT_BATCH_SIZE]
                        await session.execute(
                            sqlite_insert(CodexUsageEventRecord)
                            .values(
                                [
                                    {
                                        "id": event.id,
                                        "thread_id": event.thread_id,
                                        "source_path": event.source_path,
                                        "occurred_at": event.occurred_at,
                                        "model_id": event.model_id,
                                        "project_path": event.project_path,
                                        "input_tokens": event.input_tokens,
                                        "cached_input_tokens": event.cached_input_tokens,
                                        "output_tokens": event.output_tokens,
                                        "reasoning_output_tokens": event.reasoning_output_tokens,
                                        "total_tokens": event.total_tokens,
                                    }
                                    for event in event_batch
                                ]
                            )
                            .on_conflict_do_nothing(index_elements=["id"])
                        )
                for cursor in batch.cursors:
                    record = await session.get(CodexUsageCursorRecord, cursor.source_path)
                    if record is None:
                        record = CodexUsageCursorRecord(source_path=cursor.source_path)
                        session.add(record)
                    record.thread_id = cursor.thread_id
                    record.byte_offset = cursor.byte_offset
                    record.input_tokens = cursor.input_tokens
                    record.cached_input_tokens = cursor.cached_input_tokens
                    record.output_tokens = cursor.output_tokens
                    record.reasoning_output_tokens = cursor.reasoning_output_tokens
                    record.total_tokens = cursor.total_tokens
                    record.current_model = cursor.current_model
                    record.project_path = cursor.project_path
            return True

    async def summary(self, *, days: int) -> CodexUsage | None:
        if not await self._sync():
            return None
        first_day = datetime.now(SHANGHAI).date() - timedelta(days=days - 1)
        cutoff = datetime.combine(first_day, datetime.min.time(), tzinfo=SHANGHAI).astimezone(UTC)
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(CodexUsageEventRecord).where(CodexUsageEventRecord.occurred_at >= cutoff)
                )
            ).all()

        by_day: dict[str, int] = defaultdict(int)
        by_model: dict[str, tuple[int, set[str]]] = defaultdict(lambda: (0, set()))
        by_project: dict[tuple[str, str], tuple[int, set[str]]] = defaultdict(lambda: (0, set()))
        for row in rows:
            tokens = row.total_tokens
            model = row.model_id
            cwd = row.project_path
            project_name = Path(cwd).name if cwd else "未记录项目"
            occurred_at = row.occurred_at
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=UTC)
            day = occurred_at.astimezone(SHANGHAI).date().isoformat()
            by_day[day] += tokens
            model_tokens, model_threads = by_model[model]
            model_threads.add(row.thread_id)
            by_model[model] = (model_tokens + tokens, model_threads)
            project_tokens, project_threads = by_project[(project_name, cwd)]
            project_threads.add(row.thread_id)
            by_project[(project_name, cwd)] = (project_tokens + tokens, project_threads)

        return CodexUsage(
            total_tokens=sum(row.total_tokens for row in rows),
            sessions=len({row.thread_id for row in rows}),
            daily=[{"date": day, "total_tokens": tokens} for day, tokens in sorted(by_day.items())],
            models=[
                {"name": name, "total_tokens": values[0], "sessions": len(values[1])}
                for name, values in sorted(
                    by_model.items(), key=lambda item: item[1][0], reverse=True
                )
            ],
            projects=[
                {
                    "name": name,
                    "path": path,
                    "total_tokens": values[0],
                    "sessions": len(values[1]),
                }
                for (name, path), values in sorted(
                    by_project.items(), key=lambda item: item[1][0], reverse=True
                )[:8]
            ],
        )
