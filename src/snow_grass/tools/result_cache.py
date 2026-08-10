from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from snow_grass.persistence.models import ToolResultCacheRecord, utc_now
from snow_grass.persistence.repository import ToolResultCacheRepository
from snow_grass.skills.schema import LoadedSkill

CacheStatus = Literal["hit", "miss", "expired", "refresh", "disabled"]
CachePolicy = Literal["prefer-cache", "refresh", "no-store"]


@dataclass(frozen=True, slots=True)
class CacheDecision:
    status: CacheStatus
    cache_key: str | None = None
    scope_type: str | None = None
    scope_id: str | None = None
    normalized_args: list[str] | None = None
    record: ToolResultCacheRecord | None = None

    @property
    def result_json(self) -> dict[str, object] | None:
        return self.record.result_json if self.record is not None else None


class ToolResultReuseService:
    def __init__(
        self, *, repository: ToolResultCacheRepository, workspace_id: str
    ) -> None:
        self._repository = repository
        self._workspace_id = workspace_id

    async def resolve(
        self,
        *,
        skill: LoadedSkill,
        script: str,
        args: list[str],
        session_id: str | None,
        run_id: str,
        policy: CachePolicy,
        force_refresh: bool,
    ) -> CacheDecision:
        config = skill.manifest.cache
        if not config.enabled or policy == "no-store":
            return CacheDecision(status="disabled")

        scope_id = self._scope_id(
            scope=config.scope,
            session_id=session_id,
            run_id=run_id,
        )
        if scope_id is None:
            return CacheDecision(status="disabled")
        normalized_args = [self._normalize_arg(skill.manifest.id, value) for value in args]
        cache_key = self._cache_key(
            scope_type=config.scope,
            scope_id=scope_id,
            skill=skill,
            script=script,
            args=normalized_args,
        )
        base = CacheDecision(
            status="refresh" if force_refresh or policy == "refresh" else "miss",
            cache_key=cache_key,
            scope_type=config.scope,
            scope_id=scope_id,
            normalized_args=normalized_args,
        )
        if base.status == "refresh":
            return base

        try:
            record = await self._repository.get_by_key(cache_key)
        except Exception:
            return CacheDecision(status="disabled")
        if record is None:
            return base
        if self._as_utc(record.expires_at) <= utc_now():
            return CacheDecision(
                status="expired",
                cache_key=cache_key,
                scope_type=config.scope,
                scope_id=scope_id,
                normalized_args=normalized_args,
                record=record,
            )
        try:
            record = await self._repository.mark_hit(record.id) or record
        except Exception:
            pass
        return CacheDecision(
            status="hit",
            cache_key=cache_key,
            scope_type=config.scope,
            scope_id=scope_id,
            normalized_args=normalized_args,
            record=record,
        )

    async def save(
        self,
        *,
        decision: CacheDecision,
        skill: LoadedSkill,
        script: str,
        session_id: str | None,
        result_json: dict[str, object],
        ok: bool,
    ) -> ToolResultCacheRecord | None:
        if decision.cache_key is None or decision.scope_type is None or decision.scope_id is None:
            return None
        ttl_seconds = (
            skill.manifest.cache.ttl_seconds
            if ok
            else skill.manifest.cache.error_ttl_seconds
        )
        if ttl_seconds <= 0:
            return None
        try:
            return await self._repository.upsert(
                workspace_id=self._workspace_id,
                session_id=session_id,
                scope_type=decision.scope_type,
                scope_id=decision.scope_id,
                skill_id=skill.manifest.id,
                skill_version=skill.manifest.version,
                script=script,
                arguments_json=decision.normalized_args or [],
                cache_key=decision.cache_key,
                result_json=result_json,
                status="success" if ok else "error",
                expires_at=utc_now() + timedelta(seconds=ttl_seconds),
            )
        except Exception:
            return None

    @classmethod
    def event_data(
        cls, decision: CacheDecision, record: ToolResultCacheRecord | None = None
    ) -> dict[str, object]:
        current = record or decision.record
        data: dict[str, object] = {"cache_status": decision.status}
        if decision.cache_key:
            data["cache_key"] = decision.cache_key
        if current is None:
            return data
        created_at = cls._as_utc(current.created_at)
        expires_at = cls._as_utc(current.expires_at)
        data.update(
            {
                "cached_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "age_ms": max(0, round((utc_now() - created_at).total_seconds() * 1_000)),
                "hit_count": current.hit_count,
            }
        )
        return data

    def _scope_id(
        self, *, scope: str, session_id: str | None, run_id: str
    ) -> str | None:
        if scope == "workspace":
            return self._workspace_id
        if scope == "session":
            return session_id
        return run_id

    def _cache_key(
        self,
        *,
        scope_type: str,
        scope_id: str,
        skill: LoadedSkill,
        script: str,
        args: list[str],
    ) -> str:
        serialized = json.dumps(
            {
                "scope_type": scope_type,
                "scope_id": scope_id,
                "skill_id": skill.manifest.id,
                "skill_version": skill.manifest.version,
                "script": script,
                "args": args,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_arg(skill_id: str, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip()
        if normalized.isascii():
            normalized = normalized.casefold()
        if skill_id == "weather" and len(normalized) > 2 and normalized.endswith("市"):
            normalized = normalized[:-1]
        return normalized

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
