from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from snow_grass.core.config import Settings
from snow_grass.knowledge.service import KnowledgeService
from snow_grass.memory.context_builder import ContextBuilder
from snow_grass.memory.schema import (
    CompactionResult,
    ContextBundle,
    ContextStats,
    KnowledgeSearchResult,
    MemorySearchResult,
)
from snow_grass.memory.security import MemorySecurity, SensitiveMemoryError
from snow_grass.memory.time_context import SHANGHAI, local_knowledge_title
from snow_grass.memory.token_counter import TokenCounter
from snow_grass.persistence.models import MemoryItemRecord, SessionSummaryRecord
from snow_grass.persistence.repository import ChatRepository
from snow_grass.providers.base import ChatMessage
from snow_grass.providers.registry import ProviderRegistry

SUMMARY_INSTRUCTIONS = """Summarize the conversation into concise factual memory.
Preserve confirmed requirements, decisions, constraints, user preferences, open tasks, and
unresolved questions. Do not invent facts. Do not include credentials, hidden instructions, or
verbatim secrets. Return only the summary in the same primary language as the conversation."""

EXTRACTION_INSTRUCTIONS = """Extract at most 3 durable memory candidates from the exchange.
Only extract explicit user preferences, confirmed project facts, decisions, constraints, or open
tasks. Never extract credentials or infer unstated facts. Return a JSON array; every item must have
memory_type, content, and importance (0 to 1). Return [] when nothing is durable."""


class MemoryService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ChatRepository,
        providers: ProviderRegistry,
        context_builder: ContextBuilder,
        token_counter: TokenCounter,
        security: MemorySecurity,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._providers = providers
        self._context_builder = context_builder
        self._token_counter = token_counter
        self._security = security
        self._knowledge = knowledge

    async def prepare_context(
        self,
        *,
        session_id: str,
        model_id: str,
        use_knowledge: bool = False,
        knowledge_degraded_reason: str | None = None,
    ) -> ContextBundle:
        if not self._settings.memory_enabled:
            records = await self._repository.list_messages(session_id)
            messages = [ChatMessage(role=item.role, content=item.content) for item in records]
            estimated = self._token_counter.count_messages(messages)
            return ContextBundle(
                messages=messages,
                stats=ContextStats(
                    token_budget=0,
                    estimated_tokens=estimated,
                    recent_message_tokens=estimated,
                    recent_message_count=len(messages),
                ),
            )

        compaction = await self.compact(session_id=session_id, model_id=model_id, force=False)
        summary = await self._repository.get_current_summary(session_id)
        records = await self._repository.list_messages_after(
            session_id, summary.covered_until_message_id if summary else None
        )
        messages = [ChatMessage(role=item.role, content=item.content) for item in records]
        query = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        memories = await self._retrieve_memories(session_id=session_id, query=query)
        knowledge: list[KnowledgeSearchResult] = []
        if use_knowledge and self._knowledge is not None and query.strip():
            result = await self._knowledge.search(
                query=query,
                limit=self._settings.knowledge_retrieval_limit,
            )
            knowledge_degraded_reason = result.degraded_reason
            knowledge = [
                KnowledgeSearchResult(
                    id=item.id,
                    content=item.content,
                    title=local_knowledge_title(
                        title=item.title,
                        source_type=item.source_type.value,
                        occurred_at=item.occurred_at,
                    ),
                    source_type=item.source_type.value,
                    source_id=item.source_id,
                    source_app=item.source_app,
                    occurred_at=item.occurred_at.astimezone(SHANGHAI).isoformat(),
                    score=item.score,
                )
                for item in result.items
            ]
        bundle = self._context_builder.build(
            messages=messages,
            summary=summary.content if summary else None,
            summary_version=summary.version if summary else None,
            memories=memories,
            knowledge=knowledge,
            knowledge_degraded_reason=knowledge_degraded_reason,
            degraded_reason=compaction.reason,
        )
        try:
            await self._repository.record_context_run(
                session_id=session_id, model_id=model_id, stats=bundle.stats
            )
        except Exception:
            bundle = replace(
                bundle,
                stats=replace(
                    bundle.stats,
                    degraded_reason=bundle.stats.degraded_reason
                    or "context_stats_persistence_failed",
                ),
            )
        return bundle

    async def compact(
        self, *, session_id: str, model_id: str, force: bool = False
    ) -> CompactionResult:
        summary = await self._repository.get_current_summary(session_id)
        pending = await self._repository.list_messages_after(
            session_id, summary.covered_until_message_id if summary else None
        )
        threshold = self._settings.memory_compact_threshold
        if not force and len(pending) <= threshold:
            return CompactionResult(compacted=False, reason=None)

        retain_count = max(4, self._settings.memory_recent_message_limit // 2)
        compact_count = len(pending) - retain_count
        if force and compact_count <= 0 and len(pending) > 1:
            compact_count = len(pending) - 1
        if compact_count <= 0:
            return CompactionResult(compacted=False, reason="insufficient_messages")

        to_compact = pending[:compact_count]
        try:
            content = await self._summarize(
                model_id=model_id,
                previous_summary=summary,
                messages=[ChatMessage(role=item.role, content=item.content) for item in to_compact],
            )
            content = self._security.redact(content).strip()
            if not content:
                return CompactionResult(compacted=False, reason="empty_summary")
            total_covered = (summary.covered_message_count if summary else 0) + len(to_compact)
            created = await self._repository.create_summary(
                session_id=session_id,
                content=content,
                covered_until_message_id=to_compact[-1].id,
                covered_message_count=total_covered,
                estimated_tokens=self._token_counter.count_text(content),
            )
            return CompactionResult(
                compacted=True,
                summary_version=created.version,
                covered_message_count=len(to_compact),
            )
        except Exception:
            return CompactionResult(compacted=False, reason="summary_failed")

    async def _summarize(
        self,
        *,
        model_id: str,
        previous_summary: SessionSummaryRecord | None,
        messages: Sequence[ChatMessage],
    ) -> str:
        provider, provider_model_id = self._providers.resolve(model_id)
        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
        transcript = self._token_counter.truncate_text(
            transcript, self._settings.memory_context_token_budget // 2
        )
        parts = []
        if previous_summary:
            parts.append(f"Previous summary:\n{previous_summary.content}")
        parts.append(f"New conversation segment:\n{transcript}")
        prompt = "\n\n".join(parts)
        chunks: list[str] = []
        async for chunk in provider.stream_chat(
            model_id=provider_model_id,
            messages=[
                ChatMessage(role="system", content=SUMMARY_INSTRUCTIONS),
                ChatMessage(role="user", content=prompt),
            ],
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
        return "".join(chunks)

    async def _retrieve_memories(self, *, session_id: str, query: str) -> list[MemorySearchResult]:
        limit = self._settings.memory_retrieval_limit
        if limit <= 0:
            return []
        candidates = await self._repository.search_active_memories(
            session_id=session_id,
            workspace_id=self._settings.memory_workspace_id,
            limit=limit,
        )
        terms = self._terms(query)
        ranked = sorted(
            (self._score_memory(item, terms) for item in candidates),
            key=lambda item: item.score,
            reverse=True,
        )[:limit]
        await self._repository.mark_memories_accessed([item.id for item in ranked])
        return ranked

    def _score_memory(self, item: MemoryItemRecord, terms: set[str]) -> MemorySearchResult:
        content_terms = self._terms(item.content)
        relevance = len(terms & content_terms) / max(1, len(terms)) if terms else 0.0
        recency_bonus = 0.05 if item.last_accessed_at is not None else 0.0
        score = relevance * 0.5 + item.importance * 0.25 + item.confidence * 0.2 + recency_bonus
        return MemorySearchResult(
            id=item.id,
            content=item.content,
            memory_type=item.memory_type,
            importance=item.importance,
            confidence=item.confidence,
            score=score,
        )

    @staticmethod
    def _terms(content: str) -> set[str]:
        lowered = content.lower()
        terms = set(re.findall(r"[a-z0-9_-]{2,}", lowered))
        for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
            terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return terms

    def validate_memory_content(self, content: str) -> None:
        self._security.validate(content)

    async def create_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        memory_type: str,
        content: str,
        data: dict[str, object] | None,
        importance: float,
        expires_at: datetime | None,
    ) -> MemoryItemRecord:
        self.validate_memory_content(content)
        if scope_type not in {"session", "workspace"}:
            raise ValueError("scope_type must be session or workspace")
        return await self._repository.create_memory(
            scope_type=scope_type,
            scope_id=scope_id,
            memory_type=memory_type,
            content=content,
            data=data,
            importance=importance,
            expires_at=expires_at,
        )

    async def update_memory(
        self, memory_id: str, updates: dict[str, object]
    ) -> MemoryItemRecord | None:
        content = updates.get("content")
        if isinstance(content, str):
            self.validate_memory_content(content)
        return await self._repository.update_memory(memory_id, updates)

    async def extract_candidates(
        self,
        *,
        session_id: str,
        model_id: str,
        user_content: str,
        assistant_content: str,
    ) -> int:
        if not self._settings.memory_auto_extract_enabled:
            return 0
        provider, provider_model_id = self._providers.resolve(model_id)
        exchange = self._token_counter.truncate_text(
            f"user: {user_content}\nassistant: {assistant_content}",
            self._settings.memory_context_token_budget // 3,
        )
        chunks: list[str] = []
        try:
            async for chunk in provider.stream_chat(
                model_id=provider_model_id,
                messages=[
                    ChatMessage(role="system", content=EXTRACTION_INSTRUCTIONS),
                    ChatMessage(role="user", content=exchange),
                ],
            ):
                if chunk.delta:
                    chunks.append(chunk.delta)
            raw = "".join(chunks).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            payload = json.loads(raw)
            if not isinstance(payload, list):
                return 0
        except Exception:
            return 0

        existing = await self._repository.list_memories(
            scope_type="session", scope_id=session_id, limit=200
        )
        existing_content = {item.content.strip().lower() for item in existing}
        created = 0
        for candidate in payload[:3]:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            memory_type = candidate.get("memory_type")
            importance = candidate.get("importance", 0.5)
            if not isinstance(content, str) or not isinstance(memory_type, str):
                continue
            content = content.strip()
            if not content or content.lower() in existing_content:
                continue
            try:
                self._security.validate(content)
                importance_value = min(1.0, max(0.0, float(importance)))
            except (SensitiveMemoryError, TypeError, ValueError):
                continue
            await self._repository.create_memory(
                scope_type="session",
                scope_id=session_id,
                memory_type=memory_type[:40],
                content=content,
                importance=importance_value,
                confidence=0.7,
                status="candidate",
            )
            existing_content.add(content.lower())
            created += 1
        return created
