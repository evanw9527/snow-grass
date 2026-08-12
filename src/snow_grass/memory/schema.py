from __future__ import annotations

from dataclasses import dataclass, field

from snow_grass.providers.base import ChatMessage


@dataclass(frozen=True, slots=True)
class ContextStats:
    token_budget: int
    estimated_tokens: int
    summary_tokens: int = 0
    memory_tokens: int = 0
    recent_message_tokens: int = 0
    recent_message_count: int = 0
    memory_count: int = 0
    knowledge_tokens: int = 0
    knowledge_count: int = 0
    knowledge_degraded_reason: str | None = None
    trimmed_message_count: int = 0
    summary_version: int | None = None
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "token_budget": self.token_budget,
            "estimated_tokens": self.estimated_tokens,
            "summary_tokens": self.summary_tokens,
            "memory_tokens": self.memory_tokens,
            "recent_message_tokens": self.recent_message_tokens,
            "recent_message_count": self.recent_message_count,
            "memory_count": self.memory_count,
            "knowledge_tokens": self.knowledge_tokens,
            "knowledge_count": self.knowledge_count,
            "knowledge_degraded_reason": self.knowledge_degraded_reason,
            "trimmed_message_count": self.trimmed_message_count,
            "summary_version": self.summary_version,
            "degraded_reason": self.degraded_reason,
        }


@dataclass(frozen=True, slots=True)
class ContextBundle:
    messages: list[ChatMessage]
    system_memory: str = ""
    stats: ContextStats = field(
        default_factory=lambda: ContextStats(token_budget=0, estimated_tokens=0)
    )


@dataclass(frozen=True, slots=True)
class CompactionResult:
    compacted: bool
    summary_version: int | None = None
    covered_message_count: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    id: str
    content: str
    memory_type: str
    importance: float
    confidence: float
    score: float


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    id: str
    content: str
    title: str
    source_type: str
    source_id: str
    source_app: str | None
    occurred_at: str
    score: float
