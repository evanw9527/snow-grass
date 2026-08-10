from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import Protocol

from snow_grass.providers.base import ChatMessage


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[ChatMessage]) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...


class ConservativeTokenCounter:
    """Provider-neutral estimate; provider usage remains the source of truth after a call."""

    _MESSAGE_OVERHEAD = 6

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, ceil(len(text.encode("utf-8")) / 3))

    def count_messages(self, messages: Sequence[ChatMessage]) -> int:
        return sum(
            self._MESSAGE_OVERHEAD
            + self.count_text(message.role)
            + self.count_text(message.content)
            for message in messages
        )

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.count_text(text) <= max_tokens:
            return text
        marker = "\n…[内容已按上下文预算截断]…\n"
        available_bytes = max(0, max_tokens * 3 - len(marker.encode("utf-8")))
        encoded = text.encode("utf-8")
        head_size = available_bytes // 2
        tail_size = available_bytes - head_size
        head = encoded[:head_size].decode("utf-8", errors="ignore")
        tail = encoded[-tail_size:].decode("utf-8", errors="ignore") if tail_size else ""
        return f"{head}{marker}{tail}"
