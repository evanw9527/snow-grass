from __future__ import annotations

import re


class SensitiveMemoryError(ValueError):
    pass


class MemorySecurity:
    _PATTERNS = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]{8,}",
            re.IGNORECASE,
        ),
    )

    def contains_sensitive(self, content: str) -> bool:
        return any(pattern.search(content) for pattern in self._PATTERNS)

    def validate(self, content: str) -> None:
        if self.contains_sensitive(content):
            raise SensitiveMemoryError("Memory content contains sensitive credentials")

    def redact(self, content: str) -> str:
        redacted = content
        for pattern in self._PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
