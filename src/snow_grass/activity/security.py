from __future__ import annotations

import re


class ActivitySecurity:
    """Activity-owned text sanitizer; intentionally independent from Memory."""

    _patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(
            r"(?i)\b(api[_ -]?key|access[_ -]?token|password|passwd)\s*[:=]\s*\S+"
        ),
    )

    def sanitize(self, text: str) -> str:
        sanitized = text
        for pattern in self._patterns:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
