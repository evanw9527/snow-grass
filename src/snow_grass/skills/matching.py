from __future__ import annotations

import re

METADATA_STOP_TERMS = {
    "使用",
    "用户",
    "当前",
    "任务",
    "完成",
    "支持",
    "提供",
    "通过",
    "信息",
    "相关",
}


def selection_score(
    *,
    skill_id: str,
    name: str,
    description: str,
    keywords: list[str],
    content: str,
) -> int:
    normalized = content.casefold()
    keyword_score = sum(
        20
        for keyword in keywords
        if keyword.strip() and keyword.casefold() in normalized
    )
    name_score = _metadata_match_score(
        f"{skill_id} {name}", normalized, weight=3
    )
    description_score = _metadata_match_score(description, normalized, weight=1)
    return keyword_score + name_score + description_score


def _metadata_match_score(metadata: str, normalized: str, *, weight: int) -> int:
    best = 0
    normalized_words = set(re.findall(r"[a-z0-9-]+", normalized))
    for word in re.findall(r"[a-z0-9-]+", metadata.casefold()):
        if len(word) >= 3 and word in normalized_words:
            best = max(best, min(len(word), 8) * weight)

    for chunk in re.findall(r"[\u3400-\u9fff]+", metadata):
        max_size = min(6, len(chunk))
        for size in range(2, max_size + 1):
            for start in range(len(chunk) - size + 1):
                term = chunk[start : start + size]
                if term in METADATA_STOP_TERMS:
                    continue
                if term in normalized:
                    best = max(best, size * weight)
    return best
