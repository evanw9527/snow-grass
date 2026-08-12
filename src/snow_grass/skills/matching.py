from __future__ import annotations

import re
from collections.abc import Sequence

from snow_grass.skills.schema import SkillIntentRule

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
    intent_rules: Sequence[SkillIntentRule] = (),
    content: str,
) -> int:
    normalized = content.casefold()
    keyword_score = sum(
        20 for keyword in keywords if keyword.strip() and keyword.casefold() in normalized
    )
    name_score = _metadata_match_score(f"{skill_id} {name}", normalized, weight=3)
    description_score = _metadata_match_score(description, normalized, weight=1)
    intent_score = _intent_match_score(intent_rules, normalized)
    return keyword_score + name_score + description_score + intent_score


def _intent_match_score(rules: Sequence[SkillIntentRule], normalized: str) -> int:
    best = 0
    for rule in rules:
        if any(term.casefold() in normalized for term in rule.none_of if term.strip()):
            continue
        groups = [[term.casefold() for term in group if term.strip()] for group in rule.all_of]
        groups = [group for group in groups if group]
        if groups and all(any(term in normalized for term in group) for group in groups):
            best = max(best, 40 + len(groups) * 10)
    return best


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
