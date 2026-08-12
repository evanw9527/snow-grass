from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LocalClassification:
    category: str
    severity: int
    confidence: float
    should_respond: bool
    needs_fallback: bool = False


class LocalActivityClassifier(Protocol):
    def classify(self, text: str) -> LocalClassification: ...


class WeightedLexiconClassifier:
    """Small deterministic classifier for the common, unambiguous cases.

    It intentionally sends mixed or context-heavy language to the configured LLM.
    The local result never contains the original text or a generated diagnosis.
    """

    _categories: dict[str, tuple[str, ...]] = {
        "frustrated": ("沮丧", "挫败", "卡死", "受够", "崩溃", "frustrated", "stuck"),
        "anxious": ("焦虑", "担心", "害怕", "慌", "压力好大", "anxious", "worried"),
        "angry": ("生气", "愤怒", "气死", "烦死", "讨厌", "angry", "furious"),
        "tired": ("累死", "好累", "太累", "疲惫", "撑不住", "exhausted", "burned out"),
        "sad": ("难过", "伤心", "想哭", "很痛苦", "sad", "miserable"),
    }
    _intensifiers = ("真的", "非常", "太", "特别", "极其", "好", "很", "so ", "very ")
    _first_person = ("我", "自己", "俺", "i ", "i'm", "im ", "my ")
    _negations = ("不", "没", "并非", "不是", "没有", "not ", "isn't", "wasn't")
    _technical = (
        "测试", "单测", "报错", "异常", "错误码", "接口", "请求", "登录失败", "构建失败",
        "编译", "bug", "error", "exception", "test", "build failed", "http", "status code",
    )
    _reported_speech = ("他说", "她说", "用户说", "文案是", "引用", "原文", "反馈说")
    _complex = ("但是", "不过", "然而", "可我", "虽然", "but ", "though", "however")

    def classify(self, text: str) -> LocalClassification:
        normalized = " ".join(text.casefold().split())
        if not normalized:
            return self._neutral(1.0)

        matches: list[tuple[str, str]] = []
        for category, phrases in self._categories.items():
            matches.extend((category, phrase) for phrase in phrases if phrase in normalized)

        if not matches:
            return self._neutral(0.96)
        if any(marker in normalized for marker in self._reported_speech):
            return self._neutral(0.9)
        if any(marker in normalized for marker in self._complex):
            return LocalClassification("none", 0, 0.45, False, True)

        emotional_phrases = [phrase for _, phrase in matches]
        technical = any(marker in normalized for marker in self._technical)
        first_person = any(marker in f"{normalized} " for marker in self._first_person)
        negated = any(self._is_negated(normalized, phrase) for phrase in emotional_phrases)
        if negated:
            return self._neutral(0.91)
        if technical and not first_person:
            return self._neutral(0.94)

        # A direct, first-person emotional phrase is safe to resolve locally. A bare
        # negative word remains ambiguous and is delegated to the remote fallback.
        if first_person or any(phrase in normalized for phrase in ("好累", "累死", "烦死", "气死")):
            category = matches[0][0]
            intensified = any(word in normalized for word in self._intensifiers)
            severity = 2 if intensified or len(matches) > 1 else 1
            return LocalClassification(category, severity, 0.91, True)
        return LocalClassification("none", 0, 0.5, False, True)

    def _is_negated(self, text: str, phrase: str) -> bool:
        index = text.find(phrase)
        prefix = text[max(0, index - 5) : index]
        return any(
            re.search(re.escape(negation) + r".{0,3}$", prefix)
            for negation in self._negations
        )

    @staticmethod
    def _neutral(confidence: float) -> LocalClassification:
        return LocalClassification("none", 0, confidence, False)
