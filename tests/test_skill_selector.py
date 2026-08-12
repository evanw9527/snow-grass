from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from snow_grass.agent.skill_selector import HybridSkillSelector
from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelInfo,
    ModelRequestOptions,
    ModelStreamChunk,
    TokenUsage,
)
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.registry import SkillRegistry
from snow_grass.skills.schema import (
    LoadedSkill,
    SkillIntentRule,
    SkillManifest,
    SkillSelection,
)


class RouterProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        self.calls += 1
        assert options == ModelRequestOptions(response_format="json_object")
        assert tools is None
        yield ModelStreamChunk(delta=self.response)
        yield ModelStreamChunk(usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14))


def skill(
    skill_id: str,
    name: str,
    description: str,
    *,
    selection: SkillSelection,
) -> LoadedSkill:
    return LoadedSkill(
        manifest=SkillManifest(
            id=skill_id,
            name=name,
            version="1.0.0",
            description=description,
            selection=selection,
        ),
        instructions="执行当前技能。",
    )


def setup_selector(response: str) -> tuple[HybridSkillSelector, RouterProvider]:
    work = skill(
        "work-recap",
        "工作回顾",
        "总结个人日常工作",
        selection=SkillSelection(
            intent_rules=[
                SkillIntentRule(
                    all_of=[
                        ["今天", "最近"],
                        ["做", "忙", "工作"],
                        ["什么", "啥", "总结"],
                    ]
                )
            ]
        ),
    )
    weather = skill(
        "weather",
        "天气查询",
        "查询天气和气温",
        selection=SkillSelection(keywords=["天气"]),
    )
    registry = SkillRegistry({"work-recap": work, "weather": weather})
    provider = RouterProvider(response)
    providers = ProviderRegistry()
    providers.register(
        ModelInfo(
            id="router-model",
            name="Router",
            provider="test",
            provider_name="Test",
            available=True,
        ),
        provider=provider,
    )
    return HybridSkillSelector(providers=providers, skills=registry), provider


async def test_strong_rule_skips_model_router() -> None:
    selector, provider = setup_selector("{}")
    result = await selector.select(
        model_id="router-model",
        messages=[ChatMessage(role="user", content="我今天做了啥")],
        explicit_skill_id=None,
    )
    assert result.skill is not None
    assert result.skill.manifest.id == "work-recap"
    assert result.source == "rule"
    assert provider.calls == 0


async def test_model_router_resolves_uncertain_intent_and_counts_usage() -> None:
    selector, provider = setup_selector(
        '{"selected_skill_id":"work-recap","confidence":0.91,"reason":"工作回顾"}'
    )
    result = await selector.select(
        model_id="router-model",
        messages=[ChatMessage(role="user", content="帮我梳理一下之前都忙了些什么")],
        explicit_skill_id=None,
    )
    assert result.skill is not None
    assert result.skill.manifest.id == "work-recap"
    assert result.source == "model"
    assert result.confidence == 0.91
    assert result.usage.total_tokens == 14
    assert provider.calls == 1


async def test_low_confidence_model_decision_selects_no_skill() -> None:
    selector, _ = setup_selector(
        '{"selected_skill_id":"weather","confidence":0.3,"reason":"不确定"}'
    )
    result = await selector.select(
        model_id="router-model",
        messages=[ChatMessage(role="user", content="随便聊聊")],
        explicit_skill_id=None,
    )
    assert result.skill is None
    assert result.source == "none"
