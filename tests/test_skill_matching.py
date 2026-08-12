from snow_grass.skills.matching import selection_score
from snow_grass.skills.schema import SkillIntentRule

WORK_RECAP_RULE = SkillIntentRule(
    all_of=[
        ["今天", "今日", "昨天", "昨日", "最近", "近期", "本周"],
        ["做", "干", "忙", "工作"],
        ["什么", "啥", "哪些", "总结", "回顾"],
    ]
)


def score(content: str) -> int:
    return selection_score(
        skill_id="work-recap",
        name="工作回顾",
        description="生成个人工作回顾",
        keywords=[],
        intent_rules=[WORK_RECAP_RULE],
        content=content,
    )


def test_intent_rule_accepts_colloquial_variants_and_word_order() -> None:
    assert score("我今天做了啥") > 0
    assert score("今天我都干什么了") > 0
    assert score("最近主要在忙哪些工作") > 0
    assert score("本周工作总结") > 0


def test_intent_rule_rejects_queries_missing_work_intent() -> None:
    assert score("今天杭州天气怎么样") == 0
    assert score("帮我总结这篇文章") == 0
