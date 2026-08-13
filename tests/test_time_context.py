from __future__ import annotations

from datetime import UTC, datetime

from snow_grass.knowledge.service import _work_overview_range
from snow_grass.memory.time_context import current_time_context, local_knowledge_title


def test_relative_date_context_uses_shanghai_calendar() -> None:
    now = datetime(2026, 8, 12, 16, 30, tzinfo=UTC)

    context = current_time_context(now)

    assert "today=2026-08-13" in context
    assert "yesterday=2026-08-12" in context
    assert "tomorrow=2026-08-14" in context


def test_yesterday_work_range_is_previous_shanghai_day() -> None:
    now = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)

    start, end = _work_overview_range("总结昨天的工作", now=now) or (None, None)

    assert start == datetime(2026, 8, 11, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 8, 12, 15, 59, 59, 999999, tzinfo=UTC)


def test_legacy_utc_summary_title_is_rendered_as_local_date() -> None:
    occurred_at = datetime(2026, 8, 11, 16, 0, tzinfo=UTC)

    title = local_knowledge_title(
        title="工作总结 · 2026-08-11 16:00",
        source_type="hourly_summary",
        occurred_at=occurred_at,
    )

    assert title == "工作总结 · 2026-08-12 00:00"
