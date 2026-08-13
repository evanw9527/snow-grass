from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")


def current_time_context(now: datetime | None = None) -> str:
    """Return an explicit local calendar anchor for relative-date reasoning."""
    local_now = now.astimezone(SHANGHAI) if now is not None else datetime.now(SHANGHAI)
    today = local_now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    return (
        "<current_time>\n"
        f"Timezone: Asia/Shanghai (UTC+08:00). Current local datetime: "
        f"{local_now.isoformat(timespec='seconds')}.\n"
        f"Calendar anchors: today={today.isoformat()}, "
        f"yesterday={yesterday.isoformat()}, tomorrow={tomorrow.isoformat()}.\n"
        "Resolve relative dates from these anchors. Dates shown on retrieved knowledge records "
        "are also Asia/Shanghai local time. Never reinterpret them as UTC.\n"
        "</current_time>"
    )


def local_knowledge_title(
    *, title: str, source_type: str, occurred_at: datetime
) -> str:
    """Repair legacy generated summary titles that embedded a UTC calendar date."""
    if source_type != "hourly_summary" or not title.startswith("工作总结 · "):
        return title
    local_time = occurred_at.astimezone(SHANGHAI)
    return f"工作总结 · {local_time:%Y-%m-%d %H:00}"
