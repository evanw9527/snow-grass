from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    type: str
    run_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_sse(self) -> str:
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            **self.data,
        }
        return f"event: {self.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
