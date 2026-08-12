from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from snow_grass.providers.base import TokenUsage


class PetAction(StrEnum):
    idle = "idle"
    like = "like"
    wave = "wave"
    dance = "dance"
    sleep = "sleep"
    angry = "angry"
    comfort = "comfort"


class PetRespondRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4_000)
    available_actions: list[PetAction] = Field(min_length=1, max_length=7)
    model_id: str | None = None


class PetRespondResponse(BaseModel):
    reply: str = Field(min_length=1, max_length=200)
    action: PetAction
    usage: TokenUsage = Field(default_factory=TokenUsage)
