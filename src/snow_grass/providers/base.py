from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ChatFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatFunction


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatToolFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatToolFunction


class ToolCallDelta(BaseModel):
    index: int = Field(ge=0)
    id: str = ""
    name: str = ""
    arguments: str = ""


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    provider_name: str
    capabilities: list[str] = Field(default_factory=list)
    available: bool


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelStreamChunk(BaseModel):
    delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list)
    usage: TokenUsage | None = None


class ModelRequestOptions(BaseModel):
    response_format: Literal["text", "json_object"] = "text"


class ProviderError(RuntimeError):
    pass


class ModelProvider(Protocol):
    def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]: ...
