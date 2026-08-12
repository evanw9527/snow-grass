from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from snow_grass.providers.base import (
    ChatMessage,
    ChatTool,
    ModelRequestOptions,
    ModelStreamChunk,
    ProviderError,
    TokenUsage,
    ToolCallDelta,
)


class OpenAICompatibleProvider:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream_chat(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ChatTool] | None = None,
        options: ModelRequestOptions | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        try:
            message_params = [
                cast(
                    ChatCompletionMessageParam,
                    message.model_dump(exclude_none=True),
                )
                for message in messages
            ]
            request: dict[str, Any] = {
                "model": model_id,
                "messages": message_params,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools:
                request["tools"] = [
                        cast(
                            ChatCompletionToolParam,
                            tool.model_dump(mode="json"),
                        )
                        for tool in tools
                    ]
                request["tool_choice"] = "auto"
            if options and options.response_format == "json_object":
                request["response_format"] = {"type": "json_object"}
            stream = await self._client.chat.completions.create(**request)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                content = delta.content if delta else None
                if content:
                    yield ModelStreamChunk(delta=content)
                if delta and delta.tool_calls:
                    yield ModelStreamChunk(
                        tool_call_deltas=[
                            ToolCallDelta(
                                index=tool_call.index,
                                id=tool_call.id or "",
                                name=(
                                    tool_call.function.name
                                    if tool_call.function and tool_call.function.name
                                    else ""
                                ),
                                arguments=(
                                    tool_call.function.arguments
                                    if tool_call.function and tool_call.function.arguments
                                    else ""
                                ),
                            )
                            for tool_call in delta.tool_calls
                        ]
                    )
                if chunk.usage:
                    yield ModelStreamChunk(
                        usage=TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )
                    )
        except APIError as exc:
            raise ProviderError("The model provider request failed") from exc
