from __future__ import annotations

import json

from pydantic import ValidationError

from snow_grass.persistence.repository import ChatRepository
from snow_grass.pet.schemas import PetAction, PetRespondRequest, PetRespondResponse
from snow_grass.providers.base import ChatMessage, ModelRequestOptions, TokenUsage
from snow_grass.providers.registry import ProviderRegistry

SYSTEM_PROMPT = """你是一只友善、活泼但回答简短的中文桌宠。
根据用户输入给出不超过 200 字的回复，并选择一个可用动作。
必须只输出 JSON 对象，格式为 {"reply":"回复","action":"idle"}。
action 必须来自用户提供的 available_actions，禁止输出其他动作。
不要输出 Markdown、代码围栏、解释或额外字段。"""


class PetResponseService:
    def __init__(self, *, providers: ProviderRegistry, repository: ChatRepository) -> None:
        self._providers = providers
        self._repository = repository

    async def respond(self, payload: PetRespondRequest) -> PetRespondResponse:
        session = await self._repository.get_session(payload.session_id)
        if session is None:
            raise LookupError("Pet session not found")
        model_id = payload.model_id or session.model_id
        provider, provider_model_id = self._providers.resolve(model_id)

        await self._repository.add_message(
            session_id=payload.session_id,
            role="user",
            content=payload.content,
            model_id=model_id,
        )
        history = await self._repository.list_messages(payload.session_id)
        actions = ", ".join(action.value for action in payload.available_actions)
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            *[
                ChatMessage(role=item.role, content=item.content)
                for item in history[-12:]
            ],
            ChatMessage(role="system", content=f"available_actions: [{actions}]"),
        ]

        chunks: list[str] = []
        usage = TokenUsage()
        async for chunk in provider.stream_chat(
            model_id=provider_model_id,
            messages=messages,
            options=ModelRequestOptions(response_format="json_object"),
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.usage is not None:
                usage = chunk.usage

        response = self._parse("".join(chunks), payload.available_actions, usage)
        await self._repository.add_message(
            session_id=payload.session_id,
            role="assistant",
            content=response.reply,
            model_id=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
        return response

    @staticmethod
    def _parse(
        content: str,
        available_actions: list[PetAction],
        usage: TokenUsage,
    ) -> PetRespondResponse:
        fallback_action = (
            PetAction.idle if PetAction.idle in available_actions else available_actions[0]
        )
        try:
            raw = json.loads(content.strip())
            response = PetRespondResponse.model_validate({**raw, "usage": usage})
            if response.action not in available_actions:
                return response.model_copy(update={"action": fallback_action})
            return response
        except (json.JSONDecodeError, TypeError, ValidationError):
            return PetRespondResponse(
                reply="我听到了，不过刚刚有一点走神。再和我说一次吧。",
                action=fallback_action,
                usage=usage,
            )
