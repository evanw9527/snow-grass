from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from snow_grass.core.config import ModelCatalog
from snow_grass.providers.base import ModelInfo, ModelProvider, ProviderError
from snow_grass.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    info: ModelInfo
    provider_model_id: str
    provider: ModelProvider | None


class ProviderRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ProviderBinding] = {}

    @classmethod
    def from_catalog(
        cls, catalog: ModelCatalog, api_keys: Mapping[str, str] | None = None
    ) -> ProviderRegistry:
        registry = cls()
        key_source = api_keys if api_keys is not None else os.environ
        for provider_id, provider_config in catalog.providers.items():
            api_key = key_source.get(provider_config.api_key_env, "").strip()
            provider: ModelProvider | None = None
            if api_key and provider_config.adapter == "openai_compatible":
                provider = OpenAICompatibleProvider(
                    api_key=api_key, base_url=provider_config.base_url
                )
            for model in provider_config.models:
                registry.register(
                    ModelInfo(
                        id=model.id,
                        name=model.name,
                        provider=provider_id,
                        provider_name=provider_config.name,
                        capabilities=model.capabilities,
                        available=provider is not None,
                    ),
                    provider=provider,
                    provider_model_id=model.provider_model_id,
                )
        return registry

    def register(
        self,
        info: ModelInfo,
        *,
        provider: ModelProvider | None,
        provider_model_id: str | None = None,
    ) -> None:
        if info.id in self._models:
            raise ValueError(f"Duplicate model id: {info.id}")
        self._models[info.id] = ProviderBinding(
            info=info.model_copy(update={"available": provider is not None}),
            provider_model_id=provider_model_id or info.id,
            provider=provider,
        )

    def list_models(self) -> list[ModelInfo]:
        return [binding.info for binding in self._models.values()]

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def get_model_info(self, model_id: str) -> ModelInfo:
        binding = self._models.get(model_id)
        if binding is None:
            raise ProviderError(f"Unknown model: {model_id}")
        return binding.info

    def resolve(self, model_id: str) -> tuple[ModelProvider, str]:
        binding = self._models.get(model_id)
        if binding is None:
            raise ProviderError(f"Unknown model: {model_id}")
        if binding.provider is None:
            raise ProviderError(f"Model {model_id} is not available: provider key is missing")
        return binding.provider, binding.provider_model_id
