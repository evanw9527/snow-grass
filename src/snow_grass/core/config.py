from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CatalogModel(BaseModel):
    id: str
    provider_model_id: str
    name: str
    capabilities: list[str] = Field(default_factory=lambda: ["chat", "streaming"])


class CatalogProvider(BaseModel):
    name: str
    adapter: str
    base_url: str
    api_key_env: str
    models: list[CatalogModel]


class ModelCatalog(BaseModel):
    providers: dict[str, CatalogProvider]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="SNOW_GRASS_",
        extra="ignore",
    )

    environment: str = "development"
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'snow_grass.db'}"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    default_model_id: str = "deepseek-chat"
    models_config_path: Path = PROJECT_ROOT / "config/models.yaml"
    skills_path: Path = PROJECT_ROOT / "skills"
    skill_admin_enabled: bool = True
    skill_max_instruction_chars: int = Field(default=100_000, ge=1_000, le=1_000_000)
    skill_max_test_cases: int = Field(default=50, ge=0, le=500)
    skill_max_package_files: int = Field(default=200, ge=2, le=2_000)
    skill_max_package_chars: int = Field(default=2_000_000, ge=10_000, le=20_000_000)
    skill_script_execution_enabled: bool = True
    skill_script_timeout_seconds: int = Field(default=30, ge=1, le=300)
    skill_script_max_output_chars: int = Field(default=65_536, ge=1_024, le=1_000_000)
    codex_state_db_path: Path = Path.home() / ".codex/state_5.sqlite"
    memory_enabled: bool = True
    memory_workspace_id: str = Field(default="default", min_length=1, max_length=120)
    memory_recent_message_limit: int = Field(default=24, ge=4, le=200)
    memory_context_token_budget: int = Field(default=12_000, ge=1_000, le=1_000_000)
    memory_output_token_reserve: int = Field(default=4_000, ge=256, le=500_000)
    memory_compact_threshold: int = Field(default=32, ge=8, le=1_000)
    memory_summary_target_tokens: int = Field(default=1_200, ge=128, le=16_000)
    memory_retrieval_limit: int = Field(default=8, ge=0, le=50)
    memory_auto_extract_enabled: bool = False
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    glm_api_key: SecretStr | None = Field(default=None, validation_alias="GLM_API_KEY")

    def resolved_models_config_path(self) -> Path:
        return self._resolve(self.models_config_path)

    def resolved_skills_path(self) -> Path:
        return self._resolve(self.skills_path)

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    def load_model_catalog(self) -> ModelCatalog:
        path = self.resolved_models_config_path()
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        return ModelCatalog.model_validate(raw)

    def provider_api_keys(self) -> dict[str, str]:
        values = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "GLM_API_KEY": self.glm_api_key,
        }
        return {
            name: secret.get_secret_value()
            for name, secret in values.items()
            if secret is not None and secret.get_secret_value().strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
