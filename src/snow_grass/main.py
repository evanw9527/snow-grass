from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from snow_grass.agent.runner import AgentRunner
from snow_grass.api.router import api_router
from snow_grass.core.config import Settings, get_settings
from snow_grass.local_usage import CodexUsageService
from snow_grass.memory.context_builder import ContextBuilder
from snow_grass.memory.security import MemorySecurity
from snow_grass.memory.service import MemoryService
from snow_grass.memory.token_counter import ConservativeTokenCounter
from snow_grass.persistence.database import Database
from snow_grass.persistence.repository import ChatRepository, ToolResultCacheRepository
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.executor import SkillScriptExecutor
from snow_grass.skills.registry import SkillRegistry
from snow_grass.skills.repository import SkillRepository
from snow_grass.skills.service import SkillService
from snow_grass.skills.validation import SkillValidator
from snow_grass.tools.registry import ToolRegistry
from snow_grass.tools.result_cache import ToolResultReuseService


def create_app(
    *,
    settings: Settings | None = None,
    database: Database | None = None,
    providers: ProviderRegistry | None = None,
    skills: SkillRegistry | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_database = database or Database(app_settings.database_url)
    app_providers = providers or ProviderRegistry.from_catalog(
        app_settings.load_model_catalog(), app_settings.provider_api_keys()
    )
    app_skills = skills or SkillRegistry.load(app_settings.resolved_skills_path())
    tools = ToolRegistry()
    skill_executor = SkillScriptExecutor(
        enabled=app_settings.skill_script_execution_enabled,
        timeout_seconds=app_settings.skill_script_timeout_seconds,
        max_output_chars=app_settings.skill_script_max_output_chars,
    )
    repository = ChatRepository(app_database.session_factory)
    tool_result_cache = ToolResultReuseService(
        repository=ToolResultCacheRepository(app_database.session_factory),
        workspace_id=app_settings.memory_workspace_id,
    )
    codex_usage = CodexUsageService(app_database.session_factory, app_settings.codex_state_db_path)
    skill_repository = SkillRepository(app_database.session_factory)
    skill_service = SkillService(
        repository=skill_repository,
        registry=app_skills,
        validator=SkillValidator(
            settings=app_settings,
            providers=app_providers,
            tools=tools,
            security=MemorySecurity(),
        ),
    )
    token_counter = ConservativeTokenCounter()
    context_builder = ContextBuilder(
        token_counter=token_counter,
        token_budget=app_settings.memory_context_token_budget,
        recent_message_limit=app_settings.memory_recent_message_limit,
        summary_target_tokens=app_settings.memory_summary_target_tokens,
    )
    memory = MemoryService(
        settings=app_settings,
        repository=repository,
        providers=app_providers,
        context_builder=context_builder,
        token_counter=token_counter,
        security=MemorySecurity(),
    )
    runner = AgentRunner(
        providers=app_providers,
        skills=app_skills,
        tools=tools,
        skill_executor=skill_executor,
        tool_result_cache=tool_result_cache,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await app_database.create_schema()
        await skill_service.initialize()
        yield
        await app_database.dispose()

    application = FastAPI(title="Snow Grass API", version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.settings = app_settings
    application.state.database = app_database
    application.state.repository = repository
    application.state.codex_usage = codex_usage
    application.state.providers = app_providers
    application.state.skills = app_skills
    application.state.skill_service = skill_service
    application.state.skill_executor = skill_executor
    application.state.tool_result_cache = tool_result_cache
    application.state.tools = tools
    application.state.runner = runner
    application.state.memory = memory
    application.include_router(api_router)
    return application


app = create_app()
