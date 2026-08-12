from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from snow_grass.activity.intelligence import ActivityIntelligence
from snow_grass.activity.local_classifier import WeightedLexiconClassifier
from snow_grass.activity.repository import ActivityRepository
from snow_grass.activity.router import activity_router
from snow_grass.activity.service import ActivityService
from snow_grass.activity.summary_pack import register_activity_summary_pack
from snow_grass.agent.runner import AgentRunner
from snow_grass.agent.skill_selector import HybridSkillSelector
from snow_grass.api.router import api_router
from snow_grass.core.config import Settings, get_settings
from snow_grass.feedback.flow_pack import register_negative_feedback_pack
from snow_grass.knowledge.repository import KnowledgeRepository
from snow_grass.knowledge.router import knowledge_router
from snow_grass.knowledge.service import KnowledgeService
from snow_grass.local_usage import CodexUsageService
from snow_grass.memory.context_builder import ContextBuilder
from snow_grass.memory.security import MemorySecurity
from snow_grass.memory.service import MemoryService
from snow_grass.memory.token_counter import ConservativeTokenCounter
from snow_grass.persistence.database import Database
from snow_grass.persistence.repository import ChatRepository, ToolResultCacheRepository
from snow_grass.pet.router import pet_router
from snow_grass.pet.service import PetResponseService
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.executor import SkillScriptExecutor
from snow_grass.skills.registry import SkillRegistry
from snow_grass.skills.repository import SkillRepository
from snow_grass.skills.service import SkillService
from snow_grass.skills.validation import SkillValidator
from snow_grass.tools.registry import ToolRegistry
from snow_grass.tools.result_cache import ToolResultReuseService
from snow_grass.workflow.compiler import WorkflowCompiler
from snow_grass.workflow.component_router import component_router
from snow_grass.workflow.executor import WorkflowExecutor
from snow_grass.workflow.registry import BusinessPackRegistry, RuntimeFactoryCatalog
from snow_grass.workflow.repository import WorkflowRepository
from snow_grass.workflow.router import workflow_router
from snow_grass.workflow.runtime import ComponentRuntimeRegistry
from snow_grass.workflow.service import ComponentService, WorkflowService


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
    pet_service = PetResponseService(providers=app_providers, repository=repository)
    activity_intelligence = ActivityIntelligence(
        classifier=WeightedLexiconClassifier(),
        providers=app_providers,
    )
    activity_repository = ActivityRepository(app_database.session_factory)
    knowledge_repository = KnowledgeRepository(
        app_database.session_factory,
        workspace_id=app_settings.memory_workspace_id,
    )
    knowledge_service = KnowledgeService(
        repository=knowledge_repository,
        activity_repository=activity_repository,
        search_timeout_ms=app_settings.knowledge_search_timeout_ms,
        summary_retention_days=app_settings.knowledge_summary_retention_days,
    )
    workflow_factories = RuntimeFactoryCatalog()
    workflow_packs = BusinessPackRegistry()
    workflow_packs.register(
        register_activity_summary_pack(
            workflow_factories, activity_intelligence, activity_repository
        )
    )
    workflow_packs.register(register_negative_feedback_pack(workflow_factories))
    workflow_repository = WorkflowRepository(app_database.session_factory)
    workflow_runtimes = ComponentRuntimeRegistry(workflow_factories)
    workflow_compiler = WorkflowCompiler(workflow_repository, workflow_factories)
    workflow_service = WorkflowService(
        repository=workflow_repository,
        packs=workflow_packs,
        factories=workflow_factories,
        compiler=workflow_compiler,
        executor=WorkflowExecutor(workflow_runtimes),
    )
    component_service = ComponentService(
        repository=workflow_repository,
        factories=workflow_factories,
        runtimes=workflow_runtimes,
        packs=workflow_packs,
    )
    activity_service = ActivityService(
        activity_repository,
        retention_days=app_settings.activity_retention_days,
        intelligence=activity_intelligence,
        workflows=workflow_service,
        knowledge=knowledge_service,
    )
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
        knowledge_token_budget=app_settings.knowledge_context_token_budget,
    )
    memory = MemoryService(
        settings=app_settings,
        repository=repository,
        providers=app_providers,
        context_builder=context_builder,
        token_counter=token_counter,
        security=MemorySecurity(),
        knowledge=knowledge_service,
    )
    runner = AgentRunner(
        providers=app_providers,
        skill_selector=HybridSkillSelector(
            providers=app_providers,
            skills=app_skills,
            rule_threshold=app_settings.skill_selection_rule_threshold,
            rule_margin=app_settings.skill_selection_rule_margin,
            model_confidence_threshold=app_settings.skill_selection_model_confidence,
            timeout_seconds=app_settings.skill_selection_model_timeout_seconds,
        ),
        tools=tools,
        skill_executor=skill_executor,
        tool_result_cache=tool_result_cache,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await app_database.create_schema()
        await skill_service.initialize()
        await workflow_service.initialize()
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
    application.state.pet_service = pet_service
    application.state.activity_service = activity_service
    application.state.knowledge_service = knowledge_service
    application.state.knowledge_repository = knowledge_repository
    application.state.workflow_service = workflow_service
    application.state.component_service = component_service
    application.include_router(api_router)
    application.include_router(pet_router)
    application.include_router(activity_router)
    application.include_router(knowledge_router)
    application.include_router(workflow_router)
    application.include_router(component_router)
    return application


app = create_app()
