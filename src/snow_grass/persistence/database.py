from __future__ import annotations

from functools import partial
from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Domain models register their own tables on the shared metadata without importing
# chat, memory, agent, or pet services.
from snow_grass.activity import models as activity_models  # noqa: F401
from snow_grass.knowledge import models as knowledge_models  # noqa: F401
from snow_grass.persistence.models import Base
from snow_grass.workflow import models as workflow_models  # noqa: F401


class Database:
    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await self._migrate_sqlite_workflow_v2_schema(connection)
            await self._migrate_sqlite_message_usage(connection)
            await self._migrate_sqlite_activity_analysis(connection)
            await self._migrate_sqlite_activity_summary_flow(connection)
            await self._migrate_sqlite_activity_summary_knowledge(connection)
            await self._migrate_sqlite_session_knowledge(connection)
            await self._migrate_sqlite_session_runtime(connection)
            await self._initialize_sqlite_knowledge_fts(connection)

    async def _migrate_sqlite_workflow_v2_schema(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        tables = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
        definitions = {
            "workflows": {
                "description": "TEXT NOT NULL DEFAULT ''",
                "status": "VARCHAR(20) NOT NULL DEFAULT 'active'",
            },
            "workflow_versions": {
                "component_manifest_digest": "VARCHAR(128) NOT NULL DEFAULT ''",
            },
            "component_versions": {
                "icon": "VARCHAR(120) NOT NULL DEFAULT 'tool'",
                "implementation_source": "TEXT NOT NULL DEFAULT ''",
                "source_language": "VARCHAR(40) NOT NULL DEFAULT 'python'",
                "source_path": "VARCHAR(500)",
            },
            "workflow_runs": {
                "resolved_version_id": "VARCHAR(36)",
                "node_count": "INTEGER NOT NULL DEFAULT 0",
                "executed_node_count": "INTEGER NOT NULL DEFAULT 0",
                "succeeded_node_count": "INTEGER NOT NULL DEFAULT 0",
                "failed_node_count": "INTEGER NOT NULL DEFAULT 0",
                "skipped_node_count": "INTEGER NOT NULL DEFAULT 0",
                "duration_ms": "INTEGER NOT NULL DEFAULT 0",
                "node_summary": "JSON NOT NULL DEFAULT '{}'",
                "details_purged_at": "DATETIME",
            },
            "workflow_node_runs": {
                "component_version_id": "VARCHAR(36)",
                "input_port_summary": "JSON NOT NULL DEFAULT '{}'",
                "output_port_summary": "JSON NOT NULL DEFAULT '{}'",
                "skip_reason": "VARCHAR(120)",
            },
        }
        for table_name, columns_to_add in definitions.items():
            if table_name not in tables:
                continue
            columns = await connection.run_sync(
                partial(_sqlite_column_names, table_name=table_name)
            )
            for column_name, definition in columns_to_add.items():
                if column_name not in columns:
                    await connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
                    )

    async def _migrate_sqlite_message_usage(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("chat_messages")
            }
        )
        for column_name in ("input_tokens", "output_tokens", "total_tokens"):
            if column_name not in columns:
                await connection.execute(
                    text(
                        f"ALTER TABLE chat_messages ADD COLUMN {column_name} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                )

    async def _migrate_sqlite_activity_analysis(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        tables = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
        if "activity_events" not in tables:
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("activity_events")
            }
        )
        definitions = {
            "analysis_source": "VARCHAR(20)",
            "analysis_category": "VARCHAR(24)",
            "analysis_action": "VARCHAR(20)",
            "analysis_severity": "INTEGER",
            "analysis_should_respond": "BOOLEAN",
            "analyzed_at": "DATETIME",
        }
        for column_name, definition in definitions.items():
            if column_name not in columns:
                await connection.execute(
                    text(f"ALTER TABLE activity_events ADD COLUMN {column_name} {definition}")
                )

    async def _migrate_sqlite_activity_summary_flow(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        tables = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
        if "hourly_activity_summaries" not in tables:
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"]
                for column in inspect(sync_connection).get_columns("hourly_activity_summaries")
            }
        )
        definitions = {
            "flow_version_id": "VARCHAR(36)",
            "flow_run_id": "VARCHAR(36)",
            "quality_status": "VARCHAR(40)",
        }
        for column_name, definition in definitions.items():
            if column_name not in columns:
                await connection.execute(
                    text(
                        "ALTER TABLE hourly_activity_summaries "
                        f"ADD COLUMN {column_name} {definition}"
                    )
                )

    async def _migrate_sqlite_activity_summary_knowledge(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        tables = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
        if "hourly_activity_summaries" not in tables:
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"]
                for column in inspect(sync_connection).get_columns("hourly_activity_summaries")
            }
        )
        for column_name, definition in {
            "title": "VARCHAR(240) NOT NULL DEFAULT ''",
            "keywords_json": "TEXT NOT NULL DEFAULT '[]'",
        }.items():
            if column_name not in columns:
                await connection.execute(
                    text(
                        "ALTER TABLE hourly_activity_summaries "
                        f"ADD COLUMN {column_name} {definition}"
                    )
                )

    async def _migrate_sqlite_session_knowledge(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("chat_sessions")
            }
        )
        if "knowledge_enabled" not in columns:
            await connection.execute(
                text(
                    "ALTER TABLE chat_sessions ADD COLUMN knowledge_enabled "
                    "BOOLEAN NOT NULL DEFAULT 0"
                )
            )

    async def _migrate_sqlite_session_runtime(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("chat_sessions")
            }
        )
        if "runtime_id" not in columns:
            await connection.execute(
                text(
                    "ALTER TABLE chat_sessions ADD COLUMN runtime_id "
                    "VARCHAR(40) NOT NULL DEFAULT 'native'"
                )
            )

    async def _initialize_sqlite_knowledge_fts(self, connection: AsyncConnection) -> None:
        if connection.dialect.name != "sqlite":
            return
        try:
            await connection.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_documents_fts USING fts5("
                    "title, content, keywords_json, content='knowledge_documents', "
                    "content_rowid='rowid', tokenize='trigram')"
                )
            )
            await connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS knowledge_documents_ai AFTER INSERT ON "
                    "knowledge_documents WHEN new.status = 'active' BEGIN "
                    "INSERT INTO knowledge_documents_fts(rowid,title,content,keywords_json) "
                    "VALUES(new.rowid,new.title,new.content,new.keywords_json); END"
                )
            )
            await connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS knowledge_documents_ad AFTER DELETE ON "
                    "knowledge_documents WHEN old.status = 'active' BEGIN "
                    "INSERT INTO knowledge_documents_fts("
                    "knowledge_documents_fts,rowid,title,content,keywords_json) "
                    "VALUES('delete',old.rowid,old.title,old.content,old.keywords_json); END"
                )
            )
            await connection.execute(
                text(
                    "CREATE TRIGGER IF NOT EXISTS knowledge_documents_au AFTER UPDATE ON "
                    "knowledge_documents BEGIN "
                    "INSERT INTO knowledge_documents_fts("
                    "knowledge_documents_fts,rowid,title,content,keywords_json) "
                    "SELECT 'delete',old.rowid,old.title,old.content,old.keywords_json "
                    "WHERE old.status = 'active'; "
                    "INSERT INTO knowledge_documents_fts(rowid,title,content,keywords_json) "
                    "SELECT new.rowid,new.title,new.content,new.keywords_json "
                    "WHERE new.status = 'active'; END"
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO knowledge_documents_fts(knowledge_documents_fts) VALUES('rebuild')"
                )
            )
        except Exception:
            # Knowledge search has a parameterized SQL fallback; schema startup stays available.
            return

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.session_factory()


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _sqlite_column_names(sync_connection: Any, *, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(sync_connection).get_columns(table_name)}
