from __future__ import annotations

from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from snow_grass.persistence.models import Base


class Database:
    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await self._migrate_sqlite_message_usage(connection)

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

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.session_factory()


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
