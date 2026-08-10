from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from snow_grass.persistence.models import (
    ManagedSkillRecord,
    SkillAuditRecord,
    SkillDraftRecord,
    SkillReleaseRecord,
    SkillRuntimeStateRecord,
    utc_now,
)


class SkillRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_skills(self) -> list[ManagedSkillRecord]:
        statement = select(ManagedSkillRecord).order_by(ManagedSkillRecord.updated_at.desc())
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def get_skill(self, skill_id: str) -> ManagedSkillRecord | None:
        async with self._session_factory() as session:
            return await session.get(ManagedSkillRecord, skill_id)

    async def get_draft(self, skill_id: str) -> SkillDraftRecord | None:
        async with self._session_factory() as session:
            return await session.get(SkillDraftRecord, skill_id)

    async def create_skill(
        self,
        *,
        skill_id: str,
        name: str,
        description: str,
        content: dict[str, object],
        content_hash: str,
    ) -> tuple[ManagedSkillRecord, SkillDraftRecord]:
        skill = ManagedSkillRecord(id=skill_id, name=name, description=description)
        draft = SkillDraftRecord(
            skill_id=skill_id,
            revision=1,
            content=content,
            content_hash=content_hash,
        )
        async with self._session_factory() as session:
            session.add_all([skill, draft])
            await session.commit()
            await session.refresh(skill)
            await session.refresh(draft)
        return skill, draft

    async def save_draft(
        self,
        *,
        skill_id: str,
        expected_revision: int,
        content: dict[str, object],
        content_hash: str,
        name: str,
        description: str,
    ) -> SkillDraftRecord | None:
        async with self._session_factory() as session:
            draft = await session.get(SkillDraftRecord, skill_id)
            skill = await session.get(ManagedSkillRecord, skill_id)
            if draft is None or skill is None or draft.revision != expected_revision:
                return None
            draft.revision += 1
            draft.content = content
            draft.content_hash = content_hash
            draft.validated_hash = None
            draft.validation_report = None
            draft.updated_at = utc_now()
            skill.name = name
            skill.description = description
            skill.updated_at = utc_now()
            await session.commit()
            await session.refresh(draft)
            return draft

    async def save_validation(
        self,
        *,
        skill_id: str,
        expected_revision: int,
        validated_hash: str,
        report: dict[str, object],
    ) -> SkillDraftRecord | None:
        async with self._session_factory() as session:
            draft = await session.get(SkillDraftRecord, skill_id)
            if draft is None or draft.revision != expected_revision:
                return None
            draft.validated_hash = validated_hash
            draft.validation_report = report
            draft.updated_at = utc_now()
            await session.commit()
            await session.refresh(draft)
            return draft

    async def publish(
        self,
        *,
        skill_id: str,
        version: str,
        content: dict[str, object],
        checksum: str,
        release_notes: str | None,
        activate: bool,
    ) -> SkillReleaseRecord | None:
        async with self._session_factory() as session:
            skill = await session.get(ManagedSkillRecord, skill_id)
            if skill is None:
                return None
            duplicate = await session.scalar(
                select(SkillReleaseRecord).where(
                    SkillReleaseRecord.skill_id == skill_id,
                    SkillReleaseRecord.version == version,
                )
            )
            if duplicate is not None:
                return None
            release = SkillReleaseRecord(
                id=str(uuid4()),
                skill_id=skill_id,
                version=version,
                content=content,
                checksum=checksum,
                release_notes=release_notes,
            )
            session.add(release)
            skill.lifecycle_status = "published"
            if activate:
                skill.active_release_id = release.id
            skill.updated_at = utc_now()
            await session.commit()
            await session.refresh(release)
            return release

    async def list_releases(self, skill_id: str) -> list[SkillReleaseRecord]:
        statement = (
            select(SkillReleaseRecord)
            .where(SkillReleaseRecord.skill_id == skill_id)
            .order_by(SkillReleaseRecord.published_at.desc())
        )
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())

    async def get_release(self, release_id: str) -> SkillReleaseRecord | None:
        async with self._session_factory() as session:
            return await session.get(SkillReleaseRecord, release_id)

    async def activate_release(
        self, *, skill_id: str, release_id: str
    ) -> tuple[str | None, SkillReleaseRecord] | None:
        async with self._session_factory() as session:
            skill = await session.get(ManagedSkillRecord, skill_id)
            release = await session.get(SkillReleaseRecord, release_id)
            if skill is None or release is None or release.skill_id != skill_id:
                return None
            previous = skill.active_release_id
            skill.active_release_id = release_id
            skill.lifecycle_status = "published"
            skill.updated_at = utc_now()
            await session.commit()
            await session.refresh(release)
            return previous, release

    async def active_releases(self) -> list[tuple[ManagedSkillRecord, SkillReleaseRecord]]:
        statement = (
            select(ManagedSkillRecord, SkillReleaseRecord)
            .join(
                SkillReleaseRecord,
                SkillReleaseRecord.id == ManagedSkillRecord.active_release_id,
            )
            .where(ManagedSkillRecord.lifecycle_status == "published")
        )
        async with self._session_factory() as session:
            return list((await session.execute(statement)).tuples().all())

    async def set_enabled(self, skill_id: str, enabled: bool) -> None:
        async with self._session_factory() as session:
            state = await session.get(SkillRuntimeStateRecord, skill_id)
            if state is None:
                state = SkillRuntimeStateRecord(skill_id=skill_id, enabled=enabled)
                session.add(state)
            else:
                state.enabled = enabled
                state.updated_at = utc_now()
            skill = await session.get(ManagedSkillRecord, skill_id)
            if skill is not None:
                skill.enabled = enabled
                skill.updated_at = utc_now()
            await session.commit()

    async def enabled_overrides(self) -> dict[str, bool]:
        async with self._session_factory() as session:
            records = list((await session.scalars(select(SkillRuntimeStateRecord))).all())
            return {record.skill_id: record.enabled for record in records}

    async def archive(self, skill_id: str) -> ManagedSkillRecord | None:
        async with self._session_factory() as session:
            skill = await session.get(ManagedSkillRecord, skill_id)
            if skill is None:
                return None
            skill.lifecycle_status = "archived"
            skill.enabled = False
            skill.updated_at = utc_now()
            state = await session.get(SkillRuntimeStateRecord, skill_id)
            if state is None:
                session.add(SkillRuntimeStateRecord(skill_id=skill_id, enabled=False))
            else:
                state.enabled = False
                state.updated_at = utc_now()
            await session.commit()
            await session.refresh(skill)
            return skill

    async def add_audit(
        self,
        *,
        skill_id: str,
        action: str,
        release_id: str | None = None,
        revision: int | None = None,
        details: dict[str, object] | None = None,
    ) -> SkillAuditRecord:
        record = SkillAuditRecord(
            id=str(uuid4()),
            skill_id=skill_id,
            action=action,
            release_id=release_id,
            revision=revision,
            details=details,
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def list_audits(self, skill_id: str, limit: int = 50) -> list[SkillAuditRecord]:
        statement = (
            select(SkillAuditRecord)
            .where(SkillAuditRecord.skill_id == skill_id)
            .order_by(SkillAuditRecord.created_at.desc())
            .limit(limit)
        )
        async with self._session_factory() as session:
            return list((await session.scalars(statement)).all())
