from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

from snow_grass.persistence.models import SkillDraftRecord, SkillReleaseRecord
from snow_grass.skills.registry import SkillError, SkillRegistry
from snow_grass.skills.repository import SkillRepository
from snow_grass.skills.schema import SkillDraftContent, SkillValidationReport
from snow_grass.skills.validation import SkillValidator

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class SkillServiceError(ValueError):
    pass


class SkillNotFoundError(SkillServiceError):
    pass


class SkillConflictError(SkillServiceError):
    def __init__(self, message: str, *, current_revision: int | None = None) -> None:
        super().__init__(message)
        self.current_revision = current_revision


class SkillService:
    def __init__(
        self,
        *,
        repository: SkillRepository,
        registry: SkillRegistry,
        validator: SkillValidator,
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._validator = validator

    async def initialize(self) -> None:
        await self.refresh_registry()

    async def refresh_registry(self) -> None:
        managed = {}
        for skill, release in await self._repository.active_releases():
            content = SkillDraftContent.from_storage(release.content, skill_id=skill.id)
            managed[skill.id] = content.to_loaded_skill(
                skill_id=skill.id,
                version=release.version,
                enabled=skill.enabled,
            )
        overrides = await self._repository.enabled_overrides()
        self._registry.replace_managed_snapshot(managed)
        self._registry.apply_enabled_overrides(overrides)

    async def list_admin(
        self, *, query: str = "", source: str | None = None, status: str | None = None
    ) -> list[dict[str, object]]:
        normalized = query.casefold().strip()
        items: list[dict[str, object]] = []
        if source in {None, "builtin"}:
            for info in self._registry.list_skills():
                if info.id not in self._registry.builtin_ids():
                    continue
                items.append(
                    {
                        "id": info.id,
                        "source": "builtin",
                        "name": info.name,
                        "description": info.description,
                        "lifecycle_status": "published",
                        "enabled": info.enabled,
                        "draft_revision": None,
                        "active_version": info.version,
                        "has_unpublished_changes": False,
                        "updated_at": None,
                    }
                )
        if source in {None, "managed"}:
            for skill in await self._repository.list_skills():
                draft = await self._repository.get_draft(skill.id)
                releases = await self._repository.list_releases(skill.id)
                active = next(
                    (item for item in releases if item.id == skill.active_release_id), None
                )
                has_changes = bool(
                    draft
                    and (
                        active is None
                        or self.content_hash(
                            SkillDraftContent.from_storage(draft.content, skill_id=skill.id)
                        )
                        != active.checksum
                    )
                )
                items.append(
                    {
                        "id": skill.id,
                        "source": "managed",
                        "name": skill.name,
                        "description": skill.description,
                        "lifecycle_status": skill.lifecycle_status,
                        "enabled": skill.enabled,
                        "draft_revision": draft.revision if draft else None,
                        "active_version": active.version if active else None,
                        "has_unpublished_changes": has_changes,
                        "updated_at": skill.updated_at,
                    }
                )
        if status:
            items = [item for item in items if item["lifecycle_status"] == status]
        if normalized:
            items = [
                item
                for item in items
                if normalized in f"{item['id']} {item['name']} {item['description']}".casefold()
            ]
        return sorted(items, key=lambda item: (item["source"] != "managed", str(item["name"])))

    async def get_detail(self, skill_id: str) -> dict[str, object]:
        if skill_id in self._registry.builtin_ids():
            loaded = self._registry.get_loaded(skill_id)
            content = SkillDraftContent(files=loaded.package_files)
            return {
                "id": skill_id,
                "source": "builtin",
                "lifecycle_status": "published",
                "enabled": loaded.manifest.enabled,
                "draft": {
                    "revision": 0,
                    "content": content.model_dump(mode="json"),
                    "content_hash": self.content_hash(content),
                    "validated_hash": self.content_hash(content),
                    "validation_report": None,
                    "updated_at": None,
                },
                "active_release": {
                    "id": f"builtin:{skill_id}:{loaded.manifest.version}",
                    "version": loaded.manifest.version,
                    "checksum": self.content_hash(content),
                    "release_notes": "Built-in Skill",
                    "published_at": None,
                    "active": True,
                },
                "releases": [],
                "audits": [],
                "readonly": True,
            }
        skill = await self._repository.get_skill(skill_id)
        if skill is None:
            raise SkillNotFoundError(f"Unknown skill: {skill_id}")
        draft = await self._repository.get_draft(skill_id)
        releases = await self._repository.list_releases(skill_id)
        audits = await self._repository.list_audits(skill_id)
        return {
            "id": skill.id,
            "source": "managed",
            "lifecycle_status": skill.lifecycle_status,
            "enabled": skill.enabled,
            "draft": self._draft_dict(draft, skill_id=skill.id) if draft else None,
            "active_release": self._release_dict(
                next((item for item in releases if item.id == skill.active_release_id), None),
                active_id=skill.active_release_id,
            ),
            "releases": [
                self._release_dict(item, active_id=skill.active_release_id) for item in releases
            ],
            "audits": [
                {
                    "id": item.id,
                    "action": item.action,
                    "release_id": item.release_id,
                    "revision": item.revision,
                    "details": item.details,
                    "actor": item.actor,
                    "created_at": item.created_at,
                }
                for item in audits
            ],
            "readonly": False,
        }

    async def create(self, *, skill_id: str, name: str, description: str) -> dict[str, object]:
        if skill_id in self._registry.builtin_ids() or await self._repository.get_skill(skill_id):
            raise SkillConflictError(f"Skill id already exists: {skill_id}")
        content = SkillDraftContent.starter(
            skill_id=skill_id, name=name, description=description
        )
        await self._repository.create_skill(
            skill_id=skill_id,
            name=name,
            description=description,
            content=content.model_dump(mode="json"),
            content_hash=self.content_hash(content),
        )
        await self._repository.add_audit(skill_id=skill_id, action="create", revision=1)
        return await self.get_detail(skill_id)

    async def save_draft(
        self, *, skill_id: str, expected_revision: int, content: SkillDraftContent
    ) -> dict[str, object]:
        self._require_managed(skill_id)
        try:
            content.validate_paths()
            manifest = content.package_manifest()
            package_name, description, _ = content.skill_metadata()
        except ValueError as exc:
            raise SkillServiceError(str(exc)) from exc
        if manifest.id != skill_id:
            raise SkillServiceError("manifest.yaml id cannot be changed")
        if package_name != skill_id:
            raise SkillServiceError("SKILL.md name must match the Skill id")
        content_hash = self.content_hash(content)
        saved = await self._repository.save_draft(
            skill_id=skill_id,
            expected_revision=expected_revision,
            content=content.model_dump(mode="json"),
            content_hash=content_hash,
            name=content.display_name(),
            description=description,
        )
        if saved is None:
            current = await self._repository.get_draft(skill_id)
            raise SkillConflictError(
                "Draft revision conflict",
                current_revision=current.revision if current else None,
            )
        await self._repository.add_audit(skill_id=skill_id, action="save", revision=saved.revision)
        return self._draft_dict(saved, skill_id=skill_id)

    async def validate_draft(
        self, *, skill_id: str, expected_revision: int
    ) -> SkillValidationReport:
        self._require_managed(skill_id)
        draft = await self._repository.get_draft(skill_id)
        if draft is None:
            raise SkillNotFoundError(f"Draft not found: {skill_id}")
        if draft.revision != expected_revision:
            raise SkillConflictError("Draft revision conflict", current_revision=draft.revision)
        content = SkillDraftContent.from_storage(draft.content, skill_id=skill_id)
        report = self._validator.validate(
            skill_id=skill_id, content=content, content_hash=draft.content_hash
        )
        await self._repository.save_validation(
            skill_id=skill_id,
            expected_revision=expected_revision,
            validated_hash=draft.content_hash,
            report=report.model_dump(mode="json"),
        )
        await self._repository.add_audit(
            skill_id=skill_id,
            action="validate",
            revision=draft.revision,
            details={"valid": report.valid, "issue_count": len(report.issues)},
        )
        return report

    async def publish(
        self,
        *,
        skill_id: str,
        version: str,
        expected_revision: int,
        activate: bool,
        release_notes: str | None,
    ) -> dict[str, object]:
        self._require_managed(skill_id)
        if not SEMVER_PATTERN.fullmatch(version):
            raise SkillServiceError("Version must be valid SemVer")
        draft = await self._repository.get_draft(skill_id)
        if draft is None:
            raise SkillNotFoundError(f"Draft not found: {skill_id}")
        if draft.revision != expected_revision:
            raise SkillConflictError("Draft revision conflict", current_revision=draft.revision)
        if draft.validated_hash != draft.content_hash or not draft.validation_report:
            raise SkillServiceError("Current draft must be validated before publishing")
        report = SkillValidationReport.model_validate(draft.validation_report)
        if not report.valid:
            raise SkillServiceError("Draft validation contains errors")
        if any(item.version == version for item in await self._repository.list_releases(skill_id)):
            raise SkillServiceError(f"Version already exists: {version}")
        content = SkillDraftContent.from_storage(draft.content, skill_id=skill_id)
        if content.package_manifest().version != version:
            raise SkillServiceError(
                "Publish version must match version in manifest.yaml"
            )
        release = await self._repository.publish(
            skill_id=skill_id,
            version=version,
            content=draft.content,
            checksum=draft.content_hash,
            release_notes=release_notes,
            activate=activate,
        )
        if release is None:
            raise SkillServiceError("Unable to publish release")
        await self._repository.add_audit(
            skill_id=skill_id,
            action="publish",
            release_id=release.id,
            revision=draft.revision,
            details={"version": version, "activate": activate},
        )
        if activate:
            await self.refresh_registry()
        response = self._release_dict(
            release, active_id=release.id if activate else None
        )
        assert response is not None
        return response

    async def activate_release(self, *, skill_id: str, release_id: str) -> dict[str, object]:
        self._require_managed(skill_id)
        skill = await self._repository.get_skill(skill_id)
        previous_release = (
            await self._repository.get_release(skill.active_release_id)
            if skill and skill.active_release_id
            else None
        )
        activated = await self._repository.activate_release(
            skill_id=skill_id, release_id=release_id
        )
        if activated is None:
            raise SkillNotFoundError("Release not found for skill")
        _, release = activated
        action = (
            "rollback"
            if previous_release
            and self._semver_tuple(release.version) < self._semver_tuple(previous_release.version)
            else "activate"
        )
        await self._repository.add_audit(
            skill_id=skill_id,
            action=action,
            release_id=release.id,
            details={
                "previous_version": previous_release.version if previous_release else None,
                "active_version": release.version,
            },
        )
        await self.refresh_registry()
        return {
            "previous_version": previous_release.version if previous_release else None,
            "active_version": release.version,
            "activated_at": datetime.now(UTC),
        }

    async def set_enabled(self, skill_id: str, enabled: bool) -> dict[str, object]:
        try:
            canonical_id = self._registry.canonical_id(skill_id)
        except SkillError:
            canonical_id = skill_id
        if (
            canonical_id not in self._registry.builtin_ids()
            and not await self._repository.get_skill(canonical_id)
        ):
            raise SkillNotFoundError(f"Unknown skill: {skill_id}")
        await self._repository.set_enabled(canonical_id, enabled)
        await self._repository.add_audit(
            skill_id=canonical_id, action="enable" if enabled else "disable"
        )
        await self.refresh_registry()
        info = next(
            (item for item in self._registry.list_skills() if item.id == canonical_id),
            None,
        )
        if info is None:
            raise SkillServiceError("Skill must have an active release before it can be enabled")
        return info.model_dump(mode="json")

    async def clone(
        self,
        *,
        skill_id: str,
        new_id: str,
        new_name: str,
        release_id: str | None = None,
    ) -> dict[str, object]:
        if new_id in self._registry.builtin_ids() or await self._repository.get_skill(new_id):
            raise SkillConflictError(f"Skill id already exists: {new_id}")
        if skill_id in self._registry.builtin_ids():
            loaded = self._registry.get_loaded(skill_id)
            content = SkillDraftContent(files=loaded.package_files).clone_as(
                skill_id=new_id, name=new_name
            )
        else:
            source = (
                await self._repository.get_release(release_id)
                if release_id
                else await self._repository.get_draft(skill_id)
            )
            if source is None or getattr(source, "skill_id", skill_id) != skill_id:
                raise SkillNotFoundError("Clone source not found")
            content = SkillDraftContent.from_storage(
                source.content, skill_id=skill_id
            ).clone_as(
                skill_id=new_id, name=new_name
            )
        _, description, _ = content.skill_metadata()
        await self._repository.create_skill(
            skill_id=new_id,
            name=new_name,
            description=description,
            content=content.model_dump(mode="json"),
            content_hash=self.content_hash(content),
        )
        await self._repository.add_audit(
            skill_id=new_id, action="clone", revision=1, details={"source_skill_id": skill_id}
        )
        return await self.get_detail(new_id)

    async def archive(self, *, skill_id: str, expected_revision: int) -> dict[str, object]:
        self._require_managed(skill_id)
        draft = await self._repository.get_draft(skill_id)
        if draft is None or draft.revision != expected_revision:
            raise SkillConflictError(
                "Draft revision conflict",
                current_revision=draft.revision if draft else None,
            )
        skill = await self._repository.archive(skill_id)
        if skill is None:
            raise SkillNotFoundError(f"Unknown skill: {skill_id}")
        await self._repository.add_audit(
            skill_id=skill_id, action="archive", revision=draft.revision
        )
        await self.refresh_registry()
        return {"id": skill.id, "lifecycle_status": skill.lifecycle_status, "enabled": False}

    def _require_managed(self, skill_id: str) -> None:
        if skill_id in self._registry.builtin_ids():
            raise SkillServiceError("Built-in skills are read-only")

    @staticmethod
    def content_hash(content: SkillDraftContent) -> str:
        serialized = json.dumps(
            content.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _draft_dict(
        draft: SkillDraftRecord, *, skill_id: str
    ) -> dict[str, object]:
        return {
            "revision": draft.revision,
            "content": SkillDraftContent.from_storage(
                draft.content, skill_id=skill_id
            ).model_dump(mode="json"),
            "content_hash": draft.content_hash,
            "validated_hash": draft.validated_hash,
            "validation_report": draft.validation_report,
            "updated_at": draft.updated_at,
        }

    @staticmethod
    def _release_dict(
        release: SkillReleaseRecord | None, *, active_id: str | None
    ) -> dict[str, object] | None:
        if release is None:
            return None
        return {
            "id": release.id,
            "version": release.version,
            "checksum": release.checksum,
            "release_notes": release.release_notes,
            "published_at": release.published_at,
            "active": release.id == active_id,
        }

    @staticmethod
    def _semver_tuple(version: str) -> tuple[int, int, int]:
        match = SEMVER_PATTERN.fullmatch(version)
        if match is None:
            return (0, 0, 0)
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
