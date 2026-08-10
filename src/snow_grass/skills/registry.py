from __future__ import annotations

from pathlib import Path

from snow_grass.skills.matching import selection_score
from snow_grass.skills.schema import LoadedSkill, SkillDraftContent, SkillInfo


class SkillError(ValueError):
    pass


class SkillRegistry:
    def __init__(self, skills: dict[str, LoadedSkill] | None = None) -> None:
        self._builtin_skills = skills or {}
        self._managed_skills: dict[str, LoadedSkill] = {}
        self._enabled_overrides: dict[str, bool] = {}

    @classmethod
    def load(cls, root: Path) -> SkillRegistry:
        skills: dict[str, LoadedSkill] = {}
        if not root.exists():
            return cls(skills)
        for manifest_path in sorted(root.rglob("manifest.yaml")):
            package = SkillDraftContent.from_directory(manifest_path.parent)
            manifest = package.package_manifest()
            if manifest.id in skills:
                raise SkillError(f"Duplicate skill id: {manifest.id}")
            try:
                skills[manifest.id] = package.to_loaded_skill(
                    skill_id=manifest.id,
                    version=manifest.version,
                    enabled=manifest.enabled,
                )
            except ValueError as exc:
                raise SkillError(f"Invalid Skill package {manifest_path.parent}: {exc}") from exc
        return cls(skills)

    def list_skills(self) -> list[SkillInfo]:
        return [skill.info() for skill in self._snapshot().values()]

    def set_enabled(self, skill_id: str, enabled: bool) -> SkillInfo:
        self._require(skill_id)
        self._enabled_overrides = {**self._enabled_overrides, skill_id: enabled}
        return self._require(skill_id).info()

    def replace_managed_snapshot(self, skills: dict[str, LoadedSkill]) -> None:
        if set(skills) & set(self._builtin_skills):
            duplicate = sorted(set(skills) & set(self._builtin_skills))[0]
            raise SkillError(f"Managed skill conflicts with builtin skill: {duplicate}")
        self._managed_skills = dict(skills)

    def apply_enabled_overrides(self, overrides: dict[str, bool]) -> None:
        self._enabled_overrides = dict(overrides)

    def get_loaded(self, skill_id: str) -> LoadedSkill:
        return self._require(skill_id).model_copy(deep=True)

    def canonical_id(self, skill_id: str) -> str:
        skill = self._require(skill_id)
        return skill.manifest.id

    def builtin_ids(self) -> set[str]:
        return set(self._builtin_skills)

    def _snapshot(self) -> dict[str, LoadedSkill]:
        merged = {**self._builtin_skills, **self._managed_skills}
        snapshot: dict[str, LoadedSkill] = {}
        for skill_id, skill in merged.items():
            copied = skill.model_copy(deep=True)
            override_keys = [skill_id, *skill.manifest.aliases]
            override = next(
                (
                    self._enabled_overrides[key]
                    for key in override_keys
                    if key in self._enabled_overrides
                ),
                None,
            )
            if override is not None:
                copied.manifest.enabled = override
            snapshot[skill_id] = copied
        return snapshot

    def select(
        self, *, content: str, model_id: str, explicit_skill_id: str | None = None
    ) -> LoadedSkill | None:
        if explicit_skill_id:
            skill = self._require(explicit_skill_id)
            self._validate_available(skill, model_id)
            return skill

        normalized = content.casefold()
        candidates: list[tuple[int, LoadedSkill]] = []
        for skill in self._snapshot().values():
            if not skill.manifest.enabled or not self._supports_model(skill, model_id):
                continue
            if skill.manifest.selection.mode != "auto":
                continue
            score = selection_score(
                skill_id=skill.manifest.id,
                name=skill.manifest.name,
                description=skill.manifest.description,
                keywords=skill.manifest.selection.keywords,
                content=normalized,
            )
            if score:
                candidates.append((score, skill))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _require(self, skill_id: str) -> LoadedSkill:
        snapshot = self._snapshot()
        skill = snapshot.get(skill_id)
        if skill is None:
            skill = next(
                (
                    candidate
                    for candidate in snapshot.values()
                    if skill_id in candidate.manifest.aliases
                ),
                None,
            )
        if skill is None:
            raise SkillError(f"Unknown skill: {skill_id}")
        return skill

    def _validate_available(self, skill: LoadedSkill, model_id: str) -> None:
        if not skill.manifest.enabled:
            raise SkillError(f"Skill {skill.manifest.id} is disabled")
        if not self._supports_model(skill, model_id):
            raise SkillError(f"Skill {skill.manifest.id} does not support model {model_id}")

    @staticmethod
    def _supports_model(skill: LoadedSkill, model_id: str) -> bool:
        allowed = skill.manifest.models.allowed
        return not allowed or model_id in allowed
