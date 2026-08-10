from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SkillSelection(BaseModel):
    mode: str = "auto"
    keywords: list[str] = Field(default_factory=list)


class SkillModels(BaseModel):
    allowed: list[str] = Field(default_factory=list)


class SkillTools(BaseModel):
    allowed: list[str] = Field(default_factory=list)


class SkillLimits(BaseModel):
    max_steps: int = Field(default=8, ge=1, le=50)
    timeout_seconds: int = Field(default=120, ge=1, le=1800)


class SkillCache(BaseModel):
    enabled: bool = False
    scope: Literal["run", "session", "workspace"] = "session"
    ttl_seconds: int = Field(default=600, ge=1, le=86_400)
    error_ttl_seconds: int = Field(default=20, ge=0, le=3_600)


class SkillManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    name: str
    version: str
    description: str
    enabled: bool = True
    aliases: list[str] = Field(default_factory=list)
    selection: SkillSelection = Field(default_factory=SkillSelection)
    models: SkillModels = Field(default_factory=SkillModels)
    tools: SkillTools = Field(default_factory=SkillTools)
    limits: SkillLimits = Field(default_factory=SkillLimits)
    cache: SkillCache = Field(default_factory=SkillCache)


class SkillPackageManifest(BaseModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: str = Field(
        default="0.0.0",
        pattern=(
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
        ),
    )
    enabled: bool = True
    aliases: list[str] = Field(default_factory=list)
    selection: SkillSelection = Field(default_factory=SkillSelection)
    models: SkillModels = Field(default_factory=SkillModels)
    tools: SkillTools = Field(default_factory=SkillTools)
    limits: SkillLimits = Field(default_factory=SkillLimits)
    cache: SkillCache = Field(default_factory=SkillCache)


class SkillInfo(BaseModel):
    id: str
    name: str
    version: str
    description: str
    enabled: bool
    allowed_models: list[str]
    allowed_tools: list[str]


class LoadedSkill(BaseModel):
    manifest: SkillManifest
    instructions: str
    package_files: dict[str, str] = Field(default_factory=dict)

    def info(self) -> SkillInfo:
        return SkillInfo(
            id=self.manifest.id,
            name=self.manifest.name,
            version=self.manifest.version,
            description=self.manifest.description,
            enabled=self.manifest.enabled,
            allowed_models=self.manifest.models.allowed,
            allowed_tools=self.manifest.tools.allowed,
        )


class SkillTestCase(BaseModel):
    input: str = Field(min_length=1, max_length=10_000)
    model_id: str = Field(min_length=1, max_length=120)
    expected_selected: bool = True


class SkillDraftContent(BaseModel):
    """A complete, UTF-8 text Skill package keyed by relative POSIX path."""

    files: dict[str, str] = Field(min_length=2, max_length=200)

    @classmethod
    def starter(cls, *, skill_id: str, name: str, description: str) -> SkillDraftContent:
        manifest = SkillPackageManifest(id=skill_id)
        quoted_name = json.dumps(skill_id, ensure_ascii=False)
        quoted_description = json.dumps(description, ensure_ascii=False)
        return cls(
            files={
                "manifest.yaml": yaml.safe_dump(
                    manifest.model_dump(mode="json"),
                    allow_unicode=True,
                    sort_keys=False,
                ),
                "SKILL.md": (
                    "---\n"
                    f"name: {quoted_name}\n"
                    f"description: {quoted_description}\n"
                    "---\n\n"
                    f"# {name}\n\n"
                    "在这里编写技能的执行流程、约束和输出要求。\n"
                ),
                "agents/openai.yaml": yaml.safe_dump(
                    {
                        "interface": {
                            "display_name": name,
                            "short_description": description[:80],
                            "default_prompt": f"使用 {name} 完成当前任务。",
                        }
                    },
                    allow_unicode=True,
                    sort_keys=False,
                ),
            }
        )

    @classmethod
    def from_storage(
        cls, raw: dict[str, object], *, skill_id: str
    ) -> SkillDraftContent:
        """Load the package format or migrate a pre-package declarative draft in memory."""
        if "files" in raw:
            return cls.model_validate(raw)
        name = str(raw.get("name") or skill_id)
        description = str(raw.get("description") or "Migrated Skill package")
        package = cls.starter(skill_id=skill_id, name=name, description=description)
        manifest = SkillPackageManifest(
            id=skill_id,
            selection=SkillSelection.model_validate(raw.get("selection") or {}),
            models=SkillModels.model_validate(raw.get("models") or {}),
            tools=SkillTools.model_validate(raw.get("tools") or {}),
            limits=SkillLimits.model_validate(raw.get("limits") or {}),
        )
        files = dict(package.files)
        files["manifest.yaml"] = yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        )
        _, _, starter_body = package.skill_metadata()
        instructions = str(raw.get("instructions") or starter_body).strip()
        files["SKILL.md"] = (
            "---\n"
            f"name: {json.dumps(skill_id, ensure_ascii=False)}\n"
            "description: "
            f"{json.dumps(description, ensure_ascii=False)}\n"
            "---\n\n"
            f"{instructions}\n"
        )
        test_cases = raw.get("test_cases")
        if isinstance(test_cases, list) and test_cases:
            files["tests/selection.yaml"] = yaml.safe_dump(
                test_cases, allow_unicode=True, sort_keys=False
            )
        return cls(files=files)

    @classmethod
    def from_directory(cls, root: Path) -> SkillDraftContent:
        files: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            try:
                files[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Skill package file must be UTF-8 text: {relative}"
                ) from exc
        return cls(files=files)

    def validate_paths(self) -> None:
        for path in self.files:
            pure = PurePosixPath(path)
            if (
                not path
                or len(path) > 240
                or "\\" in path
                or pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise ValueError(f"Unsafe package path: {path}")
        for required in ("manifest.yaml", "SKILL.md"):
            if required not in self.files:
                raise ValueError(f"Missing required package file: {required}")

    def package_manifest(self) -> SkillPackageManifest:
        raw = yaml.safe_load(self.files.get("manifest.yaml", "")) or {}
        return SkillPackageManifest.model_validate(raw)

    def skill_metadata(self) -> tuple[str, str, str]:
        document = self.files.get("SKILL.md", "")
        if not document.startswith("---\n"):
            raise ValueError("SKILL.md must start with YAML frontmatter")
        closing = document.find("\n---\n", 4)
        if closing < 0:
            raise ValueError("SKILL.md frontmatter is not closed")
        metadata = yaml.safe_load(document[4:closing]) or {}
        if not isinstance(metadata, dict) or set(metadata) != {"name", "description"}:
            raise ValueError(
                "SKILL.md frontmatter must contain only name and description"
            )
        name = str(metadata.get("name", "")).strip()
        description = str(metadata.get("description", "")).strip()
        body = document[closing + 5 :].strip()
        if not name or len(name) > 120:
            raise ValueError("SKILL.md name must contain 1-120 characters")
        if not description or len(description) > 1_000:
            raise ValueError("SKILL.md description must contain 1-1000 characters")
        return name, description, body

    def display_name(self) -> str:
        serialized = self.files.get("agents/openai.yaml")
        if serialized:
            raw = yaml.safe_load(serialized) or {}
            if isinstance(raw, dict) and isinstance(raw.get("interface"), dict):
                value = str(raw["interface"].get("display_name") or "").strip()
                if value:
                    return value
        name, _, _ = self.skill_metadata()
        return name

    def selection_tests(self) -> list[SkillTestCase]:
        serialized = self.files.get("tests/selection.yaml")
        if not serialized:
            return []
        raw = yaml.safe_load(serialized) or []
        if not isinstance(raw, list):
            raise ValueError("tests/selection.yaml must contain a YAML list")
        return [SkillTestCase.model_validate(item) for item in raw]

    def with_version(self, version: str) -> SkillDraftContent:
        manifest = self.package_manifest().model_copy(update={"version": version})
        files = dict(self.files)
        files["manifest.yaml"] = yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        )
        return self.model_copy(update={"files": files})

    def clone_as(self, *, skill_id: str, name: str) -> SkillDraftContent:
        manifest = self.package_manifest().model_copy(
            update={"id": skill_id, "version": "0.0.0", "enabled": True}
        )
        _, description, body = self.skill_metadata()
        files = dict(self.files)
        files["manifest.yaml"] = yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        )
        files["SKILL.md"] = (
            "---\n"
            f"name: {json.dumps(skill_id, ensure_ascii=False)}\n"
            "description: "
            f"{json.dumps(description, ensure_ascii=False)}\n"
            "---\n\n"
            f"{body}\n"
        )
        agents = files.get("agents/openai.yaml")
        if agents:
            raw = yaml.safe_load(agents) or {}
            interface = raw.setdefault("interface", {})
            interface["display_name"] = name
            interface["default_prompt"] = f"使用 {name} 完成当前任务。"
            files["agents/openai.yaml"] = yaml.safe_dump(
                raw, allow_unicode=True, sort_keys=False
            )
        return self.model_copy(update={"files": files})

    def to_loaded_skill(
        self, *, skill_id: str, version: str, enabled: bool
    ) -> LoadedSkill:
        self.validate_paths()
        package_manifest = self.package_manifest()
        if package_manifest.id != skill_id:
            raise ValueError(
                f"manifest.yaml id {package_manifest.id!r} does not match {skill_id!r}"
            )
        package_name, description, instructions = self.skill_metadata()
        if package_name != skill_id:
            raise ValueError(f"SKILL.md name must match skill id {skill_id!r}")
        return LoadedSkill(
            manifest=SkillManifest(
                id=skill_id,
                name=self.display_name(),
                version=version,
                description=description,
                enabled=enabled,
                aliases=package_manifest.aliases,
                selection=package_manifest.selection,
                models=package_manifest.models,
                tools=package_manifest.tools,
                limits=package_manifest.limits,
                cache=package_manifest.cache,
            ),
            instructions=instructions,
            package_files=dict(self.files),
        )


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    field: str
    message: str


class SkillTestResult(BaseModel):
    input: str
    model_id: str
    expected_selected: bool
    actual_selected: bool
    passed: bool


class SkillValidationReport(BaseModel):
    valid: bool
    validated_hash: str
    issues: list[ValidationIssue] = Field(default_factory=list)
    test_results: list[SkillTestResult] = Field(default_factory=list)
