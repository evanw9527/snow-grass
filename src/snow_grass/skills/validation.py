from __future__ import annotations

import ast

import yaml

from snow_grass.core.config import Settings
from snow_grass.memory.security import MemorySecurity
from snow_grass.providers.registry import ProviderRegistry
from snow_grass.skills.matching import selection_score
from snow_grass.skills.schema import (
    SkillDraftContent,
    SkillTestResult,
    SkillValidationReport,
    ValidationIssue,
)
from snow_grass.tools.registry import ToolRegistry


class SkillValidator:
    def __init__(
        self,
        *,
        settings: Settings,
        providers: ProviderRegistry,
        tools: ToolRegistry,
        security: MemorySecurity,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._tools = tools
        self._security = security

    def validate(
        self, *, skill_id: str, content: SkillDraftContent, content_hash: str
    ) -> SkillValidationReport:
        issues: list[ValidationIssue] = []
        try:
            content.validate_paths()
        except ValueError as exc:
            issues.append(self._error("invalid_package_path", "files", str(exc)))

        if len(content.files) > self._settings.skill_max_package_files:
            issues.append(
                self._error(
                    "too_many_files",
                    "files",
                    f"技能包不能超过 {self._settings.skill_max_package_files} 个文件",
                )
            )
        package_chars = sum(len(value) for value in content.files.values())
        if package_chars > self._settings.skill_max_package_chars:
            issues.append(
                self._error(
                    "package_too_large",
                    "files",
                    f"技能包文本不能超过 {self._settings.skill_max_package_chars} 字符",
                )
            )

        for path, serialized in content.files.items():
            if self._security.contains_sensitive(serialized):
                issues.append(self._error("sensitive_content", path, f"{path} 包含疑似密钥或凭据"))

        try:
            manifest = content.package_manifest()
            if manifest.id != skill_id:
                issues.append(
                    self._error(
                        "skill_id_mismatch",
                        "manifest.yaml",
                        f"manifest.yaml 中的 id 必须是 {skill_id}",
                    )
                )
        except ValueError as exc:
            issues.append(self._error("invalid_manifest", "manifest.yaml", str(exc)))
            return self._report(content_hash, issues, [])

        try:
            _, _, instructions = content.skill_metadata()
            if not instructions:
                issues.append(
                    self._error("empty_instructions", "SKILL.md", "SKILL.md 正文不能为空")
                )
            if len(instructions) > self._settings.skill_max_instruction_chars:
                issues.append(
                    self._error(
                        "instructions_too_long",
                        "SKILL.md",
                        "SKILL.md 正文超过允许长度",
                    )
                )
            if len(content.files["SKILL.md"].splitlines()) > 500:
                issues.append(
                    self._warning(
                        "skill_md_too_long",
                        "SKILL.md",
                        "SKILL.md 超过 500 行，建议将细节拆到 references/",
                    )
                )
        except ValueError as exc:
            issues.append(self._error("invalid_skill_md", "SKILL.md", str(exc)))

        agents_metadata = content.files.get("agents/openai.yaml")
        if agents_metadata:
            try:
                agents = yaml.safe_load(agents_metadata) or {}
                interface = agents.get("interface") if isinstance(agents, dict) else None
                required = {"display_name", "short_description", "default_prompt"}
                if not isinstance(interface, dict) or not required <= set(interface):
                    raise ValueError(
                        "agents/openai.yaml interface 缺少 display_name、"
                        "short_description 或 default_prompt"
                    )
            except (ValueError, yaml.YAMLError) as exc:
                issues.append(
                    self._error("invalid_agents_metadata", "agents/openai.yaml", str(exc))
                )

        for unnecessary in ("README.md", "CHANGELOG.md", "INSTALLATION_GUIDE.md"):
            if unnecessary in content.files:
                issues.append(
                    self._warning(
                        "extraneous_package_file",
                        unnecessary,
                        f"{unnecessary} 通常不应放入技能包",
                    )
                )

        model_catalog = {model.id: model for model in self._providers.list_models()}
        for model_id in manifest.models.allowed:
            model = model_catalog.get(model_id)
            if model is None:
                issues.append(
                    self._error("unknown_model", "manifest.yaml", f"未知模型：{model_id}")
                )
            elif not model.available:
                issues.append(
                    self._warning(
                        "model_unavailable",
                        "manifest.yaml",
                        f"模型 {model_id} 尚未配置可用 Key",
                    )
                )

        for tool_name in manifest.tools.allowed:
            if not self._tools.list_allowed([tool_name]):
                issues.append(
                    self._error(
                        "unknown_tool",
                        "manifest.yaml",
                        f"工具未在服务端注册：{tool_name}",
                    )
                )

        keywords = [keyword.strip() for keyword in manifest.selection.keywords if keyword.strip()]
        try:
            display_name = content.display_name()
            _, description, _ = content.skill_metadata()
        except ValueError:
            display_name = skill_id
            description = ""
        try:
            test_cases = content.selection_tests()
        except ValueError as exc:
            issues.append(self._error("invalid_selection_tests", "tests/selection.yaml", str(exc)))
            test_cases = []
        if len(test_cases) > self._settings.skill_max_test_cases:
            issues.append(
                self._error(
                    "too_many_test_cases",
                    "tests/selection.yaml",
                    f"测试用例不能超过 {self._settings.skill_max_test_cases} 个",
                )
            )

        test_results: list[SkillTestResult] = []
        for test_case in test_cases:
            supports_model = (
                not manifest.models.allowed or test_case.model_id in manifest.models.allowed
            )
            selected = supports_model and bool(
                selection_score(
                    skill_id=skill_id,
                    name=display_name,
                    description=description,
                    keywords=keywords,
                    intent_rules=manifest.selection.intent_rules,
                    content=test_case.input,
                )
            )
            passed = selected == test_case.expected_selected
            test_results.append(
                SkillTestResult(
                    input=test_case.input,
                    model_id=test_case.model_id,
                    expected_selected=test_case.expected_selected,
                    actual_selected=selected,
                    passed=passed,
                )
            )
            if not passed:
                issues.append(
                    self._error(
                        "selection_test_failed",
                        "tests/selection.yaml",
                        f"选择测试失败：{test_case.input[:60]}",
                    )
                )

        script_paths = [path for path in content.files if path.startswith("scripts/")]
        if manifest.cache.enabled and not script_paths:
            issues.append(
                self._error(
                    "cache_without_script",
                    "manifest.yaml",
                    "启用 cache 时技能包必须包含 scripts/ 下的 Python 脚本",
                )
            )
        for path in script_paths:
            if not path.endswith(".py"):
                issues.append(
                    self._error(
                        "unsupported_script_type",
                        path,
                        "当前运行时只允许 scripts/ 下的 Python .py 文件",
                    )
                )
                continue
            try:
                ast.parse(content.files[path], filename=path)
            except SyntaxError as exc:
                issues.append(
                    self._error(
                        "invalid_python_script",
                        path,
                        f"Python 脚本语法错误（第 {exc.lineno or 0} 行）",
                    )
                )
        if script_paths:
            issues.append(
                self._warning(
                    "scripts_run_as_server_user",
                    "scripts/",
                    "脚本发布后可由 Agent 调用，并以 Snow Grass 服务进程账号运行；只应发布可信代码",
                )
            )

        try:
            content.to_loaded_skill(skill_id=skill_id, version=manifest.version, enabled=True)
        except ValueError as exc:
            issues.append(self._error("invalid_runtime_package", "files", str(exc)))

        return self._report(content_hash, issues, test_results)

    @staticmethod
    def _report(
        content_hash: str,
        issues: list[ValidationIssue],
        test_results: list[SkillTestResult],
    ) -> SkillValidationReport:
        return SkillValidationReport(
            valid=not any(issue.severity == "error" for issue in issues),
            validated_hash=content_hash,
            issues=issues,
            test_results=test_results,
        )

    @staticmethod
    def _error(code: str, field: str, message: str) -> ValidationIssue:
        return ValidationIssue(severity="error", code=code, field=field, message=message)

    @staticmethod
    def _warning(code: str, field: str, message: str) -> ValidationIssue:
        return ValidationIssue(severity="warning", code=code, field=field, message=message)
