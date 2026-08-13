from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import monotonic

from pydantic import BaseModel, Field

from snow_grass.skills.schema import LoadedSkill

RUN_SKILL_SCRIPT_TOOL = "run_skill_script"


class SkillScriptExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SkillScriptResult(BaseModel):
    ok: bool
    script: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    duration_ms: int = Field(default=0, ge=0)

    def to_tool_message(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


class SkillScriptExecutor:
    """Runs reviewed Python files from the active immutable Skill package.

    This is an execution policy boundary, not an OS-level sandbox. Only trusted Skills should be
    published while script execution is enabled.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        timeout_seconds: int,
        max_output_chars: int,
        max_args: int = 32,
        max_arg_chars: int = 2_000,
    ) -> None:
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars
        self._max_args = max_args
        self._max_arg_chars = max_arg_chars

    def list_scripts(self, skill: LoadedSkill) -> list[str]:
        if not self._enabled:
            return []
        return sorted(
            path
            for path in skill.package_files
            if self._is_safe_package_path(path)
            and path.startswith("scripts/")
            and path.endswith(".py")
        )

    async def execute(
        self, *, skill: LoadedSkill, script: str, args: Sequence[str]
    ) -> SkillScriptResult:
        started_at = monotonic()
        self._validate_request(skill=skill, script=script, args=args)
        timeout_seconds = min(
            self._timeout_seconds,
            skill.manifest.limits.timeout_seconds,
        )

        with TemporaryDirectory(prefix="snow-grass-skill-") as temporary_directory:
            package_root = Path(temporary_directory)
            self._materialize_package(skill=skill, package_root=package_root)
            script_path = self._resolved_script_path(package_root, script)

            environment = self._subprocess_environment()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(script_path),
                *args,
                cwd=package_root,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_task = asyncio.create_task(self._read_limited(process.stdout))
            stderr_task = asyncio.create_task(self._read_limited(process.stderr))
            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                process.kill()
                await process.wait()

            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
            duration_ms = max(0, round((monotonic() - started_at) * 1_000))
            return SkillScriptResult(
                ok=not timed_out and process.returncode == 0,
                script=script,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                truncated=stdout_truncated or stderr_truncated,
                duration_ms=duration_ms,
            )

    def _validate_request(
        self, *, skill: LoadedSkill, script: str, args: Sequence[str]
    ) -> None:
        if not self._enabled:
            raise SkillScriptExecutionError(
                "script_execution_disabled", "Skill script execution is disabled"
            )
        if script not in self.list_scripts(skill):
            raise SkillScriptExecutionError(
                "script_not_allowed", "The requested script is not in the active Skill package"
            )
        if len(args) > self._max_args:
            raise SkillScriptExecutionError("too_many_arguments", "Too many script arguments")
        if any(
            not isinstance(argument, str)
            or len(argument) > self._max_arg_chars
            or "\x00" in argument
            for argument in args
        ):
            raise SkillScriptExecutionError(
                "invalid_argument", "Script arguments must be bounded strings without NUL bytes"
            )

    @classmethod
    def _materialize_package(cls, *, skill: LoadedSkill, package_root: Path) -> None:
        resolved_root = package_root.resolve()
        for relative_path, serialized in skill.package_files.items():
            if not cls._is_safe_package_path(relative_path):
                raise SkillScriptExecutionError(
                    "unsafe_package_path", "The active Skill package contains an unsafe path"
                )
            destination = (package_root / PurePosixPath(relative_path)).resolve()
            if not destination.is_relative_to(resolved_root):
                raise SkillScriptExecutionError(
                    "unsafe_package_path", "The active Skill package contains an unsafe path"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized, encoding="utf-8")

    @staticmethod
    def _resolved_script_path(package_root: Path, script: str) -> Path:
        resolved_root = package_root.resolve()
        script_path = (package_root / PurePosixPath(script)).resolve()
        if not script_path.is_relative_to(resolved_root):
            raise SkillScriptExecutionError("unsafe_script_path", "Script path is unsafe")
        return script_path

    async def _read_limited(
        self, stream: asyncio.StreamReader
    ) -> tuple[str, bool]:
        chunks: list[bytes] = []
        captured = 0
        truncated = False
        byte_limit = self._max_output_chars * 4
        while chunk := await stream.read(8_192):
            remaining = byte_limit - captured
            if remaining > 0:
                accepted = chunk[:remaining]
                chunks.append(accepted)
                captured += len(accepted)
            if len(chunk) > max(remaining, 0):
                truncated = True
        decoded = b"".join(chunks).decode("utf-8", errors="replace")
        if len(decoded) > self._max_output_chars:
            return decoded[: self._max_output_chars], True
        return decoded, truncated

    @staticmethod
    def _is_safe_package_path(path: str) -> bool:
        pure = PurePosixPath(path)
        return bool(
            path
            and "\\" not in path
            and not pure.is_absolute()
            and all(part not in {"", ".", ".."} for part in pure.parts)
        )

    @staticmethod
    def _subprocess_environment() -> dict[str, str]:
        environment = {
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment
