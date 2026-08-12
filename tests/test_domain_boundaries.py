from __future__ import annotations

import ast

from snow_grass.core.config import PROJECT_ROOT


def test_activity_domain_is_decoupled() -> None:
    domain = PROJECT_ROOT / "src/snow_grass/activity"
    forbidden = {
        "snow_grass.agent",
        "snow_grass.api",
        "snow_grass.memory",
        "snow_grass.pet",
        "snow_grass.persistence.repository",
    }
    violations: list[str] = []
    for path in domain.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(prefix) for prefix in forbidden):
                    violations.append(f"{path.name}: {node.module}")
    assert violations == []
