from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from snow_grass.persistence.database import Database
from snow_grass.workflow.data_migration import WorkflowV2DataMigration


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def run(args: argparse.Namespace) -> dict[str, object]:
    database_path = Path(args.database).resolve()
    backup_path = Path(args.backup_path).resolve() if args.backup_path else None
    if args.mode == "backup":
        if backup_path is None:
            raise ValueError("--backup-path is required for backup")
        shutil.copy2(database_path, backup_path)
        return {"backup": str(backup_path), "sha256": checksum(backup_path)}
    if args.mode == "apply" and (backup_path is None or not backup_path.is_file()):
        raise ValueError("apply requires a readable --backup-path")
    temporary = tempfile.TemporaryDirectory() if args.mode == "dry-run" else None
    effective_path = database_path
    if temporary is not None:
        effective_path = Path(temporary.name) / database_path.name
        shutil.copy2(database_path, effective_path)
    database = Database(f"sqlite+aiosqlite:///{effective_path}")
    try:
        await database.create_schema()
        migration = WorkflowV2DataMigration(database.session_factory)
        report = await (
            migration.verify()
            if args.mode == "verify"
            else migration.migrate(dry_run=False)
        )
        result = report.__dict__.copy()
        if backup_path is not None:
            result["backup_sha256"] = checksum(backup_path)
        return result
    finally:
        await database.dispose()
        if temporary is not None:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate workflow graphs to component DAG v2")
    parser.add_argument("mode", choices=("backup", "dry-run", "apply", "verify"))
    parser.add_argument("--database", required=True)
    parser.add_argument("--backup-path")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
