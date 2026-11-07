#!/usr/bin/env python3
"""Check that direct preview build pins match the hash-locked install set."""
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*)$"
)
LOCK_ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)\s*\\?$")
HASH = re.compile(r"^--hash=sha256:[0-9a-f]{64}$")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_direct_requirements(path: Path) -> dict[str, str]:
    requirements = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        match = REQUIREMENT.fullmatch(line)
        if not match:
            raise ValueError(f"{path.name}:{line_number}: expected an exact package pin.")
        name, version = match.groups()
        key = normalize_name(name)
        if key in requirements:
            raise ValueError(f"{path.name}:{line_number}: duplicate package {name}.")
        requirements[key] = version
    return requirements


def read_locked_requirements(path: Path) -> dict[str, str]:
    requirements = {}
    current = None
    hash_count = 0

    def finish() -> None:
        if current is not None and hash_count == 0:
            raise ValueError(f"{path.name}: {current[0]} has no SHA-256 artifact hash.")

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[0].isspace():
            line = raw_line.strip()
            if line.endswith("\\"):
                line = line[:-1].rstrip()
            if not HASH.fullmatch(line):
                raise ValueError(f"{path.name}:{line_number}: invalid SHA-256 hash entry.")
            hash_count += 1
            continue

        finish()
        match = LOCK_ENTRY.fullmatch(raw_line)
        if not match:
            raise ValueError(f"{path.name}:{line_number}: expected an exact package pin.")
        name, version = match.groups()
        key = normalize_name(name)
        if key in requirements:
            raise ValueError(f"{path.name}:{line_number}: duplicate package {name}.")
        requirements[key] = version
        current = (name, version)
        hash_count = 0

    finish()
    if not requirements:
        raise ValueError(f"{path.name}: no packages were locked.")
    return requirements


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Preview build requirements are locked for Python 3.12.")

    direct_path = ROOT / "requirements-preview-build.txt"
    lock_path = ROOT / "requirements-preview-build.lock"
    direct = read_direct_requirements(direct_path)
    locked = read_locked_requirements(lock_path)
    for name, version in direct.items():
        if locked.get(name) != version:
            raise SystemExit(f"{name}=={version} is missing or differs in {lock_path.name}.")

    print(f"Verified {len(direct)} direct pins against {len(locked)} hashed packages.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
