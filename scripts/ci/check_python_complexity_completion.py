#!/usr/bin/env python3
"""Check completion after the full measured complexipy ratchet has run."""

import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASELINE_NAME = "complexipy-snapshot.json"
THRESHOLD = 15
HISTORICAL_RELEASE_DIRECTORIES = frozenset(
    {
        "releases/web-decision-0.1.0",
        "releases/web-decision-0.2.0",
        "releases/web-decision-0.3.0",
    }
)


@dataclass(frozen=True)
class Finding:
    path: str
    name: str
    complexity: int


def normalized_python_path(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label} path: {value}")
    normalized = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\0" in value
        or re.match(r"^[A-Za-z]:", value)
        or normalized.is_absolute()
        or normalized.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or normalized.suffix != ".py"
    ):
        raise ValueError(f"Invalid {label} path: {value}")
    return value


def validate_snapshot(value: object) -> list[Finding]:
    if not isinstance(value, list):
        raise ValueError("Python complexity baseline must be a list.")

    findings: list[Finding] = []
    paths: set[str] = set()
    finding_keys: set[tuple[str, str]] = set()
    for entry in value:
        name, functions = validate_file_entry(entry)
        if name.casefold() in paths:
            raise ValueError(
                f"Python complexity baseline contains a duplicate path: {name}"
            )
        paths.add(name.casefold())
        findings.extend(
            validate_function(function, name, finding_keys) for function in functions
        )
    return findings


def validate_file_entry(entry: object) -> tuple[str, list[object]]:
    if not isinstance(entry, dict) or set(entry) != {"path", "file_name", "functions"}:
        raise ValueError("Python complexity baseline contains a malformed file entry.")
    name = normalized_python_path(entry.get("path"), "Python complexity baseline")
    if entry.get("file_name") != PurePosixPath(name).name:
        raise ValueError(f"Python complexity baseline has an invalid file name: {name}")
    functions = entry.get("functions")
    if not isinstance(functions, list):
        raise ValueError(f"Python complexity baseline has invalid functions in {name}")
    return name, functions


def validate_function(
    function: object, path: str, finding_keys: set[tuple[str, str]]
) -> Finding:
    if not isinstance(function, dict) or set(function) != {"name", "complexity"}:
        raise ValueError(
            f"Python complexity baseline has a malformed function in {path}"
        )
    name = function.get("name")
    complexity = function.get("complexity")
    if (
        not isinstance(name, str)
        or not name.strip()
        or type(complexity) is not int
        or complexity <= THRESHOLD
    ):
        raise ValueError(f"Python complexity baseline has an invalid finding in {path}")
    key = (path, name)
    if key in finding_keys:
        raise ValueError(f"Python complexity baseline has a duplicate finding: {key}")
    finding_keys.add(key)
    return Finding(path, name, complexity)


def load_snapshot(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"Python complexity baseline is missing or unreadable: {path}"
        ) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot parse Python complexity baseline: {error}") from error
    return validate_snapshot(value)


def is_historical_release_path(name: str) -> bool:
    return any(
        name.startswith(f"{directory}/") for directory in HISTORICAL_RELEASE_DIRECTORIES
    )


def completion_violations(
    findings: list[Finding], source_paths: set[str]
) -> list[Finding]:
    for finding in findings:
        if finding.path not in source_paths:
            raise ValueError(
                f"Python complexity baseline references a missing or unscanned source: {finding.path}"
            )
    return [
        finding
        for finding in findings
        if not is_historical_release_path(finding.path)
        and finding.complexity > THRESHOLD
    ]


def require_complete(findings: list[Finding], source_paths: set[str]) -> None:
    violations = completion_violations(findings, source_paths)
    if violations:
        details = "\n".join(
            f"  {finding.path} :: {finding.name} ({finding.complexity})"
            for finding in violations
        )
        raise ValueError(
            f"Managed Python source remains above cognitive complexity {THRESHOLD}:\n{details}"
        )


def working_tree_python_sources(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "*.py",
            ],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"Could not list tracked Python sources: {error}") from error
    if result.returncode != 0:
        raise ValueError(
            "Could not list tracked Python sources: "
            + result.stderr.decode("utf-8", errors="replace")
        )

    sources: set[str] = set()
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = normalized_python_path(
            os.fsdecode(raw_name), "working tree Python source"
        )
        source = root.joinpath(*name.split("/"))
        try:
            mode = source.lstat().st_mode
        except OSError as error:
            raise ValueError(f"Python source is missing: {name}") from error
        if source.is_symlink() or not stat.S_ISREG(mode):
            raise ValueError(f"Python source is not a regular file: {name}")
        sources.add(name)
    if not sources:
        raise ValueError("No Python sources were found.")
    return sources


def run(root: Path = ROOT) -> None:
    baseline = root / BASELINE_NAME
    findings = load_snapshot(baseline)
    require_complete(findings, working_tree_python_sources(root))
    print(
        "complexipy complexity completion gate passed: "
        f"no managed baseline findings above {THRESHOLD}."
    )


if __name__ == "__main__":
    try:
        run()
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
