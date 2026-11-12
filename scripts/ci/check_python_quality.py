#!/usr/bin/env python3
"""Check Ruff output against exact-hash exceptions for frozen Python sources."""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

if __package__:
    from .git_environment import git_environment
else:
    from git_environment import git_environment

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "ruff-snapshot.json"
SNAPSHOT_ENV = "WEB_DECISION_RUFF_BASELINE"
SAFE_OUTPUT_SHA256 = "d00ed5b9a7a863ff64d402350c4a2c958454838ea6411949d961f0d9a94cb307"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def relative_path(name):
    if not isinstance(name, str):
        raise ValueError(f"Python path must be a string: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or ".." in path.parts
        or "\\" in name
        or not name.endswith(".py")
    ):
        raise ValueError(f"Invalid Python path: {name}")
    return name


def tracked_python_files():
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
        cwd=ROOT,
        env=git_environment(ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit("Could not list Git-tracked Python files.")
    names = [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
    paths = [relative_path(name) for name in names]
    if len({name.casefold() for name in paths}) != len(paths):
        raise SystemExit("Python input paths are duplicated or case-colliding.")
    for name in paths:
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f"Invalid Python input: {name}")
    return sorted(paths)


def source_lock_hashes():
    locked = defaultdict(set)
    for lock_path in sorted(
        (ROOT / "releases").glob("web-decision-*/source-lock.json")
    ):
        document = json.loads(lock_path.read_text(encoding="utf-8"))
        for entry in document.get("files", []):
            name = entry.get("path")
            digest = entry.get("sha256")
            if (
                isinstance(name, str)
                and name.endswith(".py")
                and SHA256.fullmatch(str(digest))
            ):
                locked[name].add(digest)
    return locked


def validate_frozen_entries(snapshot, locked_hashes):
    errors = []
    for name, entry in snapshot.items():
        allowed = locked_hashes.get(name, set())
        if name == "scripts/safe_output.py":
            allowed = {*allowed, SAFE_OUTPUT_SHA256}
        if entry["sha256"] not in allowed:
            errors.append(f"Ruff baseline is not source-locked: {name}")
    return errors


def lint_fingerprint(diagnostic):
    location = diagnostic.get("location")
    if location is None:
        row = diagnostic.get("row")
        column = diagnostic.get("column")
    elif isinstance(location, dict):
        row = location.get("row")
        column = location.get("column")
    else:
        raise ValueError("Ruff diagnostic has an invalid source location.")
    code = diagnostic.get("code")
    message = diagnostic.get("message")
    if (
        not isinstance(code, str)
        or not code
        or not isinstance(message, str)
        or type(row) is not int
        or row < 1
        or type(column) is not int
        or column < 1
    ):
        raise ValueError("Ruff diagnostic has invalid fields.")
    return {"code": code, "row": row, "column": column, "message": message}


def snapshot_files(document):
    if (
        not isinstance(document, dict)
        or set(document) != {"schemaVersion", "files"}
        or type(document["schemaVersion"]) is not int
        or document["schemaVersion"] != 1
        or not isinstance(document["files"], list)
    ):
        raise ValueError("Invalid Ruff snapshot header.")
    return document["files"]


def snapshot_diagnostics(name, items):
    if not isinstance(items, list):
        raise ValueError(f"Invalid Ruff snapshot diagnostics: {name}")
    expected_fields = {"code", "row", "column", "message"}
    if any(
        not isinstance(item, dict) or set(item) != expected_fields for item in items
    ):
        raise ValueError(f"Invalid Ruff snapshot diagnostics: {name}")
    diagnostics = [lint_fingerprint(item) for item in items]
    fingerprints = [json.dumps(item, sort_keys=True) for item in diagnostics]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(f"Duplicate Ruff snapshot diagnostic: {name}")
    return diagnostics


def snapshot_entry(entry):
    if not isinstance(entry, dict) or set(entry) != {
        "path",
        "sha256",
        "format",
        "lint",
    }:
        raise ValueError("Invalid Ruff snapshot file entry.")
    name = relative_path(entry["path"])
    digest = entry["sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError(f"Invalid Ruff snapshot SHA-256: {name}")
    if type(entry["format"]) is not bool:
        raise ValueError(f"Invalid Ruff snapshot diagnostics: {name}")
    diagnostics = snapshot_diagnostics(name, entry["lint"])
    if not entry["format"] and not diagnostics:
        raise ValueError(f"Empty Ruff snapshot entry: {name}")
    return name, {"sha256": digest, "format": entry["format"], "lint": diagnostics}


def validate_snapshot(document):
    snapshot = {}
    for entry in snapshot_files(document):
        name, value = snapshot_entry(entry)
        if name.casefold() in {path.casefold() for path in snapshot}:
            raise ValueError(f"Duplicate Ruff snapshot path: {name}")
        snapshot[name] = value
    return snapshot


def load_snapshot(path):
    return validate_snapshot(json.loads(path.read_text(encoding="utf-8")))


def compare_snapshots(trusted, candidate):
    errors = []
    for name, entry in candidate.items():
        previous = trusted.get(name)
        if previous is None:
            errors.append(f"Ruff baseline added an exception: {name}")
            continue
        if entry["sha256"] != previous["sha256"]:
            errors.append(f"Ruff baseline changed a pinned SHA-256: {name}")
        if entry["format"] and not previous["format"]:
            errors.append(f"Ruff baseline added a format exception: {name}")
        old_lint = {json.dumps(item, sort_keys=True) for item in previous["lint"]}
        new_lint = {json.dumps(item, sort_keys=True) for item in entry["lint"]}
        if not new_lint <= old_lint:
            errors.append(f"Ruff baseline added lint exceptions: {name}")
    return errors


def ruff_commands(files):
    paths = list(files)
    common = [
        "--isolated",
        "--no-respect-gitignore",
        "--no-force-exclude",
        "--target-version",
        "py312",
        "--line-length",
        "88",
        "--no-cache",
        "--output-format",
        "json",
    ]
    return [
        ["ruff", "format", "--check", *common, *paths],
        [
            "ruff",
            "check",
            *common,
            "--select",
            "E4,E7,E9,F,I",
            "--ignore-noqa",
            *paths,
        ],
    ]


def run_json(command):
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode not in (0, 1):
        raise SystemExit(f"Ruff command failed: {' '.join(command)}\n{result.stderr}")
    try:
        output = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        raise SystemExit(f"Ruff returned invalid JSON: {error}") from error
    if not isinstance(output, list):
        raise SystemExit("Ruff returned an invalid diagnostic list.")
    return output


def ruff_path(filename):
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve(strict=True).relative_to(ROOT).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Ruff reported an outside or missing file: {filename}"
        ) from error


def collect_results(format_rows, lint_rows):
    formatted = set()
    linted = defaultdict(list)
    for row in format_rows:
        if row.get("code") != "unformatted" or not isinstance(row.get("filename"), str):
            raise ValueError("Ruff formatter returned an invalid diagnostic.")
        formatted.add(ruff_path(row["filename"]))
    for row in lint_rows:
        name = ruff_path(row.get("filename"))
        linted[name].append(lint_fingerprint(row))
    for diagnostics in linted.values():
        diagnostics.sort(
            key=lambda item: (
                item["code"],
                item["row"],
                item["column"],
                item["message"],
            )
        )
    return formatted, dict(linted)


def verify_results(files, digests, formatted, linted, snapshot):
    errors = []
    known_paths = set(files)
    observed_paths = formatted | set(linted)
    paths = known_paths | set(snapshot) | observed_paths
    for name in sorted(paths):
        entry = snapshot.get(name)
        if entry is None:
            if name in observed_paths:
                errors.append(f"New Ruff finding without a baseline: {name}")
            continue
        if name not in known_paths:
            errors.append(f"Stale Ruff baseline path: {name}")
            continue
        if entry["sha256"] != digests.get(name):
            errors.append(f"Stale Ruff baseline SHA-256: {name}")
            continue
        if entry["format"] != (name in formatted):
            errors.append(f"Ruff format baseline differs from current output: {name}")
        actual = linted.get(name, [])
        expected = entry["lint"]
        if actual != expected:
            errors.append(f"Ruff lint baseline differs from current output: {name}")
    return errors


def main():
    if not SNAPSHOT.is_file():
        raise SystemExit("Ruff snapshot is missing.")
    files = tracked_python_files()
    snapshot = load_snapshot(SNAPSHOT)
    frozen_errors = validate_frozen_entries(snapshot, source_lock_hashes())
    if frozen_errors:
        raise SystemExit("\n".join(frozen_errors))
    trusted_path = os.environ.get(SNAPSHOT_ENV)
    if trusted_path:
        trusted = load_snapshot(Path(trusted_path))
        baseline_errors = compare_snapshots(trusted, snapshot)
        if baseline_errors:
            raise SystemExit("\n".join(baseline_errors))

    digests = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
    }
    format_command, lint_command = ruff_commands(files)
    format_rows = run_json(format_command)
    lint_rows = run_json(lint_command)
    formatted, linted = collect_results(format_rows, lint_rows)
    errors = verify_results(files, digests, formatted, linted, snapshot)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "Ruff found no new issues; verified "
        f"{len(formatted)} frozen format exception(s) and "
        f"{sum(map(len, linted.values()))} frozen lint diagnostic(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
