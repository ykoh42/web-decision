#!/usr/bin/env python3
"""Check every tracked Python file without modifying the reviewed baseline."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

if __package__:
    from .git_environment import git_environment
else:
    from git_environment import git_environment

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "complexipy-snapshot.json"


def python_files():
    result = subprocess.check_output(
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
    )
    return [Path(name.decode()) for name in result.split(b"\0") if name]


def baseline_scores(path):
    scores = {}
    paths = set()
    for entry in json.loads(path.read_text(encoding="utf-8")):
        name = entry["path"]
        normalized = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or normalized.is_absolute()
            or normalized.as_posix() != name
            or ".." in normalized.parts
            or "\\" in name
            or not name.endswith(".py")
            or name.casefold() in paths
        ):
            raise ValueError(f"Invalid complexity baseline path: {name}")
        paths.add(name.casefold())
        for function in entry["functions"]:
            key = (name, function["name"])
            score = function["complexity"]
            if (
                not isinstance(key[1], str)
                or not key[1]
                or key in scores
                or type(score) is not int
                or score < 0
            ):
                raise ValueError(f"Invalid complexity baseline entry: {key}")
            scores[key] = score
    return scores


def check_baseline_not_loosened(base_path):
    previous = baseline_scores(base_path)
    for key, score in baseline_scores(BASELINE).items():
        if key not in previous or score > previous[key]:
            raise SystemExit(f"Python complexity baseline was loosened: {key}")


def main():
    files = python_files()
    if not files or not BASELINE.is_file():
        raise SystemExit("Python complexity input or reviewed baseline is missing.")
    base_path = os.environ.get("WEB_DECISION_COMPLEXITY_BASELINE")
    if base_path:
        check_baseline_not_loosened(Path(base_path))

    with tempfile.TemporaryDirectory(prefix="web-decision-complexity-") as temporary:
        root = Path(temporary)
        snapshot = json.loads(BASELINE.read_text(encoding="utf-8"))
        for entry in snapshot:
            entry["path"] = os.fspath(Path(entry["path"]))
        (root / BASELINE.name).write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
        )
        for name in files:
            source = ROOT / name
            if (
                name.is_absolute()
                or ".." in name.parts
                or source.is_symlink()
                or not source.is_file()
            ):
                raise SystemExit(f"Invalid Python input: {name}")
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        command = [
            "complexipy",
            "--isolated",
            "--no-ignore",
            "--check-script",
            "--max-complexity-allowed",
            "15",
            "--failed",
            "--plain",
            "--cache-dir",
            str(root / ".cache"),
            "--",
            *(name.as_posix() for name in files),
        ]
        return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
