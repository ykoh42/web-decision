#!/usr/bin/env python3
"""Load the trusted Ruff snapshot from the commit under review."""

import os
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from .git_environment import git_environment
else:
    from git_environment import git_environment

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "ruff-snapshot.json"


def git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=git_environment(ROOT),
        capture_output=True,
        check=False,
    )


def main():
    base = os.environ.get("BASE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise SystemExit("Missing or invalid base commit SHA.")
    if git("cat-file", "-e", f"{base}^{{commit}}").returncode:
        raise SystemExit("The base commit is unavailable.")
    previous = git("show", f"{base}:{SNAPSHOT}")
    if previous.returncode == 0:
        Path(sys.argv[1]).write_bytes(previous.stdout)
        return
    changes = git("diff", "--name-status", "--no-renames", base, "HEAD", "--", SNAPSHOT)
    if changes.returncode or changes.stdout.strip() != f"A\t{SNAPSHOT}".encode():
        raise SystemExit("Previous Ruff snapshot is unavailable.")
    print("Installing the initial reviewed Ruff snapshot.")


if __name__ == "__main__":
    main()
