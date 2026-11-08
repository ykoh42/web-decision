#!/usr/bin/env python3
"""Load a trusted baseline, allowing its first addition to the repository."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "complexipy-snapshot.json"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


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
        raise SystemExit("Previous complexity baseline is unavailable.")
    print("Installing the initial reviewed baseline.")


if __name__ == "__main__":
    main()
