"""Build a Git environment that supports Windows and WSL worktree markers."""

import os
import re
from pathlib import Path


def _wsl_git_directory(git_directory: str) -> str | None:
    match = re.fullmatch(r"/mnt/([a-z])/(.+)", git_directory, flags=re.IGNORECASE)
    return f"{match.group(1).upper()}:/{match.group(2)}" if match else None


def _windows_git_directory(git_directory: str) -> str | None:
    match = re.fullmatch(r"([a-z]):[\\/](.+)", git_directory, flags=re.IGNORECASE)
    if not match:
        return None
    relative = match.group(2).replace("\\", "/")
    linux_path = Path("/mnt") / match.group(1).lower() / relative
    return str(linux_path) if linux_path.exists() else None


def _host_git_directory(git_directory: str) -> str | None:
    if os.name == "nt":
        return _wsl_git_directory(git_directory)
    return _windows_git_directory(git_directory)


def git_environment(root: str | Path) -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    try:
        marker = (Path(root) / ".git").read_text(encoding="utf-8")
    except OSError:
        return environment
    match = re.fullmatch(r"gitdir:\s*(.+)\s*", marker, flags=re.MULTILINE)
    host_git_directory = _host_git_directory(match.group(1)) if match else None
    if host_git_directory is None:
        return environment
    environment["GIT_DIR"] = host_git_directory
    environment["GIT_WORK_TREE"] = os.fspath(root)
    return environment
