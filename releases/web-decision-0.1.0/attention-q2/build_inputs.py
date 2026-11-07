from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


HASH_CHUNK_BYTES = 1024 * 1024


def verify_pinned_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> str:
    path = Path(path)
    if expected_bytes < 0:
        raise ValueError(f"Invalid expected byte size for {label}.")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if info.st_size != expected_bytes:
        raise ValueError(f"{label} has an unexpected byte size: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file: {path}") from error

    try:
        opened_info = os.fstat(descriptor)
        if not stat.S_ISREG(opened_info.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
        if opened_info.st_size != expected_bytes:
            raise ValueError(f"{label} has an unexpected byte size: {path}")

        digest = hashlib.sha256()
        total_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = -1
            while total_bytes <= expected_bytes:
                chunk = source.read(min(HASH_CHUNK_BYTES, expected_bytes + 1 - total_bytes))
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > expected_bytes:
                    raise ValueError(f"{label} has an unexpected byte size: {path}")
                digest.update(chunk)
        if total_bytes != expected_bytes:
            raise ValueError(f"{label} has an unexpected byte size: {path}")

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{label} does not match its SHA-256 pin: {path}")
        return actual_sha256
    finally:
        if descriptor >= 0:
            os.close(descriptor)
