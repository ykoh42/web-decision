"""Verify the custom Kanana files against the reviewed model revision."""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_KANANA_MODEL_FILES = {
    "config.json": "64f4a103accb8946bb7ded7dc5d200f9d6041df3eaa891c04124bb41fa06a69b",
    "configuration_kanana_v_decide.py": "7dfa4a0480db133aec3a2da8411db2232aea58097deb1e626978b554c421ec93",
    "modeling_kanana_v_decide.py": "7e7abf64026f9c30e21fb663a93b5206b9b97795fea7dabc8769c967d5812be1",
    "decision_kanana_v_decide.py": "a4e33c66db53f50118005661dbc4354b0a259e01d243cc0f4afbe0e22106b3a2",
    "processing_kanana_v_decide.py": "a95089bae7012b471e349acb3755c98f74ebd6c324afe2d6217e6a97101beba9",
    "image_processing_kanana_v.py": "4794fd3dc937b85b3e2682680fb06765c06d9d7626d7f8250179cbd63bb2bbc9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_model_files(model_dir: Path) -> dict[str, str]:
    """Hash every pinned config and custom module before a loader can execute it."""
    verified: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}

    for name, expected in PINNED_KANANA_MODEL_FILES.items():
        path = model_dir / name
        try:
            if not path.is_file():
                mismatches[name] = {"expected": expected, "actual": "missing"}
                continue
            actual = sha256_file(path)
        except OSError as error:
            mismatches[name] = {
                "expected": expected,
                "actual": f"unreadable: {error.strerror}",
            }
            continue

        verified[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}

    if mismatches:
        raise SystemExit(
            f"Kanana model files differ from the pinned revision: {mismatches}"
        )
    return verified
