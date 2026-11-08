"""Exercise the exact pin and hash parser without installing model packages."""

from pathlib import Path

import pytest
from verify_preview_build_requirements import (
    read_direct_requirements,
    read_locked_requirements,
)

ROOT = Path(__file__).resolve().parents[1]
HASH_A = "a" * 64
HASH_B = "b" * 64
CONTINUATION = chr(92)


def write_lock(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "requirements.lock"
    path.write_text(content, encoding="utf-8")
    return path


def locked_entry(name: str, version: str, *hashes: str) -> str:
    lines = [f"{name}=={version} {CONTINUATION}"]
    lines.extend(f"    --hash=sha256:{digest} {CONTINUATION}" for digest in hashes[:-1])
    lines.append(f"    --hash=sha256:{hashes[-1]}")
    return "\n".join(lines) + "\n"


def test_real_preview_lock_satisfies_every_direct_pin():
    direct = read_direct_requirements(ROOT / "requirements-preview-build.txt")
    locked = read_locked_requirements(ROOT / "requirements-preview-build.lock")
    assert all(locked.get(name) == version for name, version in direct.items())


def test_reads_normalized_pins_and_multiple_hashes(tmp_path: Path):
    lock = write_lock(tmp_path, locked_entry("Example_Pkg", "1.2.3", HASH_A, HASH_B))
    assert read_locked_requirements(lock) == {"example-pkg": "1.2.3"}


def test_rejects_a_package_without_hashes_before_the_next_pin(tmp_path: Path):
    lock = write_lock(
        tmp_path,
        f"first==1.0.0\n{locked_entry('second', '2.0.0', HASH_A)}",
    )
    with pytest.raises(ValueError, match="first has no SHA-256 artifact hash"):
        read_locked_requirements(lock)


def test_rejects_invalid_hash_entries(tmp_path: Path):
    lock = write_lock(
        tmp_path,
        f"example==1.0.0 {CONTINUATION}\n    --hash=sha256:short\n",
    )
    with pytest.raises(ValueError, match="invalid SHA-256 hash entry"):
        read_locked_requirements(lock)


def test_rejects_duplicate_normalized_package_names(tmp_path: Path):
    lock = write_lock(
        tmp_path,
        locked_entry("Example_Pkg", "1.2.3", HASH_A)
        + locked_entry("example-pkg", "1.2.3", HASH_B),
    )
    with pytest.raises(ValueError, match="duplicate package"):
        read_locked_requirements(lock)


def test_rejects_unpinned_and_empty_locks(tmp_path: Path):
    malformed = write_lock(tmp_path, "example>=1.0\n")
    with pytest.raises(ValueError, match="expected an exact package pin"):
        read_locked_requirements(malformed)
    empty = write_lock(tmp_path, "# no package pins\n\n")
    with pytest.raises(ValueError, match="no packages were locked"):
        read_locked_requirements(empty)
