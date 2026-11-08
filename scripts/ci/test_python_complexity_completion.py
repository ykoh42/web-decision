"""Tests for the Python complexity completion gate."""

from pathlib import Path

import pytest

from scripts.ci.check_python_complexity_completion import (
    Finding,
    completion_violations,
    load_snapshot,
    require_complete,
    validate_snapshot,
)


def entry(
    path: str, function: str = "route", complexity: int = 16
) -> dict[str, object]:
    return {
        "path": path,
        "file_name": Path(path).name,
        "functions": [{"name": function, "complexity": complexity}],
    }


def test_excludes_only_fixed_historical_release_directories():
    findings = validate_snapshot(
        [
            entry(
                "releases/web-decision-0.1.0/legacy.py",
                "historical",
                87,
            ),
            entry(
                "releases/web-decision-0.2.0/legacy.py",
                "historical",
                87,
            ),
            entry(
                "releases/web-decision-0.3.0/legacy.py",
                "historical",
                87,
            ),
        ]
    )
    paths = {finding.path for finding in findings}
    assert completion_violations(findings, paths) == []
    require_complete(findings, paths)


def test_new_release_snapshot_and_vendor_paths_remain_managed():
    findings = validate_snapshot(
        [
            entry("releases/web-decision-0.4.0/future.py", "future"),
            entry("vendor/tvm/runtime.py", "runtime", 17),
        ]
    )
    paths = {finding.path for finding in findings}
    with pytest.raises(ValueError, match="Managed Python source remains"):
        require_complete(findings, paths)


def test_fails_for_managed_python_finding_over_15():
    findings = [Finding("scripts/route.py", "route", 16)]
    with pytest.raises(ValueError, match="Managed Python source remains"):
        require_complete(findings, {"scripts/route.py"})


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.py",
        "/absolute.py",
        "dir\\file.py",
        "C:/drive.py",
        "scripts/route.txt",
    ],
)
def test_rejects_invalid_baseline_paths(bad_path: str):
    with pytest.raises(ValueError, match="Invalid Python complexity baseline path"):
        validate_snapshot([entry(bad_path)])


def test_rejects_malformed_baseline_entries():
    with pytest.raises(ValueError, match="must be a list"):
        validate_snapshot({})
    with pytest.raises(ValueError, match="invalid functions"):
        validate_snapshot(
            [{"path": "scripts/route.py", "file_name": "route.py", "functions": None}]
        )
    with pytest.raises(ValueError, match="invalid finding"):
        validate_snapshot([entry("scripts/route.py", complexity=15)])
    malformed = entry("scripts/route.py")
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="malformed file entry"):
        validate_snapshot([malformed])


def test_rejects_findings_without_a_current_scanned_source():
    finding = Finding("scripts/missing.py", "old", 16)
    with pytest.raises(ValueError, match="missing or unscanned source"):
        completion_violations([finding], set())


def test_fails_closed_when_the_complexipy_baseline_is_missing(tmp_path: Path):
    with pytest.raises(ValueError, match="baseline is missing or unreadable"):
        load_snapshot(tmp_path / "missing.json")
