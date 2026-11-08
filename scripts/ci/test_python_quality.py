import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest
from check_python_quality import (
    ROOT,
    SAFE_OUTPUT_SHA256,
    compare_snapshots,
    ruff_commands,
    validate_frozen_entries,
    validate_snapshot,
    verify_results,
)

PREPARE = Path(__file__).with_name("prepare_python_quality_baseline.py")


def fingerprint(code="F401", row=4, column=1, message="unused import"):
    return {"code": code, "row": row, "column": column, "message": message}


def snapshot_entry(path, digest, *, formatted=False, lint=None):
    return {
        "path": path,
        "sha256": digest,
        "format": formatted,
        "lint": [] if lint is None else lint,
    }


def parse_snapshot(*entries):
    return validate_snapshot({"schemaVersion": 1, "files": list(entries)})


def test_existing_findings_are_allowed_only_for_the_exact_baselined_bytes():
    path = "releases/web-decision-0.1.0/example.py"
    digest = "a" * 64
    allowed = parse_snapshot(
        snapshot_entry(path, digest, formatted=True, lint=[fingerprint()])
    )

    assert (
        verify_results({path}, {path: digest}, {path}, {path: [fingerprint()]}, allowed)
        == []
    )
    errors = verify_results(
        {path}, {path: "b" * 64}, {path}, {path: [fingerprint()]}, allowed
    )
    assert any(
        "exact file hash" in error or "Stale Ruff baseline SHA" in error
        for error in errors
    )


def test_added_lint_diagnostic_is_not_hidden_by_an_existing_exception():
    path = "releases/web-decision-0.1.0/example.py"
    digest = "a" * 64
    allowed = parse_snapshot(snapshot_entry(path, digest, lint=[fingerprint()]))
    diagnostics = [fingerprint(), fingerprint("E401", 8, 1, "multiple imports")]

    errors = verify_results({path}, {path: digest}, set(), {path: diagnostics}, allowed)

    assert any("lint baseline differs" in error for error in errors)


def test_finding_without_an_exact_baseline_entry_fails():
    path = "scripts/new_check.py"
    errors = verify_results(
        {path}, {path: "a" * 64}, set(), {path: [fingerprint()]}, {}
    )
    assert errors == [f"New Ruff finding without a baseline: {path}"]


def test_baseline_findings_cannot_disappear_without_changing_the_file():
    path = "releases/web-decision-0.1.0/example.py"
    digest = "a" * 64
    allowed = parse_snapshot(
        snapshot_entry(path, digest, formatted=True, lint=[fingerprint()])
    )

    errors = verify_results({path}, {path: digest}, set(), {}, allowed)

    assert any("format baseline differs" in error for error in errors)
    assert any("lint baseline differs" in error for error in errors)


def test_candidate_snapshot_cannot_add_an_exception_or_replace_a_hash():
    path = "releases/web-decision-0.1.0/example.py"
    original = snapshot_entry(path, "a" * 64, lint=[fingerprint()])
    trusted = parse_snapshot(original)
    added = parse_snapshot(
        original,
        snapshot_entry("releases/web-decision-0.1.0/new.py", "c" * 64, formatted=True),
    )
    replaced = parse_snapshot(snapshot_entry(path, "b" * 64, lint=[fingerprint()]))

    assert any(
        "added an exception" in error for error in compare_snapshots(trusted, added)
    )
    assert any(
        "changed a pinned SHA" in error
        for error in compare_snapshots(trusted, replaced)
    )


def test_candidate_snapshot_cannot_add_a_lint_fingerprint():
    path = "releases/web-decision-0.1.0/example.py"
    trusted = parse_snapshot(snapshot_entry(path, "a" * 64, lint=[fingerprint()]))
    candidate = parse_snapshot(
        snapshot_entry(
            path,
            "a" * 64,
            lint=[fingerprint(), fingerprint("E401", 8, 1, "multiple imports")],
        )
    )

    assert any(
        "added lint exceptions" in error
        for error in compare_snapshots(trusted, candidate)
    )


def test_lint_command_ignores_inline_noqa_directives():
    _, lint_command = ruff_commands(["scripts/example.py"])
    assert "--ignore-noqa" in lint_command


def test_ruff_commands_are_isolated_from_exclusion_and_rule_config():
    format_command, lint_command = ruff_commands(["scripts/example.py"])

    for command in (format_command, lint_command):
        assert "--isolated" in command
        assert "--no-respect-gitignore" in command
        assert "--no-force-exclude" in command
        assert command[command.index("--target-version") + 1] == "py312"
        assert command[command.index("--line-length") + 1] == "88"
    assert lint_command[lint_command.index("--select") + 1] == "E4,E7,E9,F,I"


def test_ruff_rules_are_explicit_for_the_locked_python_version():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    ruff = configuration["tool"]["ruff"]

    assert ruff["target-version"] == "py312"
    assert ruff["lint"]["select"] == ["E4", "E7", "E9", "F", "I"]


def test_baseline_entries_must_match_a_release_source_lock_or_protected_sha():
    path = "releases/web-decision-0.1.0/example.py"
    digest = "a" * 64
    locked = defaultdict(set, {path: {digest}})
    valid = parse_snapshot(snapshot_entry(path, digest, formatted=True))

    assert validate_frozen_entries(valid, locked) == []
    assert validate_frozen_entries(
        parse_snapshot(snapshot_entry(path, "b" * 64, formatted=True)), locked
    )
    protected = parse_snapshot(
        snapshot_entry("scripts/safe_output.py", SAFE_OUTPUT_SHA256, formatted=True)
    )
    assert validate_frozen_entries(protected, {}) == []


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py", "dir\\file.py"])
def test_snapshot_rejects_non_relative_paths(path):
    with pytest.raises(ValueError, match="Invalid Python path"):
        parse_snapshot(snapshot_entry(path, "a" * 64, formatted=True))


def test_snapshot_rejects_duplicate_diagnostics():
    item = fingerprint()
    with pytest.raises(ValueError, match="Duplicate Ruff snapshot diagnostic"):
        parse_snapshot(
            snapshot_entry(
                "releases/web-decision-0.1.0/example.py",
                "a" * 64,
                lint=[item, item],
            )
        )


def test_tvm_build_script_remains_at_the_source_lock_hash_with_side_effect_imports():
    path = ROOT / "scripts/build_tvm_q4.py"
    contents = path.read_bytes()

    assert hashlib.sha256(contents).hexdigest() == (
        "706ced9e6a620cd4c653cb130e809b91be7c94b6ead6041ae85626f13fcc6deb"
    )
    source = contents.decode("utf-8")
    assert "import tvm_selective_params,tvm_narrow_int64,tvm_q4_operators" in source
    assert source.index("import tvm_selective_params") < source.index("ROOT=")


def run(*arguments, cwd, env=None):
    return subprocess.run(
        arguments, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )


def commit(project, message):
    result = run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
        cwd=project,
    )
    assert result.returncode == 0, result.stderr
    return run("git", "rev-parse", "HEAD", cwd=project).stdout.strip()


def baseline_project(project, *, with_snapshot):
    script = project / "scripts/ci/prepare_python_quality_baseline.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(PREPARE, script)
    assert run("git", "init", "--quiet", cwd=project).returncode == 0
    assert run("git", "add", "scripts", cwd=project).returncode == 0
    if with_snapshot:
        (project / "ruff-snapshot.json").write_bytes(b"trusted snapshot\n")
        assert run("git", "add", "ruff-snapshot.json", cwd=project).returncode == 0
    return commit(project, "base")


def prepare(project, base, output):
    env = os.environ.copy()
    env["BASE_SHA"] = base
    script = project / "scripts/ci/prepare_python_quality_baseline.py"
    return run(sys.executable, str(script), str(output), cwd=project, env=env)


def test_prepare_uses_the_baseline_from_the_trusted_base_commit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    base = baseline_project(project, with_snapshot=True)
    (project / "ruff-snapshot.json").write_bytes(b"candidate snapshot\n")

    output = project / "trusted.json"
    result = prepare(project, base, output)

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"trusted snapshot\n"


def test_prepare_bootstraps_only_when_the_snapshot_is_added(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    base = baseline_project(project, with_snapshot=False)
    (project / "ruff-snapshot.json").write_bytes(b"new snapshot\n")
    assert run("git", "add", "ruff-snapshot.json", cwd=project).returncode == 0
    commit(project, "add snapshot")

    result = prepare(project, base, project / "trusted.json")

    assert result.returncode == 0, result.stderr
    assert "initial reviewed Ruff snapshot" in result.stdout
    assert not (project / "trusted.json").exists()


def test_prepare_fails_if_the_previous_snapshot_is_unavailable(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    base = baseline_project(project, with_snapshot=False)

    result = prepare(project, base, project / "trusted.json")

    assert result.returncode != 0
    assert "Previous Ruff snapshot is unavailable" in result.stderr
