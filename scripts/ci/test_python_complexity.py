import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name("check_python_complexity.py")
PREPARE = Path(__file__).with_name("prepare_python_complexity_baseline.py")


def make_function(branches, *, ignore=False):
    directive = "  # noqa: complexipy" if ignore else ""
    lines = [f"def example(value):{directive}"]
    lines.extend(f"    if value == {index}: value += 1" for index in range(branches))
    return "\n".join([*lines, "    return value", ""])


def run(*args, cwd, env=None):
    return subprocess.run(
        args, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )


@pytest.fixture
def project(tmp_path):
    checker = tmp_path / "scripts/ci/check_python_complexity.py"
    checker.parent.mkdir(parents=True)
    shutil.copyfile(CHECKER, checker)
    (tmp_path / "existing.py").write_text(make_function(16))
    assert run("git", "init", "--quiet", cwd=tmp_path).returncode == 0
    snapshot = run(
        "complexipy",
        "--isolated",
        "--no-ignore",
        "--check-script",
        "--max-complexity-allowed",
        "15",
        "--snapshot-create",
        "--plain",
        "existing.py",
        cwd=tmp_path,
    )
    assert snapshot.returncode == 0, snapshot.stderr
    assert (
        run(
            "git",
            "add",
            "scripts",
            "existing.py",
            "complexipy-snapshot.json",
            cwd=tmp_path,
        ).returncode
        == 0
    )
    return tmp_path


def check(project, *, baseline=None):
    env = os.environ.copy()
    if baseline is not None:
        env["WEB_DECISION_COMPLEXITY_BASELINE"] = str(baseline)
    return run(
        sys.executable, "scripts/ci/check_python_complexity.py", cwd=project, env=env
    )


def test_existing_baseline_passes_without_changing_snapshot(project):
    snapshot = project / "complexipy-snapshot.json"
    before = snapshot.read_bytes()
    result = check(project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot.read_bytes() == before


def test_new_complex_function_is_rejected_even_if_ignored_inline(project):
    (project / "new.py").write_text(make_function(17, ignore=True))
    result = check(project)
    assert result.returncode != 0
    assert "new.py" in result.stdout + result.stderr


def test_worsening_existing_function_is_rejected(project):
    (project / "existing.py").write_text(make_function(17))
    result = check(project)
    assert result.returncode != 0
    assert "increased" in result.stdout + result.stderr


def test_hidden_tracked_file_is_checked(project):
    hidden = project / ".hidden/new.py"
    hidden.parent.mkdir()
    hidden.write_text(make_function(17))
    assert (
        run(
            "git", "add", "-f", str(hidden.relative_to(project)), cwd=project
        ).returncode
        == 0
    )
    result = check(project)
    assert result.returncode != 0
    assert ".hidden/new.py" in (result.stdout + result.stderr).replace(os.sep, "/")


def test_missing_baseline_fails_closed(project):
    (project / "complexipy-snapshot.json").unlink()
    result = check(project)
    assert result.returncode != 0
    assert "baseline is missing" in result.stderr


def test_baseline_cannot_be_loosened_from_trusted_base(project, tmp_path):
    snapshot = project / "complexipy-snapshot.json"
    trusted = tmp_path / "trusted.json"
    shutil.copyfile(snapshot, trusted)
    data = json.loads(snapshot.read_text())
    data[0]["functions"][0]["complexity"] += 1
    snapshot.write_text(json.dumps(data))
    result = check(project, baseline=trusted)
    assert result.returncode != 0
    assert "baseline was loosened" in result.stderr


def test_new_baseline_entry_is_rejected(project, tmp_path):
    snapshot = project / "complexipy-snapshot.json"
    trusted = tmp_path / "trusted.json"
    shutil.copyfile(snapshot, trusted)
    data = json.loads(snapshot.read_text())
    data[0]["functions"].append({"name": "new_function", "complexity": 16})
    snapshot.write_text(json.dumps(data))
    result = check(project, baseline=trusted)
    assert result.returncode != 0
    assert "baseline was loosened" in result.stderr


def test_removing_baseline_entry_does_not_hide_existing_violation(project, tmp_path):
    snapshot = project / "complexipy-snapshot.json"
    trusted = tmp_path / "trusted.json"
    shutil.copyfile(snapshot, trusted)
    snapshot.write_text("[]")
    result = check(project, baseline=trusted)
    assert result.returncode != 0
    assert "existing.py" in result.stdout + result.stderr


@pytest.mark.parametrize("bad_path", ["../outside.py", "/absolute.py", "dir\\file.py"])
def test_invalid_baseline_paths_fail_closed(project, tmp_path, bad_path):
    snapshot = project / "complexipy-snapshot.json"
    trusted = tmp_path / "trusted.json"
    shutil.copyfile(snapshot, trusted)
    data = json.loads(snapshot.read_text())
    data[0]["path"] = bad_path
    snapshot.write_text(json.dumps(data))
    result = check(project, baseline=trusted)
    assert result.returncode != 0
    assert "Invalid complexity baseline path" in result.stderr


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


@pytest.fixture
def baseline_project(tmp_path):
    script = tmp_path / "scripts/ci/prepare_python_complexity_baseline.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(PREPARE, script)
    assert run("git", "init", "--quiet", cwd=tmp_path).returncode == 0
    assert run("git", "add", "scripts", cwd=tmp_path).returncode == 0
    return tmp_path, commit(tmp_path, "base")


def prepare(project, base, output):
    env = os.environ.copy()
    env["BASE_SHA"] = base
    return run(
        sys.executable,
        "scripts/ci/prepare_python_complexity_baseline.py",
        str(output),
        cwd=project,
        env=env,
    )


def test_first_baseline_addition_bootstraps_for_push_or_pull_request(baseline_project):
    project, base = baseline_project
    (project / "complexipy-snapshot.json").write_text("[]")
    assert run("git", "add", "complexipy-snapshot.json", cwd=project).returncode == 0
    commit(project, "add baseline")
    output = project / "trusted.json"
    result = prepare(project, base, output)
    assert result.returncode == 0, result.stderr
    assert "initial reviewed baseline" in result.stdout
    assert not output.exists()


def test_existing_baseline_is_read_from_base_commit(baseline_project):
    project, _ = baseline_project
    snapshot = project / "complexipy-snapshot.json"
    snapshot.write_text("trusted")
    assert run("git", "add", "complexipy-snapshot.json", cwd=project).returncode == 0
    base = commit(project, "add baseline")
    snapshot.write_text("candidate")
    output = project / "trusted.json"
    result = prepare(project, base, output)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "trusted"


def test_missing_baseline_without_addition_fails(baseline_project):
    project, base = baseline_project
    result = prepare(project, base, project / "trusted.json")
    assert result.returncode != 0
    assert "baseline is unavailable" in result.stderr


def test_invalid_or_unavailable_base_commit_fails(baseline_project):
    project, _ = baseline_project
    for base in ("bad", "a" * 40):
        result = prepare(project, base, project / "trusted.json")
        assert result.returncode != 0
