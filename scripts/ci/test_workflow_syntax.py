from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOW_FILES = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


@pytest.mark.parametrize("workflow", WORKFLOW_FILES)
def test_workflow_has_valid_yaml(workflow):
    document = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(document, dict)
    assert "on" in document and "jobs" in document


def test_unquoted_colon_followed_by_space_is_rejected():
    with pytest.raises(yaml.YAMLError):
        yaml.load(
            "run: uv pip install --only-binary=:all: --python x", Loader=yaml.BaseLoader
        )


def test_package_jobs_build_the_pinned_tvm_source_without_local_patches():
    workflow = yaml.load(
        (WORKFLOWS / "package-tests.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    for job_name in ("package-tests", "package"):
        steps = workflow["jobs"][job_name]["steps"]
        prepare_tvm = next(
            step for step in steps if step.get("name") == "Prepare pinned TVM runtime"
        )
        script = prepare_tvm["run"]
        assert "apply_tvm_webgpu" not in script
        assert "submodule update --init --recursive 3rdparty/tvm-ffi" in script
        assert "emsdk activate 4.0.23 --embedded" in script
        assert "make -C .toolchains/apache-tvm-src/web -f Makefile" in script
        assert '-f "$GITHUB_WORKSPACE/scripts/tvm-browser-runtime.mk"' in script
