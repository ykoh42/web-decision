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
