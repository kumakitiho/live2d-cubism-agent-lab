from __future__ import annotations

from pathlib import Path

import yaml


def test_github_actions_runs_pytest_on_push_and_pull_request() -> None:
    workflow_path = Path(".github/workflows/pytest.yml")
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {"push", "pull_request"}
    steps = workflow["jobs"]["test"]["steps"]
    assert any(step.get("run") == "python -m pytest -q" for step in steps)
