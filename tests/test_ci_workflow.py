from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_github_actions_runs_pytest_on_push_and_pull_request() -> None:
    workflow_path = Path(".github/workflows/pytest.yml")
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert isinstance(workflow, dict)
    assert set(workflow["on"]) == {"push", "pull_request"}
    steps = workflow["jobs"]["test"]["steps"]
    assert any(step.get("run") == "python -m pytest -q" for step in steps)


def test_wall_clock_benchmarks_are_opt_in() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert "not benchmark" in pytest_options["addopts"]
    assert any(marker.startswith("benchmark:") for marker in pytest_options["markers"])
