from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_feedback_validator import load_layer_map_context, validate_asset_feedback
from tools.cubism_evaluation import (
    evaluation_to_asset_feedback,
    main,
    validate_cubism_evaluation,
)


def _sample() -> dict[str, Any]:
    return load_yaml_mapping(Path("examples/cubism_evaluation.sample.yaml"))


def _context() -> tuple[set[str], str]:
    return load_layer_map_context(Path("examples/layer_map.sample.yaml"))


def test_dev_evaluation_allows_warn_and_reports_warning() -> None:
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        _sample(),
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert report.valid
    assert report.result == "warn"
    assert any("required check is warn" in warning.message for warning in report.warnings)


def test_strict_evaluation_rejects_warn() -> None:
    data = deepcopy(_sample())
    data["validation_mode"] = "strict"
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert report.feedback_convertible
    assert any("required check is warn" in issue.message for issue in report.issues)


def test_dev_evaluation_still_rejects_fail() -> None:
    data = deepcopy(_sample())
    data["checks"][1]["status"] = "fail"
    data["summary"]["result"] = "fail"
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert report.feedback_convertible
    assert any("failed check blocks" in issue.message for issue in report.issues)


def test_pass_evaluation_has_nothing_to_convert() -> None:
    data = deepcopy(_sample())
    mouth_check = data["checks"][1]
    mouth_check["status"] = "pass"
    mouth_check["issue_type"] = None
    mouth_check["severity"] = None
    mouth_check["requested_action"] = None
    data["summary"]["result"] = "pass"
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert report.valid
    assert not report.feedback_convertible


def test_incomplete_evaluation_has_nothing_to_convert() -> None:
    data = deepcopy(_sample())
    data["validation_mode"] = "strict"
    for check in data["checks"]:
        check["status"] = "not_run"
        check["issue_type"] = None
        check["severity"] = None
        check["requested_action"] = None
    data["summary"]["result"] = "incomplete"
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert not report.feedback_convertible


def test_evaluation_requires_all_basic_categories() -> None:
    data = deepcopy(_sample())
    data["checks"] = [check for check in data["checks"] if check["category"] != "texture"]
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert any("missing required category: texture" in issue.message for issue in report.issues)


def test_strict_evaluation_cannot_bypass_basic_checks_as_optional() -> None:
    data = deepcopy(_sample())
    data["validation_mode"] = "strict"
    for check in data["checks"]:
        check["required"] = False
        check["status"] = "not_run"
        check["issue_type"] = None
        check["severity"] = None
        check["requested_action"] = None
    data["summary"]["result"] = "pass"
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert not report.feedback_convertible
    assert any("category must contain a required check" in issue.message for issue in report.issues)


def test_evaluation_rejects_unknown_layer_id() -> None:
    data = deepcopy(_sample())
    data["checks"][0]["target_layer_ids"] = ["unknown-eye-layer"]
    layer_ids, project = _context()

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=project,
    )

    assert not report.valid
    assert any("does not exist in layer map" in issue.message for issue in report.issues)


def test_evaluation_warn_converts_to_valid_asset_feedback() -> None:
    data = _sample()
    layer_ids, project = _context()

    feedback = evaluation_to_asset_feedback(data)
    issues = validate_asset_feedback(
        feedback,
        known_layer_ids=layer_ids,
        layer_map_project=project,
        layer_map_path=Path("examples/layer_map.sample.yaml"),
        base_dir=Path.cwd(),
    )

    assert issues == []
    assert [entry["target_layer_id"] for entry in feedback["feedback"]] == ["mouth_inner"]


def test_evaluation_cli_validates_and_writes_feedback(tmp_path: Path) -> None:
    output = tmp_path / "asset_feedback.yaml"

    assert (
        main(
            [
                "validate",
                "examples/cubism_evaluation.sample.yaml",
                "--layer-map",
                "examples/layer_map.sample.yaml",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "to-feedback",
                "examples/cubism_evaluation.sample.yaml",
                "--layer-map",
                "examples/layer_map.sample.yaml",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert load_yaml_mapping(output)["feedback"][0]["target_layer_id"] == "mouth_inner"


def test_strict_warn_can_still_convert_to_feedback(tmp_path: Path) -> None:
    data = deepcopy(_sample())
    data["validation_mode"] = "strict"
    evaluation = tmp_path / "strict-evaluation.yaml"
    output = tmp_path / "strict-feedback.yaml"
    evaluation.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "to-feedback",
            str(evaluation),
            "--layer-map",
            "examples/layer_map.sample.yaml",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert load_yaml_mapping(output)["feedback"][0]["status"] == "open"
