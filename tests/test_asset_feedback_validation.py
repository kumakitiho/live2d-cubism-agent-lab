from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_feedback_validator import load_layer_ids, main, validate_asset_feedback


def _sample() -> dict[str, object]:
    return load_yaml_mapping(Path("examples/asset_feedback.sample.yaml"))


def test_asset_feedback_sample_matches_layer_map() -> None:
    layer_ids = load_layer_ids(Path("examples/layer_map.sample.yaml"))
    assert validate_asset_feedback(_sample(), known_layer_ids=layer_ids) == []


def test_asset_feedback_detects_unknown_layer_id() -> None:
    data = deepcopy(_sample())
    feedback = data["feedback"]
    assert isinstance(feedback, list)
    feedback[0]["target_layer_id"] = "missing_layer"

    issues = validate_asset_feedback(data, known_layer_ids={"face_hidden_fill"})

    assert any("does not exist in layer map" in issue.message for issue in issues)


def test_asset_feedback_detects_duplicate_feedback_id() -> None:
    data = deepcopy(_sample())
    feedback = data["feedback"]
    assert isinstance(feedback, list)
    feedback.append(deepcopy(feedback[0]))

    issues = validate_asset_feedback(data)

    assert any("duplicate feedback id" in issue.message for issue in issues)


def test_asset_feedback_rejects_invalid_severity_and_action() -> None:
    data = deepcopy(_sample())
    feedback = data["feedback"]
    assert isinstance(feedback, list)
    feedback[0]["severity"] = "urgent"
    feedback[0]["requested_action"]["action"] = "click_and_fix"

    issues = validate_asset_feedback(data)

    messages = [issue.format() for issue in issues]
    assert any("severity" in message for message in messages)
    assert any("requested_action.action" in message for message in messages)


def test_asset_feedback_must_match_validated_layer_map_project_and_path() -> None:
    data = deepcopy(_sample())

    issues = validate_asset_feedback(
        data,
        known_layer_ids={"face_hidden_fill"},
        layer_map_project="different-project",
        layer_map_path=Path("generated/other-layer-map.yaml"),
        base_dir=Path.cwd(),
    )

    messages = [issue.format() for issue in issues]
    assert any("must match layer map project" in message for message in messages)
    assert any("must reference the layer map used" in message for message in messages)


def test_asset_feedback_cli_rejects_layer_map_without_project(tmp_path: Path) -> None:
    layer_map = load_yaml_mapping(Path("examples/layer_map.sample.yaml"))
    layer_map.pop("project")
    layer_map_path = tmp_path / "layer_map.yaml"
    layer_map_path.write_text(
        yaml.safe_dump(layer_map, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "examples/asset_feedback.sample.yaml",
            "--layer-map",
            str(layer_map_path),
        ]
    )

    assert exit_code == 2
