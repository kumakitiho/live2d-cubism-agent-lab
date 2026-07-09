from __future__ import annotations

from pathlib import Path

from tools.artifact_validation import (
    load_yaml_mapping,
    validate_character_spec,
    validate_layer_map,
)


def test_character_spec_sample_is_valid() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    assert validate_character_spec(data) == []


def test_layer_map_sample_is_valid() -> None:
    data = load_yaml_mapping(Path("examples/layer_map.sample.yaml"))
    assert validate_layer_map(data) == []


def test_character_spec_requires_safety_constraints() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    data["constraints"]["human_visual_review_required"] = False
    messages = [issue.format() for issue in validate_character_spec(data)]
    assert "constraints.human_visual_review_required: must be true" in messages


def test_layer_map_detects_duplicate_names() -> None:
    data = load_yaml_mapping(Path("examples/layer_map.sample.yaml"))
    data["layers"].append(dict(data["layers"][0]))
    messages = [issue.format() for issue in validate_layer_map(data)]
    assert any("duplicate layer name" in message for message in messages)
