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


def test_character_spec_rejects_open_questions() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    data["spec_provenance"]["open_questions"] = ["target runtimeを確認する"]
    messages = [issue.format() for issue in validate_character_spec(data)]
    assert any("must be empty before asset handoff" in message for message in messages)


def test_character_spec_requires_confirmed_rights() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    data["source_image"]["rights_status"] = "needs_confirmation"
    messages = [issue.format() for issue in validate_character_spec(data)]
    assert any("must be confirmed before asset handoff" in message for message in messages)


def test_character_spec_requires_human_provenance_for_intent_fields() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    confirmed = data["spec_provenance"]["user_confirmed_fields"]
    confirmed.remove("motion_level")
    data["spec_provenance"]["image_inferred_fields"].append("motion_level")

    messages = [issue.format() for issue in validate_character_spec(data)]

    assert any("human-only field cannot be image-inferred" in message for message in messages)
    assert any(
        "must include human-confirmed field: motion_level" in message for message in messages
    )


def test_character_spec_rejects_overlapping_provenance() -> None:
    data = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    data["spec_provenance"]["image_inferred_fields"].append("appearance.observed")
    data["spec_provenance"]["user_confirmed_fields"].append("appearance.observed")

    messages = [issue.format() for issue in validate_character_spec(data)]

    assert any("must not contain duplicates" in message for message in messages)
    assert any("cannot be both image-inferred" in message for message in messages)


def test_layer_map_detects_duplicate_names() -> None:
    data = load_yaml_mapping(Path("examples/layer_map.sample.yaml"))
    data["layers"].append(dict(data["layers"][0]))
    messages = [issue.format() for issue in validate_layer_map(data)]
    assert any("duplicate layer name" in message for message in messages)
