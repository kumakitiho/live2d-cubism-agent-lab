from __future__ import annotations

from pathlib import Path

import pytest

from tools.material_planner import build_material_plan, main


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    path = tmp_path / "character.png"
    path.write_bytes(b"not-decoded-by-the-planner")
    return path


def _layer_ids(plan: dict[str, object]) -> set[str]:
    parts = plan["parts"]
    assert isinstance(parts, list)
    return {str(part["layer_id"]) for part in parts}


def test_model_scope_changes_required_parts(source_image: Path) -> None:
    bust = build_material_plan(source_image, model_scope="bust_up", motion_level="minimal")
    half = build_material_plan(source_image, model_scope="half_body", motion_level="minimal")
    full = build_material_plan(source_image, model_scope="full_body", motion_level="minimal")

    assert "arm_L" not in _layer_ids(bust)
    assert "arm_L" in _layer_ids(half)
    assert "leg_L" not in _layer_ids(half)
    assert "leg_L" in _layer_ids(full)
    assert len(_layer_ids(bust)) < len(_layer_ids(half)) < len(_layer_ids(full))


def test_motion_level_replaces_coarse_parts_and_adds_detail(source_image: Path) -> None:
    minimal = build_material_plan(source_image, model_scope="bust_up", motion_level="minimal")
    standard = build_material_plan(source_image, model_scope="bust_up", motion_level="standard")
    expressive = build_material_plan(source_image, model_scope="bust_up", motion_level="expressive")

    assert "eye_L" in _layer_ids(minimal)
    assert "eye_white_L" not in _layer_ids(minimal)
    assert "eye_L" not in _layer_ids(standard)
    assert "eye_white_L" in _layer_ids(standard)
    assert "eye_closed_line_L" not in _layer_ids(standard)
    assert "eye_closed_line_L" in _layer_ids(expressive)
    assert len(_layer_ids(minimal)) < len(_layer_ids(standard)) < len(_layer_ids(expressive))


def test_inferred_parts_always_require_review(source_image: Path) -> None:
    plan = build_material_plan(source_image, model_scope="full_body", motion_level="expressive")
    parts = plan["parts"]
    assert isinstance(parts, list)
    inferred = [part for part in parts if part["inferred"] is True]

    assert inferred
    assert all(part["review_required"] is True for part in inferred)
    assert all("Infer hidden pixels" in part["prompt"]["instruction"] for part in inferred)


def test_plan_uses_three_masks_and_draw_order(source_image: Path) -> None:
    plan = build_material_plan(source_image, model_scope="bust_up", motion_level="minimal")
    parts = plan["parts"]
    assert isinstance(parts, list)

    assert all(part["target_mask"].endswith(".target.png") for part in parts)
    assert all(part["protect_mask"].endswith(".protect.png") for part in parts)
    assert all(part["edge_extension_mask"].endswith(".edge-extension.png") for part in parts)
    assert all(part["inpaint_mask"].endswith(".inpaint.png") for part in parts)
    assert all(isinstance(part["draw_order"], int) for part in parts)
    assert all(isinstance(part["overlap_margin_px"], int) for part in parts)
    assert all(
        part["generation_method"] != "extract"
        for part in parts
        if part["overlap_margin_px"] > 0
    )


def test_draw_order_places_hidden_and_back_hair_behind_visible_face(source_image: Path) -> None:
    plan = build_material_plan(source_image, model_scope="bust_up", motion_level="standard")
    parts = plan["parts"]
    assert isinstance(parts, list)
    order = {part["layer_id"]: part["draw_order"] for part in parts}

    assert order["face_hidden_fill"] < order["face_base"]
    assert order["hair_back_hidden_fill"] < order["hair_back"]
    assert order["hair_back"] < order["face_base"] < order["eye_white_L"]
    assert order["eye_white_L"] < order["eye_iris_L"] < order["eye_highlight_L"]


def test_default_cli_is_dry_run(source_image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = source_image.parent / "plan.yaml"
    result = main(
        [
            str(source_image),
            "--model-scope",
            "bust_up",
            "--motion-level",
            "minimal",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert "DRY-RUN" in capsys.readouterr().out
    assert not output.exists()


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_material_plan(
            tmp_path / "missing.png",
            model_scope="bust_up",
            motion_level="minimal",
        )
