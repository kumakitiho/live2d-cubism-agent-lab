from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw

from tools.automatic_mask_deriver import main as deriver_main
from tools.mask_derivation.pipeline import DerivationConfig, derive_masks
from tools.mask_derivation_ranker import main as ranker_main


def _mask(path: Path, box: tuple[int, int, int, int], *, value: int = 255) -> None:
    image = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(image).rectangle(box, fill=value)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    Image.new("RGBA", (20, 20), (100, 80, 60, 255)).save(tmp_path / "source.png")
    _mask(tmp_path / "masks/face.png", (5, 7, 14, 18), value=180)
    _mask(tmp_path / "masks/hair.png", (4, 2, 15, 8), value=210)
    _mask(tmp_path / "masks/background.png", (0, 0, 0, 19))
    queue: dict[str, Any] = {
        "schema_version": 3,
        "project": "mask-derivation-test",
        "source_image": {"path": "source.png"},
        "canvas": {"width": 20, "height": 20},
        "assets": [
            {
                "layer_id": "face_base",
                "role": "face",
                "side": "C",
                "source_file": "parts/face.png",
                "target_mask": "masks/face.png",
                "protect_mask": "canonical/face.protect.png",
                "edge_extension_mask": "canonical/face.edge.png",
                "inpaint_mask": "canonical/face.inpaint.png",
                "draw_order": 10,
                "segmentation_run_id": "segment-001",
                "segmentation_confidence": 0.9,
                "expected_region": {
                    "x_min": 0.2,
                    "y_min": 0.1,
                    "x_max": 0.8,
                    "y_max": 0.95,
                },
            },
            {
                "layer_id": "front_hair",
                "role": "front_hair",
                "side": "C",
                "source_file": "parts/hair.png",
                "target_mask": "masks/hair.png",
                "protect_mask": "canonical/hair.protect.png",
                "edge_extension_mask": "canonical/hair.edge.png",
                "inpaint_mask": "canonical/hair.inpaint.png",
                "draw_order": 20,
                "segmentation_run_id": "segment-001",
                "segmentation_confidence": 0.8,
            },
            {
                "layer_id": "background",
                "role": "background",
                "side": "C",
                "source_file": "parts/background.png",
                "target_mask": "masks/background.png",
                "protect_mask": "canonical/background.protect.png",
                "edge_extension_mask": "canonical/background.edge.png",
                "inpaint_mask": "canonical/background.inpaint.png",
                "draw_order": 1,
            },
        ],
        "jobs": [{"id": "unchanged", "status": "planned"}],
    }
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return queue_path, queue


def test_end_to_end_priority_a_forehead_preview_hashes_and_schema(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    original = queue_path.read_bytes()

    assert (
        deriver_main(
            [
                str(queue_path),
                "--base-dir",
                str(tmp_path),
                "--output-dir",
                "derived",
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )

    result: dict[str, Any] = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    assert queue_path.read_bytes() == original
    assert result["canvas"] == {"width": 20, "height": 20, "origin": [0, 0]}
    assert result["derived_from_segmentation"] == {"present": True, "run_ids": ["segment-001"]}
    assert {item["layer_id"] for item in result["input_masks"]} == {
        "face_base",
        "front_hair",
        "background",
    }
    assert result["canonical_queue_sha256"] == hashlib.sha256(original).hexdigest()
    assert (
        result["source_image_sha256"]
        == hashlib.sha256((tmp_path / "source.png").read_bytes()).hexdigest()
    )
    face = next(layer for layer in result["layers"] if layer["layer_id"] == "face_base")
    assert (
        face["target_mask"]["sha256"]
        == hashlib.sha256((tmp_path / "masks/face.png").read_bytes()).hexdigest()
    )
    assert face["candidates"]["protect"]["requires_review"] is True
    assert face["candidates"]["edge_extension"]["adjacent_layers"] == ["front_hair"]
    assert face["candidates"]["inpaint"]["status"] == "candidate"
    assert face["candidates"]["inpaint"]["requires_review"] is True
    for candidate in face["candidates"].values():
        if "candidate_id" not in candidate:
            continue
        with Image.open(tmp_path / candidate["soft_mask_file"]) as soft:
            assert soft.mode == "L"
            assert soft.size == (20, 20)
        with Image.open(tmp_path / candidate["preview_file"]) as preview:
            assert preview.size == (20, 20)
    with Image.open(tmp_path / face["candidates"]["protect"]["soft_mask_file"]) as protect:
        assert protect.histogram()[180] > 0
    assert not list(tmp_path.rglob("*.tmp"))

    result_schema = yaml.safe_load(
        (Path(__file__).parents[1] / "schemas/mask_derivation_result.schema.yaml").read_text(
            "utf-8"
        )
    )
    Draft202012Validator(result_schema).validate(result)

    assert (
        ranker_main(
            [
                "result.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "review.yaml",
                "--execute",
            ]
        )
        == 0
    )
    review = yaml.safe_load((tmp_path / "review.yaml").read_text("utf-8"))
    assert review["review_status"] == "pending"
    assert all(layer["requires_review"] is True for layer in review["layers"])
    review_schema = yaml.safe_load(
        (Path(__file__).parents[1] / "schemas/mask_derivation_review.schema.yaml").read_text(
            "utf-8"
        )
    )
    Draft202012Validator(review_schema).validate(review)
    assert queue == yaml.safe_load(queue_path.read_text("utf-8"))


def test_background_exclusion_canvas_clip_draw_order_and_unavailable_are_reported(
    tmp_path: Path,
) -> None:
    queue_path, _queue = _fixture(tmp_path)
    data = yaml.safe_load(queue_path.read_text("utf-8"))
    data["assets"][0]["draw_order"] = 30
    data["assets"][0]["target_mask"] = "masks/edge.png"
    _mask(tmp_path / "masks/edge.png", (0, 7, 5, 18))
    queue_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    result = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    face = next(layer for layer in result["layers"] if layer["layer_id"] == "face_base")
    conflict_types = {conflict["type"] for conflict in face["conflicts"]}
    assert "canvas_clipped" in conflict_types
    assert "draw_order_contradiction" in conflict_types
    assert face["candidates"]["inpaint"]["status"] == "unavailable"
    with Image.open(tmp_path / face["candidates"]["edge_extension"]["soft_mask_file"]) as edge:
        assert edge.getpixel((0, 6)) == 0


def test_dry_run_and_output_safety_guards(tmp_path: Path) -> None:
    queue_path, _queue = _fixture(tmp_path)
    assert (
        deriver_main([str(queue_path), "--base-dir", str(tmp_path), "--output", "planned.yaml"])
        == 0
    )
    assert not (tmp_path / "planned.yaml").exists()
    assert not (tmp_path / "mask-derivation").exists()

    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "masks/face.png",
                "--execute",
            ]
        )
        == 2
    )

    (tmp_path / "source.png").unlink()
    assert (
        deriver_main(["queue.yaml", "--base-dir", str(tmp_path), "--output", "missing-source.yaml"])
        == 2
    )
    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "../escape.yaml",
                "--execute",
            ]
        )
        == 2
    )


def test_result_is_deterministic_and_failed_layer_retry_selection(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    first, first_artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        config=DerivationConfig(),
    )
    second, second_artifacts = derive_masks(
        deepcopy(queue),
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        config=DerivationConfig(),
    )
    assert first == second
    assert first_artifacts.images.keys() == second_artifacts.images.keys()
    assert all(
        first_artifacts.images[path].tobytes() == second_artifacts.images[path].tobytes()
        for path in first_artifacts.images
    )

    retry = deepcopy(first)
    retry["layers"][0]["status"] = "failed"
    (tmp_path / "retry.yaml").write_text(yaml.safe_dump(retry, sort_keys=False), "utf-8")
    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--retry-failed-from",
                "retry.yaml",
                "--output",
                "retry-result.yaml",
            ]
        )
        == 0
    )


def test_foreground_occluder_ring_is_kept_and_prioritized(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    hair_ring = Image.new("L", (20, 20), 0)
    draw = ImageDraw.Draw(hair_ring)
    draw.rectangle((3, 5, 16, 19), fill=255)
    draw.rectangle((5, 7, 14, 18), fill=0)
    hair_ring.save(tmp_path / "masks/hair.png")

    result, _artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
    )

    face = next(layer for layer in result["layers"] if layer["layer_id"] == "face_base")
    edge = face["candidates"]["edge_extension"]
    assert edge["area_px"] > 0
    assert edge["method"] == "occluder_aware_target_dilation_ring"
    assert edge["adjacent_layers"] == ["front_hair"]


def test_lossy_slug_collision_cannot_merge_candidate_ids_or_outputs(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    queue["assets"] = queue["assets"][:2]
    queue["assets"][0]["layer_id"] = "eye/L"
    queue["assets"][1]["layer_id"] = "eye-L"
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), "utf-8")

    result, artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
    )

    candidate_ids = [
        candidate["candidate_id"]
        for layer in result["layers"]
        for candidate in layer["candidates"].values()
        if "candidate_id" in candidate
    ]
    output_paths = [
        candidate["soft_mask_file"]
        for layer in result["layers"]
        for candidate in layer["candidates"].values()
        if "candidate_id" in candidate
    ]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert len(output_paths) == len(set(output_paths))
    assert len(artifacts.images) == len(set(artifacts.images))


def test_artifact_scope_changes_with_content_even_for_reused_explicit_run_id(
    tmp_path: Path,
) -> None:
    queue_path, queue = _fixture(tmp_path)
    first, _ = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        run_id="manual-run",
    )
    first_paths = {
        candidate["soft_mask_file"]
        for layer in first["layers"]
        for candidate in layer["candidates"].values()
        if "candidate_id" in candidate
    }
    Image.new("RGBA", (20, 20), (90, 70, 50, 255)).save(tmp_path / "source.png")
    second, _ = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        run_id="manual-run",
    )
    second_paths = {
        candidate["soft_mask_file"]
        for layer in second["layers"]
        for candidate in layer["candidates"].values()
        if "candidate_id" in candidate
    }
    assert first_paths.isdisjoint(second_paths)


def test_inter_layer_inpaint_conflict_is_rendered_yellow_in_preview(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    Image.new("RGBA", (40, 40), (100, 80, 60, 255)).save(tmp_path / "source.png")
    face_mask = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(face_mask).rectangle((10, 24, 29, 38), fill=180)
    face_mask.save(tmp_path / "masks/face.png")
    hair_mask = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(hair_mask).rectangle((8, 8, 31, 25), fill=210)
    hair_mask.save(tmp_path / "masks/hair.png")
    background_mask = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(background_mask).rectangle((0, 0, 0, 39), fill=255)
    background_mask.save(tmp_path / "masks/background.png")
    queue["canvas"] = {"width": 40, "height": 40}
    queue["assets"][0]["expected_region"] = {
        "x_min": 0.15,
        "y_min": 0.15,
        "x_max": 0.85,
        "y_max": 0.98,
    }
    second_face = deepcopy(queue["assets"][0])
    second_face["layer_id"] = "face_overlay"
    second_face["draw_order"] = 11
    queue["assets"].insert(1, second_face)
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), "utf-8")

    result, artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
    )

    faces = [
        layer for layer in result["layers"] if layer["layer_id"] in {"face_base", "face_overlay"}
    ]
    assert all(
        any(conflict["type"] == "inter_layer_inpaint_overlap" for conflict in layer["conflicts"])
        for layer in faces
    )
    first_candidate = faces[0]["candidates"]["inpaint"]
    second_candidate = faces[1]["candidates"]["inpaint"]
    first_mask = artifacts.images[tmp_path / first_candidate["soft_mask_file"]]
    second_mask = artifacts.images[tmp_path / second_candidate["soft_mask_file"]]
    preview = artifacts.images[tmp_path / first_candidate["preview_file"]]
    overlap_points = [
        (x, y)
        for y in range(40)
        for x in range(40)
        if first_mask.getpixel((x, y)) and second_mask.getpixel((x, y))
    ]
    assert overlap_points
    red, green, blue, _alpha = preview.getpixel(overlap_points[-1])
    assert red > 180 and green > 140 and blue < 120


def test_partial_run_hashes_and_rechecks_unselected_context_masks(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--layer",
                "face_base",
                "--output",
                "partial.yaml",
                "--execute",
            ]
        )
        == 0
    )
    result = yaml.safe_load((tmp_path / "partial.yaml").read_text("utf-8"))
    assert [layer["layer_id"] for layer in result["layers"]] == ["face_base"]
    assert {item["layer_id"] for item in result["input_masks"]} == {
        "face_base",
        "front_hair",
        "background",
    }
    queue = yaml.safe_load((tmp_path / "queue.yaml").read_text("utf-8"))
    full, _ = derive_masks(
        queue,
        queue_path=tmp_path / "queue.yaml",
        base_dir=tmp_path,
        output_dir=tmp_path / "mask-derivation",
    )
    assert full["run_id"] != result["run_id"]
    full_face = next(layer for layer in full["layers"] if layer["layer_id"] == "face_base")
    partial_face = result["layers"][0]
    assert {
        candidate["preview_file"]
        for candidate in full_face["candidates"].values()
        if "candidate_id" in candidate
    }.isdisjoint(
        {
            candidate["preview_file"]
            for candidate in partial_face["candidates"].values()
            if "candidate_id" in candidate
        }
    )
    _mask(tmp_path / "masks/hair.png", (1, 1, 2, 2))
    assert (
        ranker_main(
            [
                "partial.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "partial-review.yaml",
                "--execute",
            ]
        )
        == 2
    )


def test_dry_run_derivation_retains_no_encoded_artifacts(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    dry_result, artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        retain_artifacts=False,
    )
    assert artifacts.png_bytes == {}
    assert artifacts.output_paths
    execute_result, execute_artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
    )
    try:
        assert dry_result == execute_result
    finally:
        execute_artifacts.close()


def test_layer_scope_serialization_is_unambiguous(tmp_path: Path) -> None:
    queue_path, queue = _fixture(tmp_path)
    template = queue["assets"][0]
    queue["assets"] = []
    for draw_order, layer_id in enumerate(("a,b", "c", "a", "b,c", "x"), start=1):
        asset = deepcopy(template)
        asset["layer_id"] = layer_id
        asset["role"] = "torso"
        asset["draw_order"] = draw_order
        queue["assets"].append(asset)
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), "utf-8")

    first, first_artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        layer_ids={"a,b", "c", "x"},
    )
    second, second_artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        layer_ids={"a", "b,c", "x"},
    )
    try:
        assert first["run_id"] != second["run_id"]
        assert first["execution_scope"] != second["execution_scope"]
    finally:
        first_artifacts.close()
        second_artifacts.close()


@pytest.mark.parametrize("tamper", ["queue", "source", "target"])
def test_ranking_rejects_queue_source_or_input_mask_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    _fixture(tmp_path)
    assert (
        deriver_main(
            [
                "queue.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    if tamper == "queue":
        (tmp_path / "queue.yaml").write_text("project: changed\n", "utf-8")
    elif tamper == "source":
        Image.new("RGBA", (20, 20), (0, 0, 0, 255)).save(tmp_path / "source.png")
    else:
        _mask(tmp_path / "masks/face.png", (1, 1, 2, 2))
    assert (
        ranker_main(
            [
                "result.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "review.yaml",
                "--execute",
            ]
        )
        == 2
    )
