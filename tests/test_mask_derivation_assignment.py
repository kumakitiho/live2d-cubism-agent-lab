from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_mask_derivation_pipeline import _fixture
from tools.backends.segmentation.integrity import file_sha256
from tools.mask_derivation.pipeline import derive_masks
from tools.mask_derivation_assignment import apply_review_plan
from tools.mask_derivation_assignment import main as assignment_main
from tools.mask_derivation_ranker import build_review_plan


def _documents(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    queue_path, queue = _fixture(tmp_path)
    result, artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
    )
    for path, image in artifacts.images.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    result_path = tmp_path / "result.yaml"
    result_path.write_text(yaml.safe_dump(result, sort_keys=False), "utf-8")
    review = build_review_plan(
        result,
        result_ref="result.yaml",
        result_sha256=file_sha256(result_path),
    )
    return queue_path, queue, result, review


def _approve(review: dict[str, Any], layer_id: str) -> None:
    review["review_status"] = "approved"
    layer = next(value for value in review["layers"] if value["layer_id"] == layer_id)
    layer["status"] = "approved"
    layer["requires_review"] = False


def test_apply_requires_review_and_updates_only_approved_layer_fields(tmp_path: Path) -> None:
    _queue_path, queue, result, review = _documents(tmp_path)
    original = deepcopy(queue)
    with pytest.raises(ValueError, match="review_status: approved"):
        apply_review_plan(queue, review, result)

    _approve(review, "face_base")
    updated, approved = apply_review_plan(queue, review, result)

    assert approved == {"face_base"}
    assert queue == original
    assert updated["assets"][1:] == original["assets"][1:]
    changed = updated["assets"][0]
    assert changed["target_mask"] == original["assets"][0]["target_mask"]
    assert changed["source_file"] == original["assets"][0]["source_file"]
    assert ".protect." in changed["protect_mask"] and changed["protect_mask"].endswith(".soft.png")
    assert ".edge_extension." in changed["edge_extension_mask"] and changed[
        "edge_extension_mask"
    ].endswith(".soft.png")
    assert ".inpaint." in changed["inpaint_mask"] and changed["inpaint_mask"].endswith(".soft.png")
    assert changed["mask_derivation_run_id"] == result["run_id"]
    assert changed["mask_derivation_status"] == "approved"


def test_apply_rejects_mixed_run_id_and_changed_result(tmp_path: Path) -> None:
    queue_path, queue, result, review = _documents(tmp_path)
    _approve(review, "face_base")
    face = next(layer for layer in result["layers"] if layer["layer_id"] == "face_base")
    face["candidates"]["protect"]["run_id"] = "other-run"
    with pytest.raises(ValueError, match="different run ID"):
        apply_review_plan(queue, review, result)

    result_path = tmp_path / "result.yaml"
    clean_result = yaml.safe_load(result_path.read_text("utf-8"))
    review_path = tmp_path / "review.yaml"
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), "utf-8")
    result_path.write_text(yaml.safe_dump(clean_result | {"status": "partial_failure"}), "utf-8")
    assert (
        assignment_main(
            [
                str(queue_path),
                str(review_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "assigned.yaml",
                "--execute",
            ]
        )
        == 2
    )


def test_cli_apply_preserves_unapproved_queue_content_and_is_atomic(tmp_path: Path) -> None:
    queue_path, queue, _result, review = _documents(tmp_path)
    original_bytes = queue_path.read_bytes()
    _approve(review, "face_base")
    review_path = tmp_path / "review.yaml"
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), "utf-8")

    assert (
        assignment_main(
            [
                str(queue_path),
                str(review_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "assigned.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assert queue_path.read_bytes() == original_bytes
    assigned = yaml.safe_load((tmp_path / "assigned.yaml").read_text("utf-8"))
    assert assigned["assets"][1:] == queue["assets"][1:]
    assert assigned["jobs"] == queue["jobs"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_apply_rejects_approved_candidate_png_tampering(tmp_path: Path) -> None:
    queue_path, _queue, _result, review = _documents(tmp_path)
    _approve(review, "face_base")
    selected = next(layer for layer in review["layers"] if layer["layer_id"] == "face_base")
    result = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    candidate_id = selected["selected"]["protect_candidate_id"]
    candidate = next(
        candidate
        for layer in result["layers"]
        for candidate in layer["candidates"].values()
        if candidate.get("candidate_id") == candidate_id
    )
    from PIL import Image

    Image.new("L", (20, 20), 255).save(tmp_path / candidate["soft_mask_file"])
    review_path = tmp_path / "review.yaml"
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), "utf-8")

    assert (
        assignment_main(
            [
                str(queue_path),
                str(review_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "tampered-assigned.yaml",
                "--execute",
            ]
        )
        == 2
    )


def test_apply_ignores_missing_unapproved_candidate_artifact(tmp_path: Path) -> None:
    queue_path, _queue, result, review = _documents(tmp_path)
    _approve(review, "face_base")
    unapproved = next(layer for layer in result["layers"] if layer["layer_id"] == "front_hair")
    (tmp_path / unapproved["candidates"]["protect"]["soft_mask_file"]).unlink()
    review_path = tmp_path / "review.yaml"
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), "utf-8")

    assert (
        assignment_main(
            [
                str(queue_path),
                str(review_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "selected-only.yaml",
                "--execute",
            ]
        )
        == 0
    )
