from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw

from tools.segmentation_candidate_ranker import main as ranker_main
from tools.segmentation_candidate_ranker import rank_candidates


def _mask(path: Path, box: tuple[int, int, int, int], *, size: tuple[int, int] = (10, 8)) -> int:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(box, fill=255)
    image.save(path)
    return image.histogram()[255]


def _candidate(
    candidate_id: str,
    mask_file: str,
    bbox: list[int],
    area: int,
    *,
    confidence: float = 0.8,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "layer_id": "eye_white_L",
        "semantic_prompt": "left eye white",
        "mask_file": mask_file,
        "soft_mask_file": mask_file,
        "binary_mask_file": mask_file,
        "preview_file": "unused.png",
        "confidence": confidence,
        "stability_score": 0.9,
        "bbox_xyxy": bbox,
        "area_px": area,
        "source_backend": "mock",
        "model_id": "mock-fixture-v1",
        "model_revision": "1",
        "prompt_provenance": {},
        "side": "L",
        "role": "eye_white",
        "expected_region": None,
        "draw_order": 10,
        "requires_review": False,
        "rejection_reasons": [],
    }


def _result(tmp_path: Path) -> dict[str, Any]:
    Image.new("RGBA", (10, 8), (50, 50, 50, 255)).save(tmp_path / "source.png")
    left_area = _mask(tmp_path / "left.png", (1, 2, 3, 4))
    right_area = _mask(tmp_path / "right.png", (6, 2, 8, 4))
    return {
        "schema_version": 1,
        "status": "completed",
        "project": "rank-test",
        "run_id": "run-1",
        "asset_generation_queue": "queue.yaml",
        "source_image": {"path": "source.png"},
        "canvas": {"width": 10, "height": 8, "origin": [0, 0]},
        "backend": "mock",
        "candidates": [
            _candidate("left", "left.png", [1, 2, 4, 5], left_area, confidence=0.78),
            _candidate("right", "right.png", [6, 2, 9, 5], right_area, confidence=0.82),
        ],
    }


def test_left_candidate_ranks_above_wrong_side_candidate(tmp_path: Path) -> None:
    ranked = rank_candidates(_result(tmp_path), base_dir=tmp_path)

    assert [candidate["candidate_id"] for candidate in ranked["candidates"]] == [
        "left",
        "right",
    ]
    assert ranked["candidates"][0]["rank"] == 1
    assert "side_position_mismatch" in ranked["candidates"][1]["rejection_reasons"]
    assert ranked["summary"]["automatic_assignment"] is False


def test_low_confidence_candidate_needs_review(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["candidates"] = [_candidate("low", "left.png", [1, 2, 4, 5], 9, confidence=0.2)]

    ranked = rank_candidates(result, base_dir=tmp_path)

    assert ranked["candidates"][0]["requires_review"] is True
    assert "low_confidence" in ranked["candidates"][0]["rejection_reasons"]


def test_duplicate_candidate_id_is_rejected(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["candidates"][1]["candidate_id"] = "left"

    with pytest.raises(ValueError, match="duplicate candidate ID"):
        rank_candidates(result, base_dir=tmp_path)


def test_mask_canvas_mismatch_is_rejected(tmp_path: Path) -> None:
    result = _result(tmp_path)
    Image.new("L", (9, 8), 255).save(tmp_path / "left.png")

    with pytest.raises(ValueError, match="mask canvas mismatch"):
        rank_candidates(result, base_dir=tmp_path)


def test_ranking_uses_published_binary_mask_not_nonzero_soft_fringe(tmp_path: Path) -> None:
    result = _result(tmp_path)
    Image.new("L", (10, 8), 1).save(tmp_path / "soft-fringe.png")
    binary = Image.new("L", (10, 8), 0)
    binary.putpixel((2, 2), 255)
    binary.save(tmp_path / "one-pixel.png")
    candidate = _candidate("thresholded", "soft-fringe.png", [2, 2, 3, 3], 1)
    candidate["binary_mask_file"] = "one-pixel.png"
    result["candidates"] = [candidate]

    ranked = rank_candidates(result, base_dir=tmp_path)

    metrics = ranked["candidates"][0]["ranking_metrics"]
    assert metrics["area_ratio"] == 0.0125
    assert "recorded_area_mismatch" not in ranked["candidates"][0]["rejection_reasons"]


def test_candidate_conflict_is_marked_for_different_layers(tmp_path: Path) -> None:
    result = _result(tmp_path)
    duplicate = dict(result["candidates"][0])
    duplicate["candidate_id"] = "overlapping-face"
    duplicate["layer_id"] = "face"
    duplicate["role"] = "face"
    duplicate["semantic_prompt"] = "face"
    result["candidates"] = [result["candidates"][0], duplicate]

    ranked = rank_candidates(result, base_dir=tmp_path)

    assert all(
        "candidate_conflict" in candidate["rejection_reasons"] for candidate in ranked["candidates"]
    )


def test_ranked_output_cannot_overwrite_canonical_queue(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result_path = tmp_path / "result.yaml"
    result_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text("project: must-survive\n", encoding="utf-8")
    before = queue_path.read_bytes()

    exit_code = ranker_main(
        [
            str(result_path),
            "--base-dir",
            str(tmp_path),
            "--output",
            str(queue_path),
            "--execute",
            "--force",
        ]
    )

    assert exit_code == 2
    assert queue_path.read_bytes() == before
