from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from PIL import Image

from tools.automatic_segmenter import build_request
from tools.automatic_segmenter import main as segmenter_main
from tools.backends.segmentation import MockSegmentationBackend
from tools.segmentation_assignment_planner import main as assignment_main
from tools.segmentation_candidate_ranker import main as ranker_main


def _write_queue(tmp_path: Path, *, source: str = "source.png") -> Path:
    queue: dict[str, Any] = {
        "schema_version": 3,
        "project": "segmentation-test",
        "source_image": {"path": source},
        "canvas": {"width": 8, "height": 6},
        "assets": [
            {
                "layer_id": "eye_white_L",
                "role": "eye_white",
                "side": "L",
                "draw_order": 10,
                "target_mask": "canonical/eye.target.png",
                "protect_mask": "canonical/eye.protect.png",
                "edge_extension_mask": "canonical/eye.edge.png",
                "inpaint_mask": "canonical/eye.inpaint.png",
                "segmentation": {
                    "semantic_prompt": "left eye white",
                    "point_prompts": [{"x": 2, "y": 2, "label": 1}],
                    "box_prompt": [1, 1, 5, 5],
                    "candidate_count": 1,
                },
            }
        ],
    }
    path = tmp_path / "queue.yaml"
    path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _source_and_fixture(tmp_path: Path, *, transparent: bool = False) -> tuple[Path, Path]:
    alpha = 0 if transparent else 255
    source = Image.new("RGBA", (8, 6), (30, 60, 90, alpha))
    source_path = tmp_path / "source.png"
    source.save(source_path)
    fixture = Image.new("L", (8, 6), 0)
    fixture.putpixel((2, 2), 63)
    fixture.putpixel((3, 2), 127)
    fixture.putpixel((4, 2), 220)
    fixture_path = tmp_path / "fixture.png"
    fixture.save(fixture_path)
    return source_path, fixture_path


def test_mock_end_to_end_preserves_soft_mask_and_canvas(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    source, fixture = _source_and_fixture(tmp_path)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "mock",
            "--fixture-mask",
            str(fixture),
            "--binary-threshold",
            "128",
            "--output",
            "result.yaml",
            "--execute",
        ]
    )

    assert exit_code == 0
    result: dict[str, Any] = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    assert result["status"] == "completed"
    assert result["asset_generation_queue_sha256"] == hashlib.sha256(
        queue.read_bytes()
    ).hexdigest()
    assert result["source_image_sha256"] == source_digest
    assert result["canvas"] == {"width": 8, "height": 6, "origin": [0, 0]}
    assert result["summary"]["automatic_assignment"] is False
    candidate = result["candidates"][0]
    assert candidate["semantic_assignment"] == {
        "layer_id": "eye_white_L",
        "role": "eye_white",
        "side": "L",
        "status": "proposed",
    }
    with Image.open(tmp_path / candidate["soft_mask_file"]) as soft:
        assert soft.mode == "L"
        assert soft.size == (8, 6)
        assert soft.getpixel((2, 2)) == 63
        assert soft.getpixel((3, 2)) == 127
        assert soft.getpixel((4, 2)) == 220
    with Image.open(tmp_path / candidate["binary_mask_file"]) as binary:
        assert binary.getpixel((2, 2)) == 0
        assert binary.getpixel((3, 2)) == 0
        assert binary.getpixel((4, 2)) == 255
    with Image.open(tmp_path / candidate["preview_file"]) as preview:
        assert preview.size == (8, 6)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    assert not list(tmp_path.rglob("*.tmp"))


def test_dry_run_does_not_require_source_or_write_output(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path, source="missing.png")

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "mock",
            "--output",
            "result.yaml",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "result.yaml").exists()
    assert not (tmp_path / "result_masks").exists()


def test_queue_prompt_fields_propagate_to_request(tmp_path: Path) -> None:
    queue_path = _write_queue(tmp_path)
    source, _fixture = _source_and_fixture(tmp_path)
    queue: dict[str, Any] = yaml.safe_load(queue_path.read_text("utf-8"))
    with Image.open(source) as opened:
        request = build_request(
            queue["assets"][0],
            opened.convert("RGBA"),
            base_dir=tmp_path,
            execute=True,
        )

    assert request.semantic_prompt == "left eye white"
    assert request.point_prompts[0].as_dict() == {"x": 2.0, "y": 2.0, "label": 1}
    assert request.box_prompt == (1.0, 1.0, 5.0, 5.0)


def test_fixture_canvas_mismatch_stops_without_outputs(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    _source, fixture = _source_and_fixture(tmp_path)
    Image.new("L", (7, 6), 255).save(fixture)

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "mock",
            "--fixture-mask",
            str(fixture),
            "--output",
            "result.yaml",
            "--execute",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "result.yaml").exists()


def test_source_change_during_segmentation_refuses_stale_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _write_queue(tmp_path)
    source, fixture = _source_and_fixture(tmp_path)
    original_segment = MockSegmentationBackend.segment

    def segment_and_replace_source(
        self: MockSegmentationBackend,
        request: Any,
        *,
        execute: bool = False,
    ) -> Any:
        result = original_segment(self, request, execute=execute)
        Image.new("RGBA", (8, 6), (200, 20, 20, 255)).save(source)
        return result

    monkeypatch.setattr(MockSegmentationBackend, "segment", segment_and_replace_source)

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "mock",
            "--fixture-mask",
            str(fixture),
            "--output",
            "result.yaml",
            "--execute",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "result.yaml").exists()
    assert not list(tmp_path.glob("result_masks/*.png"))


def test_output_collision_never_changes_canonical_queue(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    _source, fixture = _source_and_fixture(tmp_path)
    before = queue.read_bytes()

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "mock",
            "--fixture-mask",
            str(fixture),
            "--output",
            str(queue),
            "--execute",
            "--force",
        ]
    )

    assert exit_code == 2
    assert queue.read_bytes() == before


def test_mask_outside_source_alpha_requires_review(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    _source, fixture = _source_and_fixture(tmp_path, transparent=True)

    assert (
        segmenter_main(
            [
                str(queue),
                "--base-dir",
                str(tmp_path),
                "--backend",
                "mock",
                "--fixture-mask",
                str(fixture),
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    result: dict[str, Any] = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    candidate = result["candidates"][0]
    assert candidate["requires_review"] is True
    assert "mask_outside_source_alpha" in candidate["rejection_reasons"]


def test_empty_binary_candidate_is_structured_for_review(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    _source, fixture = _source_and_fixture(tmp_path)

    assert (
        segmenter_main(
            [
                str(queue),
                "--base-dir",
                str(tmp_path),
                "--backend",
                "mock",
                "--fixture-mask",
                str(fixture),
                "--binary-threshold",
                "255",
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    result: dict[str, Any] = yaml.safe_load((tmp_path / "result.yaml").read_text("utf-8"))
    candidate = result["candidates"][0]
    assert candidate["area_px"] == 0
    assert candidate["requires_review"] is True
    assert "empty_binary_mask" in candidate["rejection_reasons"]


def test_sam2_unavailable_stops_clearly_without_model_download(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    _source, _fixture = _source_and_fixture(tmp_path)

    exit_code = segmenter_main(
        [
            str(queue),
            "--base-dir",
            str(tmp_path),
            "--backend",
            "sam2",
            "--model-id",
            "not-local",
            "--output",
            "result.yaml",
            "--execute",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "result.yaml").exists()


def test_mock_full_workflow_requires_review_before_new_queue_candidate(
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path)
    _source, fixture = _source_and_fixture(tmp_path)
    original_queue_bytes = queue.read_bytes()
    assert (
        segmenter_main(
            [
                str(queue),
                "--base-dir",
                str(tmp_path),
                "--backend",
                "mock",
                "--fixture-mask",
                str(fixture),
                "--output",
                "result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        ranker_main(
            [
                "result.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "ranked.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        assignment_main(
            [
                str(queue),
                "ranked.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "assignment.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assignment_path = tmp_path / "assignment.yaml"
    assignment: dict[str, Any] = yaml.safe_load(assignment_path.read_text("utf-8"))
    assignment["review_status"] = "approved"
    assignment["assignments"][0]["status"] = "approved"
    assignment["assignments"][0]["requires_review"] = False
    assignment_path.write_text(
        yaml.safe_dump(assignment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    assert (
        assignment_main(
            [
                "apply",
                str(queue),
                str(assignment_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "queue.segmented.yaml",
                "--execute",
            ]
        )
        == 0
    )

    updated: dict[str, Any] = yaml.safe_load(
        (tmp_path / "queue.segmented.yaml").read_text("utf-8")
    )
    assert updated["assets"][0]["target_mask"].endswith(".soft.png")
    assert updated["assets"][0]["segmentation_run_id"]
    assert queue.read_bytes() == original_queue_bytes


@pytest.mark.parametrize(
    "path",
    [
        "schemas/segmentation_result.schema.yaml",
        "schemas/segmentation_ranked.schema.yaml",
        "schemas/segmentation_assignment.schema.yaml",
        "examples/segmentation_result.sample.yaml",
        "examples/segmentation_ranked.sample.yaml",
        "examples/segmentation_assignment.sample.yaml",
    ],
)
def test_segmentation_schema_and_example_yaml_is_parseable(path: str) -> None:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    if path.startswith("schemas/"):
        assert data["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert isinstance(data["required"], list)
        assert all(isinstance(field, str) for field in data["required"])
        if path.endswith("segmentation_result.schema.yaml"):
            assert {
                "asset_generation_queue",
                "asset_generation_queue_sha256",
                "asset_generation_queue_content_sha256",
                "source_image",
                "source_image_sha256",
            } <= set(data["required"])
    else:
        assert data["schema_version"] == 1


@pytest.mark.parametrize(
    ("schema_path", "example_path"),
    [
        (
            "schemas/segmentation_result.schema.yaml",
            "examples/segmentation_result.sample.yaml",
        ),
        (
            "schemas/segmentation_ranked.schema.yaml",
            "examples/segmentation_ranked.sample.yaml",
        ),
        (
            "schemas/segmentation_assignment.schema.yaml",
            "examples/segmentation_assignment.sample.yaml",
        ),
    ],
)
def test_segmentation_examples_validate_against_draft_2020_12_schema(
    schema_path: str,
    example_path: str,
) -> None:
    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    example = yaml.safe_load(Path(example_path).read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(example)
