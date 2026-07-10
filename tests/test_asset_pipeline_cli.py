from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from tools.artifact_validation import load_yaml_mapping
from tools.asset_quality_evaluator import main as quality_main
from tools.asset_recomposer import main as recompose_main
from tools.asset_refinement_planner import main as refinement_main
from tools.hidden_region_completer import main as completer_main
from tools.mask_candidate_generator import build_mask_manifest
from tools.mask_candidate_generator import main as mask_main
from tools.motion_stress_tester import main as motion_main
from tools.part_extractor import main as extractor_main


def test_all_new_tools_support_dry_run() -> None:
    queue = "examples/asset_generation_queue.sample.yaml"
    manifest = "examples/mask_manifest.sample.yaml"

    assert mask_main([queue, "--base-dir", ".", "--output", "generated/masks.yaml"]) == 0
    assert extractor_main([manifest, "--part", "eye_white_L", "--base-dir", "."]) == 0
    assert completer_main([manifest, "--part", "eye_white_L", "--base-dir", "."]) == 0
    assert (
        recompose_main(
            [
                manifest,
                "--base-dir",
                ".",
                "--output",
                "generated/reconstructed.png",
                "--difference-output",
                "generated/difference.png",
            ]
        )
        == 0
    )
    assert (
        quality_main(
            [
                manifest,
                "--base-dir",
                ".",
                "--reconstructed",
                "generated/reconstructed.png",
                "--difference-output",
                "generated/difference.png",
                "--output",
                "generated/quality.yaml",
            ]
        )
        == 0
    )
    assert (
        motion_main(
            [
                manifest,
                "--part",
                "eye_white_L",
                "--base-dir",
                ".",
                "--output",
                "generated/motion.png",
            ]
        )
        == 0
    )
    assert (
        refinement_main(
            [
                    queue,
                    "examples/asset_quality.sample.yaml",
                    "--output",
                    "generated/test-refinement-plan.yaml",
                    "--refined-queue-output",
                    "generated/test-refined-queue.yaml",
            ]
        )
        == 0
    )


def test_png_fixture_extract_recompose_quality_and_motion_pipeline(tmp_path: Path) -> None:
    source = Image.new("RGBA", (6, 4), (0, 0, 0, 0))
    for point in ((2, 1), (3, 1), (2, 2), (3, 2)):
        source.putpixel(point, (80, 120, 160, 255))
    source_path = tmp_path / "source.png"
    source.save(source_path)
    target = Image.new("L", source.size, 0)
    for point in ((2, 1), (3, 1), (2, 2), (3, 2)):
        target.putpixel(point, 255)
    target.save(tmp_path / "target.png")
    target.save(tmp_path / "protect.png")
    Image.new("L", source.size, 0).save(tmp_path / "inpaint.png")
    queue = {
        "schema_version": 3,
        "project": "fixture",
        "source_image": {"path": "source.png"},
        "canvas": {"width": 6, "height": 4},
        "assets": [
            {
                "layer_id": "fixture_part",
                "target_mask": "target.png",
                "protect_mask": "protect.png",
                "inpaint_mask": "inpaint.png",
                "source_file": "parts/fixture.png",
                "generation_method": "extract",
                "dependencies": [],
                "draw_order": 1,
                "overlap_margin_px": 0,
                "quality_status": "pending",
                "refinement_attempts": 0,
                "include_in_import": True,
            }
        ],
    }
    (tmp_path / "queue.yaml").write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = build_mask_manifest(queue, queue_ref="queue.yaml")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert (
        extractor_main(
            [
                str(manifest_path),
                "--part",
                "fixture_part",
                "--base-dir",
                str(tmp_path),
                "--execute",
            ]
        )
        == 0
    )
    assert (
        recompose_main(
            [
                str(manifest_path),
                "--base-dir",
                str(tmp_path),
                "--output",
                "reconstructed.png",
                "--difference-output",
                "recompose_difference.png",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        quality_main(
            [
                str(manifest_path),
                "--base-dir",
                str(tmp_path),
                "--reconstructed",
                "reconstructed.png",
                "--difference-output",
                "quality_difference.png",
                "--output",
                "quality.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        motion_main(
            [
                str(manifest_path),
                "--part",
                "fixture_part",
                "--distance",
                "1",
                "--base-dir",
                str(tmp_path),
                "--output",
                "motion.png",
                "--execute",
            ]
        )
        == 0
    )

    with Image.open(tmp_path / "parts/fixture.png") as extracted:
        assert extracted.size == source.size
    with Image.open(tmp_path / "reconstructed.png") as reconstructed:
        assert reconstructed.convert("RGBA").tobytes() == source.tobytes()
    with Image.open(tmp_path / "quality_difference.png") as difference:
        assert difference.getbbox() is None
    quality = load_yaml_mapping(tmp_path / "quality.yaml")
    assert quality["summary"]["result"] == "pass"
    with Image.open(tmp_path / "motion.png") as preview:
        assert preview.size == (18, 4)

    original_source = source_path.read_bytes()
    assert (
        extractor_main(
            [
                str(manifest_path),
                "--part",
                "fixture_part",
                "--base-dir",
                str(tmp_path),
                "--output",
                "source.png",
                "--execute",
                "--force",
            ]
        )
        == 2
    )
    assert source_path.read_bytes() == original_source

    original_target = (tmp_path / "target.png").read_bytes()
    assert (
        quality_main(
            [
                str(manifest_path),
                "--base-dir",
                str(tmp_path),
                "--reconstructed",
                "reconstructed.png",
                "--difference-output",
                "target.png",
                "--output",
                "quality-collision.yaml",
                "--execute",
                "--force",
            ]
        )
        == 2
    )
    assert (tmp_path / "target.png").read_bytes() == original_target

    wrong_queue = dict(queue)
    wrong_queue["canvas"] = {"width": 7, "height": 4}
    (tmp_path / "wrong_queue.yaml").write_text(
        yaml.safe_dump(wrong_queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    wrong_canvas = build_mask_manifest(wrong_queue, queue_ref="wrong_queue.yaml")
    wrong_manifest_path = tmp_path / "wrong_canvas.yaml"
    wrong_manifest_path.write_text(
        yaml.safe_dump(wrong_canvas, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    assert (
        extractor_main(
            [
                str(wrong_manifest_path),
                "--part",
                "fixture_part",
                "--base-dir",
                str(tmp_path),
                "--output",
                "parts/wrong.png",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "parts/wrong.png").exists()


def test_refinement_execute_rejects_output_outside_base_dir(tmp_path: Path) -> None:
    output = tmp_path / "same.yaml"

    assert (
        refinement_main(
            [
                "examples/asset_generation_queue.sample.yaml",
                "examples/asset_quality.sample.yaml",
                "--output",
                str(output),
                "--refined-queue-output",
                str(output),
                "--execute",
            ]
        )
        == 2
    )
    assert not output.exists()
