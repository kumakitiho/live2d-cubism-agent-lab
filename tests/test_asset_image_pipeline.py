from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    load_and_validate_mask_manifest,
    validate_asset_quality,
    validate_mask_manifest,
)
from tools.asset_quality_evaluator import (
    count_overlap_deficit,
    count_transparent_holes,
    count_white_halo,
    difference_score,
    evaluate_part,
)
from tools.asset_recomposer import difference_image, recompose_parts
from tools.asset_refinement_planner import (
    apply_refinement_plan,
    build_refinement_plan,
    next_generation_method,
)
from tools.hidden_region_completer import transparency_fill
from tools.mask_candidate_generator import build_mask_manifest
from tools.motion_stress_tester import create_motion_stress_preview, shift_part
from tools.part_extractor import extract_rgba


def _mask(size: tuple[int, int], points: set[tuple[int, int]]) -> Image.Image:
    image = Image.new("L", size, 0)
    for point in points:
        image.putpixel(point, 255)
    return image


def test_target_mask_rgba_extraction_preserves_canvas_and_source_pixels() -> None:
    source = Image.new("RGBA", (4, 3), (20, 40, 60, 255))
    source.putpixel((2, 1), (100, 110, 120, 128))
    target = _mask(source.size, {(2, 1)})

    result = extract_rgba(source, target)

    assert result.size == source.size
    assert result.getpixel((2, 1)) == (100, 110, 120, 128)
    assert result.getpixel((0, 0))[3] == 0


def test_extract_rejects_mismatched_canvas() -> None:
    source = Image.new("RGBA", (4, 3))
    target = Image.new("L", (3, 3))

    try:
        extract_rgba(source, target)
    except ValueError as exc:
        assert "canvas mismatch" in str(exc)
    else:
        raise AssertionError("canvas mismatch must be rejected")


def test_recompose_uses_draw_order_and_difference_image() -> None:
    bottom = Image.new("RGBA", (3, 3), (255, 0, 0, 255))
    top = Image.new("RGBA", (3, 3), (0, 0, 255, 0))
    top.putpixel((1, 1), (0, 0, 255, 255))

    result = recompose_parts((3, 3), [(20, top), (10, bottom)])

    assert result.getpixel((0, 0)) == (255, 0, 0, 255)
    assert result.getpixel((1, 1)) == (0, 0, 255, 255)
    assert difference_image(result, result).getbbox() is None
    changed = result.copy()
    changed.putpixel((0, 0), (0, 255, 0, 255))
    assert difference_image(result, changed).getbbox() is not None
    assert difference_score(result, result) == 0.0
    assert difference_score(result, changed) > 0.0


def test_quality_checks_detect_halo_holes_and_overlap_deficit() -> None:
    source = Image.new("RGBA", (5, 5), (180, 20, 20, 255))
    target = _mask((5, 5), {(2, 2), (3, 2)})
    part = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
    part.putpixel((2, 2), (255, 255, 255, 255))

    assert count_white_halo(part, source) == 1
    assert count_transparent_holes(part, target) == 1
    assert count_overlap_deficit(part, _mask((5, 5), {(2, 2)}), 1) == 8


def test_quality_gate_detects_modified_protected_source_pixel() -> None:
    source = Image.new("RGBA", (3, 3), (20, 40, 60, 255))
    part = source.copy()
    part.putpixel((1, 1), (21, 40, 60, 255))
    target = _mask(source.size, {(1, 1)})

    result = evaluate_part(
        part,
        source,
        target,
        overlap_margin_px=0,
        protect_mask=target,
    )

    assert result["quality_status"] == "fail"
    assert "source_pixel_difference" in result["failed_checks"]


def test_transparency_fill_respects_protect_mask() -> None:
    part = Image.new("RGBA", (5, 1), (0, 0, 0, 0))
    part.putpixel((2, 0), (40, 80, 120, 255))
    inpaint = _mask((5, 1), {(1, 0), (3, 0)})
    protect = _mask((5, 1), {(1, 0)})

    result = transparency_fill(part, inpaint, protect, iterations=1)

    assert result.getpixel((1, 0))[3] == 0
    assert result.getpixel((3, 0)) == (40, 80, 120, 255)
    assert result.getpixel((2, 0)) == part.getpixel((2, 0))


def test_motion_stress_preview_moves_part_without_changing_canvas_origin() -> None:
    part = Image.new("RGBA", (5, 3), (0, 0, 0, 0))
    part.putpixel((2, 1), (255, 0, 0, 255))

    shifted = shift_part(part, 1, 0)
    preview = create_motion_stress_preview(part, 1)

    assert shifted.size == part.size
    assert shifted.getpixel((3, 1)) == (255, 0, 0, 255)
    assert shifted.getpixel((2, 1))[3] == 0
    assert preview.size == (15, 3)
    assert preview.getpixel((1, 1)) == (255, 0, 0, 255)
    assert preview.getpixel((7, 1)) == (255, 0, 0, 255)
    assert preview.getpixel((13, 1)) == (255, 0, 0, 255)


def test_refinement_plan_requeues_failed_part_only() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    original = deepcopy(queue)

    plan = build_refinement_plan(queue, quality, quality_ref="quality.yaml")
    refined = apply_refinement_plan(queue, plan)

    assert [job["layer_id"] for job in plan["jobs"]] == ["eye_white_L"]
    before = {asset["layer_id"]: asset for asset in original["assets"]}
    after = {asset["layer_id"]: asset for asset in refined["assets"]}
    assert after["eye_white_L"]["generation_method"] == "extract_and_edge_repair"
    assert after["eye_white_L"]["refinement_attempts"] == 1
    assert after["eye_white_R"] == before["eye_white_R"]


def test_refinement_uses_source_preserving_generation_priority() -> None:
    assert next_generation_method("extract") == "extract_and_edge_repair"
    assert next_generation_method("extract_and_edge_repair") == "transparency_fill"
    assert next_generation_method("transparency_fill") == "inpaint"
    assert next_generation_method("inpaint") == "redraw"
    assert next_generation_method("redraw") == "redraw"


def test_refinement_rejects_partial_quality_coverage() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["parts"].pop()
    quality["summary"]["total_parts"] -= 1

    try:
        build_refinement_plan(queue, quality, quality_ref="quality.yaml")
    except ValueError as exc:
        assert "cover every import asset" in str(exc)
    else:
        raise AssertionError("partial quality coverage must be rejected")


def test_refinement_stops_after_three_failed_attempts() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    queue["assets"][0]["refinement_attempts"] = 3

    try:
        build_refinement_plan(queue, quality, quality_ref="quality.yaml")
    except ValueError as exc:
        assert "manual review required" in str(exc)
    else:
        raise AssertionError("fourth automatic refinement must be blocked")


def test_refinement_rejects_other_project_quality() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["project"] = "other-project"

    try:
        build_refinement_plan(queue, quality, quality_ref="quality.yaml")
    except ValueError as exc:
        assert "project must match" in str(exc)
    else:
        raise AssertionError("cross-project quality must be rejected")


def test_refinement_rejects_quality_from_another_queue_run() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["derived_from"]["asset_generation_queue"] = "generated/other-queue.yaml"

    try:
        build_refinement_plan(
            queue,
            quality,
            quality_ref="quality.yaml",
            queue_ref="examples/asset_generation_queue.sample.yaml",
        )
    except ValueError as exc:
        assert "reference the queue" in str(exc)
    else:
        raise AssertionError("quality from another queue run must be rejected")


def test_pipeline_samples_and_schemas_are_loadable_and_valid() -> None:
    mask_manifest = load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))
    asset_quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))

    assert validate_mask_manifest(mask_manifest) == []
    assert validate_asset_quality(asset_quality) == []
    for path in (
        Path("schemas/mask_manifest.schema.yaml"),
        Path("schemas/asset_quality.schema.yaml"),
    ):
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["$schema"].startswith("https://json-schema.org/")


def test_mask_manifest_requires_queue_provenance() -> None:
    manifest = load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))
    manifest.pop("derived_from")

    issues = validate_mask_manifest(manifest)

    assert any(issue.path == "derived_from" for issue in issues)


def test_checked_in_mask_manifest_is_exact_queue_derivative() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))

    assert build_mask_manifest(
        queue,
        queue_ref="examples/asset_generation_queue.sample.yaml",
    ) == load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))


def test_queue_derived_mask_manifest_drift_is_rejected(tmp_path: Path) -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    (tmp_path / "queue.yaml").write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = build_mask_manifest(queue, queue_ref="queue.yaml")
    manifest["parts"][0]["generation_method"] = "extract_and_edge_repair"
    manifest_path = tmp_path / "mask_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    try:
        load_and_validate_mask_manifest(manifest_path, base_dir=tmp_path)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("queue-derived mask manifest drift must be rejected")


def test_global_reconstruction_difference_is_a_quality_failure() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    for part in quality["parts"]:
        part["quality_status"] = "pass"
        part["failed_checks"] = []
        for key in part["metrics"]:
            part["metrics"][key] = 0
    quality["summary"]["failed_parts"] = 0
    quality["summary"]["global_difference_score"] = 0.1
    quality["summary"]["result"] = "fail"

    assert validate_asset_quality(quality) == []


def test_quality_status_and_failed_checks_must_match_metrics() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["parts"][0]["quality_status"] = "pass"
    quality["parts"][0]["failed_checks"].remove("source_pixel_difference")

    issues = validate_asset_quality(quality)

    assert any("must exactly match" in issue.message for issue in issues)
    assert any("must equal fail" in issue.message for issue in issues)
