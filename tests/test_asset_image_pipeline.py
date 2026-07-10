from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from tools.artifact_validation import load_yaml_mapping
from tools.asset_manifest_validator import validate_asset_manifest
from tools.asset_pipeline_common import (
    atomic_save_png,
    load_and_validate_mask_manifest,
    load_binary_mask,
    load_soft_mask,
    validate_asset_quality,
    validate_mask_manifest,
)
from tools.asset_quality_evaluator import (
    allowed_change_region_mask,
    count_overlap_deficit,
    count_transparent_holes,
    count_white_halo,
    difference_score,
    evaluate_part,
    foreground_reconstruction_mask,
    premultiplied_difference_image,
)
from tools.asset_queue_builder import derive_asset_manifest
from tools.asset_recomposer import difference_image, recompose_parts
from tools.asset_refinement_planner import (
    apply_refinement_plan,
    build_refinement_plan,
    next_generation_method,
    select_generation_method,
)
from tools.hidden_region_completer import extract_and_edge_repair, transparency_fill
from tools.mask_candidate_generator import build_mask_manifest
from tools.motion_stress_tester import (
    create_motion_stress_preview,
    create_part_motion_debug_preview,
    shift_part,
)
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


def test_antialiased_target_mask_preserves_soft_alpha() -> None:
    source = Image.new("RGBA", (3, 1), (20, 40, 60, 255))
    target = Image.new("L", source.size)
    target.putdata([32, 128, 255])

    result = extract_rgba(source, target)

    assert list(result.getchannel("A").tobytes()) == [32, 128, 255]


def test_soft_and_binary_mask_loaders_have_distinct_semantics(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    mask = Image.new("L", (5, 1))
    mask.putdata([0, 32, 127, 128, 255])
    mask.save(path)

    soft = load_soft_mask(path, mask.size)
    binary = load_binary_mask(path, mask.size, alpha_threshold=128)

    assert list(soft.tobytes()) == [0, 32, 127, 128, 255]
    assert list(binary.tobytes()) == [0, 0, 0, 255, 255]


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


def test_premultiplied_comparison_ignores_rgb_of_fully_transparent_pixels() -> None:
    reference = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    candidate = Image.new("RGBA", (2, 1), (0, 255, 255, 0))

    assert difference_score(reference, candidate) == 0.0
    assert premultiplied_difference_image(reference, candidate).getbbox() is None


def test_foreground_reconstruction_mask_excludes_unmodeled_background() -> None:
    source = Image.new("RGBA", (3, 1), (20, 30, 40, 255))
    reconstructed = Image.new("RGBA", source.size, (0, 0, 0, 0))
    reconstructed.putpixel((1, 0), source.getpixel((1, 0)))
    target = _mask(source.size, {(1, 0)})

    foreground = foreground_reconstruction_mask([target], reconstructed)

    assert list(foreground.tobytes()) == [0, 255, 0]
    assert difference_score(source, reconstructed, foreground) == 0.0


def test_foreground_reconstruction_mask_detects_opaque_stray_pixels() -> None:
    source = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
    source.putpixel((1, 0), (20, 30, 40, 255))
    reconstructed = source.copy()
    reconstructed.putpixel((0, 0), (200, 100, 50, 255))
    target = _mask(source.size, {(1, 0)})

    foreground = foreground_reconstruction_mask([target], reconstructed)

    assert list(foreground.tobytes()) == [255, 255, 0]
    assert difference_score(source, reconstructed, foreground) > 0.0


def test_allowed_change_region_is_separate_from_preserve_region() -> None:
    source = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
    source.putpixel((1, 0), (20, 40, 60, 255))
    part = source.copy()
    part.putpixel((2, 0), (80, 120, 160, 255))
    target = _mask(source.size, {(1, 0)})
    protect = target.copy()
    edge_extension = Image.new("L", source.size, 0)
    inpaint = _mask(source.size, {(2, 0)})

    result = evaluate_part(
        part,
        source,
        target,
        overlap_margin_px=0,
        protect_mask=protect,
        edge_extension_mask=edge_extension,
        inpaint_mask=inpaint,
        reconstructed=source,
        thresholds={"max_allowed_change_region_difference_score": 1.0},
    )

    assert result["metrics"]["preserve_region_difference_score"] == 0.0
    assert result["metrics"]["allowed_change_region_difference_score"] > 0.0
    assert result["quality_status"] == "pass"
    assert result["failed_checks"] == []

    strict = evaluate_part(
        part,
        source,
        target,
        overlap_margin_px=0,
        protect_mask=protect,
        edge_extension_mask=edge_extension,
        inpaint_mask=inpaint,
        reconstructed=source,
        thresholds={"max_allowed_change_region_difference_score": 0.0},
    )
    assert "allowed_change_region_difference_score" in strict["failed_checks"]


def test_allowed_change_region_excludes_protected_pixels() -> None:
    edge_extension = _mask((3, 1), {(0, 0), (1, 0)})
    inpaint = _mask((3, 1), {(2, 0)})
    protect = _mask((3, 1), {(1, 0)})

    allowed = allowed_change_region_mask(
        edge_extension,
        inpaint,
        protect_mask=protect,
    )

    assert list(allowed.tobytes()) == [255, 0, 255]


def test_visual_reconstruction_difference_is_attributed_to_the_part() -> None:
    source = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    source.putpixel((1, 0), (40, 80, 120, 255))
    part = source.copy()
    reconstructed = source.copy()
    reconstructed.putpixel((1, 0), (120, 80, 40, 255))
    target = _mask(source.size, {(1, 0)})

    result = evaluate_part(
        part,
        source,
        target,
        overlap_margin_px=0,
        protect_mask=target,
        reconstructed=reconstructed,
        thresholds={"max_visual_reconstruction_difference_score": 0.0},
    )

    assert result["metrics"]["preserve_region_difference_score"] == 0.0
    assert result["metrics"]["visual_reconstruction_difference_score"] > 0.0
    assert result["failed_checks"] == ["visual_reconstruction_difference_score"]


def test_quality_checks_detect_halo_holes_and_overlap_deficit() -> None:
    source = Image.new("RGBA", (5, 5), (180, 20, 20, 255))
    target = _mask((5, 5), {(2, 2), (3, 2)})
    part = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
    part.putpixel((2, 2), (255, 255, 255, 255))

    assert count_white_halo(part, source) == 1
    assert count_transparent_holes(part, target) == 1
    assert count_overlap_deficit(part, _mask((5, 5), {(2, 2)}), 1) == 0


def test_non_zero_overlap_uses_explicit_extension_mask() -> None:
    target = _mask((5, 3), {(2, 1)})
    extension = _mask((5, 3), {(1, 1), (3, 1)})
    complete = Image.new("RGBA", target.size, (0, 0, 0, 0))
    for point in ((1, 1), (2, 1), (3, 1)):
        complete.putpixel(point, (40, 80, 120, 255))
    incomplete = complete.copy()
    incomplete.putpixel((3, 1), (0, 0, 0, 0))

    assert count_overlap_deficit(
        complete,
        target,
        3,
        edge_extension_mask=extension,
    ) == 0
    assert count_overlap_deficit(
        incomplete,
        target,
        3,
        edge_extension_mask=extension,
    ) == 1


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
    assert "preserve_region_difference_score" in result["failed_checks"]


def test_preserve_region_detects_sparse_one_lsb_difference_without_rounding() -> None:
    size = (512, 512)
    source = Image.new("RGBA", size, (20, 40, 60, 255))
    part = source.copy()
    part.putpixel((256, 256), (21, 40, 60, 255))
    protect = Image.new("L", size, 255)

    result = evaluate_part(
        part,
        source,
        protect,
        overlap_margin_px=0,
        protect_mask=protect,
    )

    assert result["metrics"]["preserve_region_difference_score"] > 0.0
    assert "preserve_region_difference_score" in result["failed_checks"]


def test_transparency_fill_respects_protect_mask() -> None:
    part = Image.new("RGBA", (5, 1), (0, 0, 0, 0))
    part.putpixel((2, 0), (40, 80, 120, 255))
    inpaint = _mask((5, 1), {(1, 0), (3, 0)})
    protect = _mask((5, 1), {(1, 0)})

    result = transparency_fill(part, inpaint, protect, iterations=1)

    assert result.getpixel((1, 0))[3] == 0
    assert result.getpixel((3, 0)) == (40, 80, 120, 255)
    assert result.getpixel((2, 0)) == part.getpixel((2, 0))


def test_edge_repair_changes_only_antialiased_fringe_rgb() -> None:
    part = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
    part.putpixel((0, 0), (255, 255, 255, 64))
    part.putpixel((1, 0), (40, 80, 120, 255))
    target = Image.new("L", part.size)
    target.putdata([64, 255, 0])
    protect = Image.new("L", part.size, 0)

    result = extract_and_edge_repair(part, target, protect)

    assert result.getpixel((0, 0)) == (40, 80, 120, 64)
    assert result.getpixel((1, 0)) == part.getpixel((1, 0))
    assert result.getpixel((2, 0)) == part.getpixel((2, 0))


def test_edge_repair_extracts_source_pixels_for_extension_coverage() -> None:
    source = Image.new("RGBA", (3, 1), (40, 80, 120, 255))
    part = Image.new("RGBA", source.size, (0, 0, 0, 0))
    part.putpixel((1, 0), source.getpixel((1, 0)))
    target = _mask(source.size, {(1, 0)})
    protect = target.copy()
    edge_extension = _mask(source.size, {(0, 0), (2, 0)})

    result = extract_and_edge_repair(
        part,
        target,
        protect,
        edge_extension_mask=edge_extension,
        source_image=source,
    )

    assert result.getpixel((0, 0)) == source.getpixel((0, 0))
    assert result.getpixel((2, 0)) == source.getpixel((2, 0))
    assert count_overlap_deficit(
        result,
        target,
        2,
        edge_extension_mask=edge_extension,
    ) == 0


def test_transparency_fill_changes_only_the_inpaint_bounding_box() -> None:
    size = (32, 32)
    part = Image.new("RGBA", size, (0, 0, 0, 0))
    part.putpixel((15, 16), (40, 80, 120, 255))
    part.putpixel((0, 0), (5, 10, 15, 255))
    inpaint = Image.new("L", size, 0)
    for y in range(14, 19):
        for x in range(16, 21):
            inpaint.putpixel((x, y), 255)
    protect = Image.new("L", size, 0)

    result = transparency_fill(part, inpaint, protect, iterations=1)

    assert result.getpixel((16, 16))[3] == 255
    assert result.crop((0, 0, 32, 14)).tobytes() == part.crop((0, 0, 32, 14)).tobytes()
    assert result.crop((0, 19, 32, 32)).tobytes() == part.crop((0, 19, 32, 32)).tobytes()
    assert result.crop((0, 14, 16, 19)).tobytes() == part.crop((0, 14, 16, 19)).tobytes()
    assert result.crop((21, 14, 32, 19)).tobytes() == part.crop((21, 14, 32, 19)).tobytes()


def test_motion_stress_preview_moves_part_without_changing_canvas_origin() -> None:
    part = Image.new("RGBA", (5, 3), (0, 0, 0, 0))
    part.putpixel((2, 1), (255, 0, 0, 255))

    shifted = shift_part(part, 1, 0)
    preview = create_part_motion_debug_preview(part, 1)

    assert shifted.size == part.size
    assert shifted.getpixel((3, 1)) == (255, 0, 0, 255)
    assert shifted.getpixel((2, 1))[3] == 0
    assert preview.size == (15, 3)
    assert preview.getpixel((1, 1)) == (255, 0, 0, 255)
    assert preview.getpixel((7, 1)) == (255, 0, 0, 255)
    assert preview.getpixel((13, 1)) == (255, 0, 0, 255)


def test_motion_stress_preview_recomposes_all_import_parts() -> None:
    bottom = Image.new("RGBA", (5, 3), (200, 0, 0, 255))
    moving = Image.new("RGBA", bottom.size, (0, 0, 0, 0))
    moving.putpixel((2, 1), (0, 0, 255, 255))

    preview = create_motion_stress_preview(
        bottom.size,
        [(10, "bottom", bottom), (20, "moving", moving)],
        "moving",
        1,
    )

    assert preview.size == (15, 3)
    assert preview.getpixel((2, 1)) == (200, 0, 0, 255)
    assert preview.getpixel((1, 1)) == (0, 0, 255, 255)
    assert preview.getpixel((7, 1)) == (0, 0, 255, 255)
    assert preview.getpixel((13, 1)) == (0, 0, 255, 255)


def test_refinement_plan_requeues_failed_part_only() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    original = deepcopy(queue)

    plan = build_refinement_plan(queue, quality, quality_ref="quality.yaml")
    refined = apply_refinement_plan(queue, plan)

    assert [job["layer_id"] for job in plan["jobs"]] == ["eye_white_L"]
    before = {asset["layer_id"]: asset for asset in original["assets"]}
    after = {asset["layer_id"]: asset for asset in refined["assets"]}
    assert after["eye_white_L"]["generation_method"] == "extract"
    assert after["eye_white_L"]["refinement_attempts"] == 1
    assert after["eye_white_R"] == before["eye_white_R"]
    assert plan["jobs"][0]["requested_action"] == "reset_from_source_and_reextract"


def test_refinement_uses_source_preserving_generation_priority() -> None:
    assert next_generation_method("extract") == "extract_and_edge_repair"
    assert next_generation_method("extract_and_edge_repair") == "transparency_fill"
    assert next_generation_method("transparency_fill") == "inpaint"
    assert next_generation_method("inpaint") == "redraw"
    assert next_generation_method("redraw") == "redraw"
    assert select_generation_method("extract", ["white_halo_px"]) == "extract_and_edge_repair"
    assert select_generation_method("extract", ["transparent_hole_px"]) == "transparency_fill"


def _set_failed_check(quality: dict[str, Any], layer_id: str, check: str) -> None:
    metric_for_check = {
        "white_halo_px": "white_halo_px",
        "transparent_hole_px": "transparent_hole_px",
        "overlap_deficit_px": "overlap_deficit_px",
        "preserve_region_difference_score": "preserve_region_difference_score",
        "allowed_change_region_difference_score": (
            "allowed_change_region_difference_score"
        ),
        "visual_reconstruction_difference_score": (
            "visual_reconstruction_difference_score"
        ),
    }
    for part in quality["parts"]:
        for metric in part["metrics"]:
            part["metrics"][metric] = 0
        part["quality_status"] = "fail" if part["layer_id"] == layer_id else "pass"
        part["failed_checks"] = [check] if part["layer_id"] == layer_id else []
        if part["layer_id"] == layer_id:
            part["metrics"][metric_for_check[check]] = 1
    quality["summary"]["failed_parts"] = 1
    quality["summary"]["result"] = "fail"


def test_refinement_extract_to_inpaint_sets_invariants_and_job_operations() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    asset = next(item for item in queue["assets"] if item["layer_id"] == "eye_white_L")
    asset["generation_method"] = "extract"
    asset["inferred"] = False
    asset["review_required"] = False

    for failed_check, expected_method in (
        ("white_halo_px", "extract_and_edge_repair"),
        ("transparent_hole_px", "transparency_fill"),
        ("white_halo_px", "inpaint"),
    ):
        _set_failed_check(quality, "eye_white_L", failed_check)
        plan = build_refinement_plan(queue, quality, quality_ref="quality.yaml")
        queue = apply_refinement_plan(queue, plan)
        asset = next(item for item in queue["assets"] if item["layer_id"] == "eye_white_L")
        assert asset["generation_method"] == expected_method

    assert asset["inferred"] is True
    assert asset["review_required"] is True
    eyes_job = next(job for job in queue["jobs"] if job["id"] == "eyes")
    assert "inpaint" in eyes_job["operations"]
    assert validate_asset_manifest(derive_asset_manifest(queue)).errors == ()


def test_refinement_inpaint_to_redraw_requires_review() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    asset = next(item for item in queue["assets"] if item["layer_id"] == "eye_white_L")
    asset["generation_method"] = "inpaint"
    asset["inferred"] = True
    asset["review_required"] = True
    _set_failed_check(quality, "eye_white_L", "white_halo_px")

    refined = apply_refinement_plan(
        queue,
        build_refinement_plan(queue, quality, quality_ref="quality.yaml"),
    )
    asset = next(item for item in refined["assets"] if item["layer_id"] == "eye_white_L")
    eyes_job = next(job for job in refined["jobs"] if job["id"] == "eyes")

    assert asset["generation_method"] == "redraw"
    assert asset["review_required"] is True
    assert "redraw" in eyes_job["operations"]
    assert validate_asset_manifest(derive_asset_manifest(refined)).errors == ()


def test_atomic_png_publish_replaces_without_temp_residue(tmp_path: Path) -> None:
    output = tmp_path / "part.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(output)

    atomic_save_png(Image.new("RGBA", (2, 2), (0, 0, 255, 255)), output, force=True)

    with Image.open(output) as result:
        assert result.getpixel((0, 0)) == (0, 0, 255, 255)
    assert list(tmp_path.glob(".*.tmp")) == []


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


def test_refinement_coverage_ignores_non_import_assets() -> None:
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    guide = deepcopy(queue["assets"][0])
    guide["layer_id"] = "guide_only"
    guide["layer_name"] = "guide_only"
    guide["layer_path"] = "Guides/guide_only"
    guide["draw_order"] = 999
    guide["include_in_import"] = False
    queue["assets"].append(guide)

    plan = build_refinement_plan(queue, quality, quality_ref="quality.yaml")

    assert "guide_only" not in {job["layer_id"] for job in plan["jobs"]}


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
    assert mask_manifest["schema_version"] == 2
    assert asset_quality["schema_version"] == 2
    for path in (
        Path("schemas/mask_manifest.schema.yaml"),
        Path("schemas/asset_quality.schema.yaml"),
    ):
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["$schema"].startswith("https://json-schema.org/")
        assert data["properties"]["schema_version"]["const"] == 2


def test_mask_manifest_requires_queue_provenance() -> None:
    manifest = load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))
    manifest.pop("derived_from")

    issues = validate_mask_manifest(manifest)

    assert any(issue.path == "derived_from" for issue in issues)


def test_mask_manifest_requires_edge_extension_mask() -> None:
    manifest = load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))
    manifest["parts"][0].pop("edge_extension_mask")

    issues = validate_mask_manifest(manifest)

    assert any(issue.path.endswith("edge_extension_mask") for issue in issues)


def test_mask_manifest_rejects_shared_edge_extension_and_inpaint_path() -> None:
    manifest = load_yaml_mapping(Path("examples/mask_manifest.sample.yaml"))
    manifest["parts"][0]["edge_extension_mask"] = manifest["parts"][0]["inpaint_mask"]

    issues = validate_mask_manifest(manifest)

    assert any("must differ from inpaint_mask" in issue.message for issue in issues)


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
    manifest["parts"][0]["generation_method"] = "transparency_fill"
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


def test_visual_reconstruction_difference_uses_configurable_threshold() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    for part in quality["parts"]:
        part["quality_status"] = "pass"
        part["failed_checks"] = []
        for key in part["metrics"]:
            part["metrics"][key] = 0
    quality["summary"]["failed_parts"] = 0
    quality["summary"]["visual_reconstruction_difference_score"] = 0.1
    quality["summary"]["result"] = "fail"

    assert validate_asset_quality(quality) == []

    quality["thresholds"]["max_visual_reconstruction_difference_score"] = 0.2
    quality["summary"]["result"] = "pass"
    assert validate_asset_quality(quality) == []


def test_quality_status_and_failed_checks_must_match_metrics() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["parts"][0]["quality_status"] = "pass"
    quality["parts"][0]["failed_checks"].remove("preserve_region_difference_score")

    issues = validate_asset_quality(quality)

    assert any("must exactly match" in issue.message for issue in issues)
    assert any("must equal fail" in issue.message for issue in issues)


def test_preserve_region_threshold_must_remain_zero() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["thresholds"]["max_preserve_region_difference_score"] = 0.01

    issues = validate_asset_quality(quality)

    assert any(
        issue.path == "thresholds.max_preserve_region_difference_score"
        and "must equal 0" in issue.message
        for issue in issues
    )


def test_legacy_global_threshold_field_is_rejected() -> None:
    quality = load_yaml_mapping(Path("examples/asset_quality.sample.yaml"))
    quality["thresholds"]["max_global_difference_score"] = 0.0

    issues = validate_asset_quality(quality)

    assert any(
        issue.path == "thresholds" and "max_global_difference_score" in issue.message
        for issue in issues
    )
