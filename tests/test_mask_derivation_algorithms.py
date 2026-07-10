from __future__ import annotations

from PIL import Image, ImageDraw

from tools.mask_derivation.algorithms import (
    derive_edge_extension_mask,
    derive_forehead_inpaint_mask,
    derive_protect_mask,
    detect_candidate_conflicts,
    pixel_count,
    symmetry_warning,
)


def _rectangle(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    value: int = 255,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=value)
    return mask


def test_protect_erosion_preserves_soft_grayscale_and_configurable_radius() -> None:
    target = _rectangle((9, 9), (1, 1, 7, 7), 160)
    target.putpixel((4, 4), 91)
    alpha = Image.new("L", target.size, 255)

    no_erosion, zero_details = derive_protect_mask(target, alpha, radius_px=0)
    eroded, details = derive_protect_mask(target, alpha, radius_px=2)

    assert no_erosion.tobytes() == target.tobytes()
    assert zero_details["parameters"]["radius_px"] == 0
    assert eroded.size == target.size
    assert eroded.getpixel((1, 1)) == 0
    assert eroded.histogram()[91] > 0
    assert details["parameters"]["radius_px"] == 2


def test_thin_protect_mask_never_disappears_and_fine_role_limits_radius() -> None:
    target = Image.new("L", (7, 7), 0)
    target.putpixel((3, 3), 180)
    alpha = Image.new("L", target.size, 255)

    result, details = derive_protect_mask(
        target,
        alpha,
        radius_px=4,
        role="eyelash",
        min_area_px=1,
    )

    assert pixel_count(result) == 1
    assert details["parameters"]["radius_px"] == 0
    assert "fine_part_radius_limited" in details["warnings"]
    assert "erosion_radius_reduced_to_preserve_thin_part" in details["warnings"]


def test_edge_extension_is_soft_ring_clipped_to_canvas_and_source_alpha() -> None:
    target = _rectangle((7, 7), (2, 2, 4, 4), 128)
    alpha = Image.new("L", target.size, 255)
    alpha.putpixel((1, 3), 0)

    edge, clipped, details = derive_edge_extension_mask(target, alpha, radius_px=1)

    assert edge.size == target.size
    assert edge.getpixel((3, 3)) == 0
    assert edge.getpixel((3, 1)) == 128
    assert edge.getpixel((1, 3)) == 0
    assert clipped.getpixel((1, 3)) == 128
    assert details["parameters"]["radius_px"] == 1


def test_conflict_detection_reports_all_pair_overlaps_islands_detachment_and_area_limit() -> None:
    target = _rectangle((12, 8), (1, 2, 4, 5))
    protect = _rectangle((12, 8), (2, 2, 3, 4))
    edge = _rectangle((12, 8), (3, 3, 8, 5))
    inpaint = _rectangle((12, 8), (2, 4, 7, 6))
    inpaint.putpixel((11, 0), 255)

    conflicts, conflict_mask = detect_candidate_conflicts(
        target,
        {"protect": protect, "edge_extension": edge, "inpaint": inpaint},
        max_area_ratio=0.25,
        min_island_area_px=2,
        layer_id="face",
    )

    kinds = {conflict["type"] for conflict in conflicts}
    assert "protect_inpaint_overlap" in kinds
    assert "protect_edge_extension_overlap" in kinds
    assert "edge_extension_inpaint_overlap" in kinds
    assert "area_ratio_exceeded" in kinds
    assert "thin_isolated_region" in kinds
    assert pixel_count(conflict_mask) > 0

    detached = _rectangle((12, 8), (9, 1, 10, 2))
    detached_conflicts, _ = detect_candidate_conflicts(
        target,
        {"edge_extension": detached},
        max_area_ratio=2,
    )
    assert any(value["type"] == "detached_from_target" for value in detached_conflicts)


def test_forehead_candidate_uses_expected_face_shape_and_front_hair_occluder() -> None:
    target = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(target).ellipse((5, 6, 15, 18), fill=220)
    ImageDraw.Draw(target).rectangle((4, 0, 16, 8), fill=0)
    protect, _ = derive_protect_mask(target, Image.new("L", target.size, 255), radius_px=1)
    hair = _rectangle(target.size, (4, 2, 16, 8), 200)

    candidate, details = derive_forehead_inpaint_mask(
        target,
        protect,
        [hair],
        role="face",
        expected_region={"x_min": 0.2, "y_min": 0.1, "x_max": 0.8, "y_max": 0.95},
    )

    assert candidate is not None
    assert candidate.size == target.size
    assert pixel_count(candidate) > 0
    assert candidate.getpixel((10, 5)) > 0
    assert details["method"] == "bilateral_face_shape_under_front_hair"


def test_complete_shape_unavailable_and_left_right_asymmetry_warning() -> None:
    target = _rectangle((12, 8), (1, 2, 3, 4))
    candidate, details = derive_forehead_inpaint_mask(
        target,
        target,
        [],
        role="eye_white",
    )
    assert candidate is None
    assert details == {
        "status": "unavailable",
        "reason": "complete_shape_not_estimable",
        "derivation_reasons": ["role_not_supported_by_initial_inpaint_policy"],
    }

    left = _rectangle((12, 8), (1, 2, 2, 3))
    right = _rectangle((12, 8), (8, 1, 11, 5))
    warning = symmetry_warning(left, right, tolerance=0.1)
    assert warning is not None
    assert warning["type"] == "left_right_asymmetry"
