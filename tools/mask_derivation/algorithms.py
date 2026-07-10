from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

FINE_ROLE_TOKENS = ("eye", "eyelash", "brow", "hair_strand", "hair_tuft", "mouth_line")
FACE_ROLES = {"face", "face_base", "face_hidden_fill", "head", "skin_face"}
FRONT_HAIR_ROLES = {"front_hair", "hair_front", "bangs", "fringe"}


@dataclass(frozen=True)
class MaskMetrics:
    soft_coverage: float
    binary_area: int
    coverage_ratio: float


def _validate_radius(radius: int) -> None:
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be a non-negative integer")


def binary(mask: Image.Image, threshold: int = 1) -> Image.Image:
    if not 1 <= threshold <= 255:
        raise ValueError("threshold must be between 1 and 255")
    return mask.convert("L").point(lambda value: 255 if value >= threshold else 0, mode="L")


def pixel_count(mask: Image.Image, threshold: int = 1) -> int:
    histogram = binary(mask, threshold).histogram()
    return int(histogram[255])


def soft_coverage(mask: Image.Image) -> float:
    histogram = mask.convert("L").histogram()
    return sum(value * count for value, count in enumerate(histogram)) / 255.0


def metrics(mask: Image.Image, *, target_area: int) -> MaskMetrics:
    area = pixel_count(mask)
    return MaskMetrics(
        soft_coverage=round(soft_coverage(mask), 6),
        binary_area=area,
        coverage_ratio=round(area / target_area, 6) if target_area else 0.0,
    )


def _morphology(mask: Image.Image, radius: int, *, operation: str) -> Image.Image:
    _validate_radius(radius)
    grayscale = mask.convert("L")
    if radius == 0:
        return grayscale.copy()
    size = radius * 2 + 1
    if operation == "erode":
        return grayscale.filter(ImageFilter.MinFilter(size))
    if operation == "dilate":
        return grayscale.filter(ImageFilter.MaxFilter(size))
    raise ValueError(f"unsupported morphology operation: {operation}")


def dilate(mask: Image.Image, radius: int) -> Image.Image:
    return _morphology(mask, radius, operation="dilate")


def erode(mask: Image.Image, radius: int) -> Image.Image:
    return _morphology(mask, radius, operation="erode")


def mask_intersection(*masks: Image.Image) -> Image.Image:
    if not masks:
        raise ValueError("at least one mask is required")
    result = masks[0].convert("L")
    for mask in masks[1:]:
        if mask.size != result.size:
            raise ValueError("mask canvas mismatch")
        result = ImageChops.multiply(result, mask.convert("L"))
    return result


def mask_union(*masks: Image.Image) -> Image.Image:
    if not masks:
        raise ValueError("at least one mask is required")
    result = masks[0].convert("L")
    for mask in masks[1:]:
        if mask.size != result.size:
            raise ValueError("mask canvas mismatch")
        result = ImageChops.lighter(result, mask.convert("L"))
    return result


def derive_protect_mask(
    target_mask: Image.Image,
    source_alpha: Image.Image,
    *,
    radius_px: int = 2,
    role: str = "",
    min_area_px: int = 1,
) -> tuple[Image.Image, dict[str, Any]]:
    """Erode conservatively while retaining soft values and at least one target pixel."""
    _validate_radius(radius_px)
    if target_mask.size != source_alpha.size:
        raise ValueError("target mask and source alpha canvas mismatch")
    if min_area_px < 1:
        raise ValueError("min_area_px must be positive")
    normalized_role = role.lower().replace("-", "_")
    requested_radius = radius_px
    warnings: list[str] = []
    if any(token in normalized_role for token in FINE_ROLE_TOKENS) and radius_px > 1:
        radius_px = 1
        warnings.append("fine_part_radius_limited")
    target_area = pixel_count(target_mask)
    if target_area == 0:
        raise ValueError("target mask is empty")
    required_area = min(target_area, min_area_px)
    used_radius = radius_px
    while used_radius > 0:
        candidate = mask_intersection(erode(target_mask, used_radius), source_alpha)
        if pixel_count(candidate) >= required_area:
            break
        used_radius -= 1
    else:
        candidate = mask_intersection(target_mask, source_alpha)
    if pixel_count(candidate) == 0:
        candidate = target_mask.convert("L").copy()
        warnings.append("source_alpha_would_remove_entire_protect_mask")
    if used_radius < radius_px:
        warnings.append("erosion_radius_reduced_to_preserve_thin_part")
    data = metrics(candidate, target_area=target_area)
    return candidate, {
        "method": "target_erosion",
        "parameters": {
            "requested_radius_px": requested_radius,
            "radius_px": used_radius,
            "min_area_px": min_area_px,
            "source_alpha_gated": True,
        },
        "soft_coverage": data.soft_coverage,
        "area_px": data.binary_area,
        "coverage_ratio": data.coverage_ratio,
        "warnings": warnings,
        "derivation_reasons": [
            "conservative_target_interior",
            "source_alpha_supported",
            "soft_grayscale_preserved",
        ],
    }


def derive_edge_extension_mask(
    target_mask: Image.Image,
    source_alpha: Image.Image,
    *,
    radius_px: int = 2,
    exclusion_masks: Sequence[Image.Image] = (),
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Build a soft outer ring and report the part clipped by source alpha/exclusions."""
    _validate_radius(radius_px)
    if radius_px == 0:
        empty = Image.new("L", target_mask.size, 0)
        return (
            empty,
            empty.copy(),
            {
                "method": "target_dilation_ring",
                "parameters": {"radius_px": 0, "source_alpha_gated": True},
                "soft_coverage": 0.0,
                "area_px": 0,
                "coverage_ratio": 0.0,
                "warnings": ["zero_radius_produces_empty_edge_extension"],
                "derivation_reasons": ["configured_zero_radius"],
            },
        )
    if target_mask.size != source_alpha.size:
        raise ValueError("target mask and source alpha canvas mismatch")
    raw_ring = ImageChops.subtract(dilate(target_mask, radius_px), target_mask.convert("L"))
    allowed = source_alpha.convert("L")
    for mask in exclusion_masks:
        if mask.size != target_mask.size:
            raise ValueError("edge exclusion mask canvas mismatch")
        allowed = mask_intersection(allowed, ImageOps.invert(binary(mask)))
    candidate = mask_intersection(raw_ring, allowed)
    clipped = ImageChops.subtract(raw_ring, candidate)
    target_area = pixel_count(target_mask)
    data = metrics(candidate, target_area=target_area)
    warnings: list[str] = []
    if pixel_count(clipped):
        warnings.append("ring_clipped_by_source_alpha_or_exclusion")
    return (
        candidate,
        clipped,
        {
            "method": "target_dilation_ring",
            "parameters": {
                "radius_px": radius_px,
                "source_alpha_gated": True,
                "exclusion_count": len(exclusion_masks),
            },
            "soft_coverage": data.soft_coverage,
            "area_px": data.binary_area,
            "coverage_ratio": data.coverage_ratio,
            "warnings": warnings,
            "derivation_reasons": [
                "motion_exposure_margin",
                "source_alpha_supported",
                "soft_grayscale_preserved",
            ],
        },
    )


def _expected_region_box(
    expected_region: object,
    size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(expected_region, Mapping):
        return None
    keys = ("x_min", "y_min", "x_max", "y_max")
    values: list[float] = []
    for key in keys:
        value = expected_region.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    x_min, y_min, x_max, y_max = values
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        return None
    width, height = size
    return (
        round(x_min * width),
        round(y_min * height),
        max(round(x_max * width) - 1, round(x_min * width)),
        max(round(y_max * height) - 1, round(y_min * height)),
    )


def _estimated_face_shape(
    target_mask: Image.Image,
    *,
    expected_region: object = None,
) -> Image.Image | None:
    target_binary = binary(target_mask)
    bbox = target_binary.getbbox()
    if bbox is None:
        return None
    expected_box = _expected_region_box(expected_region, target_mask.size)
    left, top, right_exclusive, bottom_exclusive = bbox
    if expected_box is None:
        visible_width = right_exclusive - left
        visible_height = bottom_exclusive - top
        if visible_width < 3 or visible_height < 2:
            return None
        estimated_height = max(visible_height, round(visible_width * 1.15))
        bottom = bottom_exclusive - 1
        top = max(0, bottom - estimated_height + 1)
        box = (left, top, right_exclusive - 1, bottom)
    else:
        box = expected_box
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    ellipse = Image.new("L", target_mask.size, 0)
    ImageDraw.Draw(ellipse).ellipse(box, fill=255)
    # Mirroring strengthens a conservative bilateral estimate without replacing soft target edges.
    center = (box[0] + box[2]) / 2.0
    mirrored = Image.new("L", target_mask.size, 0)
    source_pixels = target_mask.convert("L").load()
    mirrored_pixels = mirrored.load()
    width, height = target_mask.size
    assert source_pixels is not None and mirrored_pixels is not None
    for y in range(height):
        for x in range(width):
            mirror_x = round(2 * center - x)
            if 0 <= mirror_x < width:
                mirrored_pixels[mirror_x, y] = source_pixels[x, y]
    if expected_box is None:
        return mask_intersection(ellipse, mask_union(target_mask, mirrored))
    return ellipse


def derive_forehead_inpaint_mask(
    target_mask: Image.Image,
    protect_mask: Image.Image,
    occluder_masks: Sequence[Image.Image],
    *,
    role: str,
    expected_region: object = None,
    occluder_expand_px: int = 1,
) -> tuple[Image.Image | None, dict[str, Any]]:
    normalized_role = role.lower().replace("-", "_").replace(" ", "_")
    if normalized_role not in FACE_ROLES:
        return None, {
            "status": "unavailable",
            "reason": "complete_shape_not_estimable",
            "derivation_reasons": ["role_not_supported_by_initial_inpaint_policy"],
        }
    if not occluder_masks:
        return None, {
            "status": "unavailable",
            "reason": "complete_shape_not_estimable",
            "derivation_reasons": ["front_hair_occluder_not_found"],
        }
    estimated = _estimated_face_shape(target_mask, expected_region=expected_region)
    if estimated is None:
        return None, {
            "status": "unavailable",
            "reason": "complete_shape_not_estimable",
            "derivation_reasons": ["visible_face_geometry_insufficient"],
        }
    occluder = mask_union(*(dilate(mask, occluder_expand_px) for mask in occluder_masks))
    missing = mask_intersection(estimated, ImageOps.invert(binary(target_mask)))
    candidate = mask_intersection(
        missing,
        ImageOps.invert(binary(protect_mask)),
        occluder,
    )
    if pixel_count(candidate) == 0:
        return None, {
            "status": "unavailable",
            "reason": "complete_shape_not_estimable",
            "derivation_reasons": ["estimated_face_and_front_hair_do_not_define_hidden_forehead"],
        }
    target_area = pixel_count(target_mask)
    data = metrics(candidate, target_area=target_area)
    return candidate, {
        "status": "candidate",
        "method": "bilateral_face_shape_under_front_hair",
        "parameters": {
            "occluder_expand_px": occluder_expand_px,
            "expected_region_used": _expected_region_box(expected_region, target_mask.size)
            is not None,
        },
        "soft_coverage": data.soft_coverage,
        "area_px": data.binary_area,
        "coverage_ratio": data.coverage_ratio,
        "warnings": ["human_face_shape_review_required"],
        "derivation_reasons": [
            "bilateral_face_shape_estimate",
            "front_hair_occluder",
            "visible_target_subtracted",
            "protect_mask_subtracted",
        ],
    }


def _small_components(
    mask: Image.Image,
    *,
    threshold: int,
    maximum_size: int,
) -> list[list[tuple[int, int]]]:
    """Return only small 8-connected components using O(canvas) byte storage."""
    if maximum_size < 1:
        return []
    foreground = bytearray(binary(mask, threshold).tobytes())
    width, height = mask.size
    small: list[list[tuple[int, int]]] = []
    for seed in range(len(foreground)):
        if foreground[seed] == 0:
            continue
        queue: deque[int] = deque([seed])
        component_size = 0
        retained: list[tuple[int, int]] = []
        while queue:
            point = queue.popleft()
            if foreground[point] == 0:
                continue
            y, x = divmod(point, width)
            left = x
            while left > 0 and foreground[y * width + left - 1]:
                left -= 1
            right = x
            while right + 1 < width and foreground[y * width + right + 1]:
                right += 1
            for current_x in range(left, right + 1):
                foreground[y * width + current_x] = 0
                component_size += 1
                if component_size <= maximum_size:
                    retained.append((current_x, y))
            for neighbor_y in (y - 1, y + 1):
                if not 0 <= neighbor_y < height:
                    continue
                scan_left = max(0, left - 1)
                scan_right = min(width - 1, right + 1)
                current_x = scan_left
                while current_x <= scan_right:
                    neighbor = neighbor_y * width + current_x
                    if foreground[neighbor]:
                        queue.append(neighbor)
                        while (
                            current_x <= scan_right and foreground[neighbor_y * width + current_x]
                        ):
                            current_x += 1
                    current_x += 1
        if component_size <= maximum_size:
            small.append(retained)
    return small


def _overlap(a: Image.Image, b: Image.Image) -> tuple[int, Image.Image]:
    overlap = mask_intersection(binary(a), binary(b))
    return pixel_count(overlap), overlap


def _conflict(
    conflict_type: str,
    *,
    area_px: int,
    severity: str = "review",
    layers: Sequence[str] = (),
    candidate_types: Sequence[str] = (),
    reason: str,
) -> dict[str, Any]:
    return {
        "type": conflict_type,
        "severity": severity,
        "area_px": area_px,
        "layers": list(layers),
        "candidate_types": list(candidate_types),
        "reason": reason,
    }


def detect_candidate_conflicts(
    target_mask: Image.Image,
    candidates: Mapping[str, Image.Image | None],
    *,
    source_alpha: Image.Image | None = None,
    max_area_ratio: float = 0.75,
    min_island_area_px: int = 2,
    layer_id: str = "",
) -> tuple[list[dict[str, Any]], Image.Image]:
    """Detect conflicts without modifying any candidate pixels."""
    if max_area_ratio <= 0:
        raise ValueError("max_area_ratio must be positive")
    target_area = pixel_count(target_mask)
    if target_area == 0:
        raise ValueError("target mask is empty")
    conflict_mask = Image.new("L", target_mask.size, 0)
    conflicts: list[dict[str, Any]] = []
    pairs = (
        ("protect", "inpaint"),
        ("protect", "edge_extension"),
        ("edge_extension", "inpaint"),
    )
    for first, second in pairs:
        first_mask = candidates.get(first)
        second_mask = candidates.get(second)
        if first_mask is None or second_mask is None:
            continue
        area, overlap = _overlap(first_mask, second_mask)
        if area:
            conflicts.append(
                _conflict(
                    f"{first}_{second}_overlap",
                    area_px=area,
                    layers=[layer_id] if layer_id else [],
                    candidate_types=[first, second],
                    reason="candidate regions overlap; human review is required",
                )
            )
            conflict_mask = mask_union(conflict_mask, overlap)
    target_neighborhood = dilate(target_mask, 1)
    for candidate_type, candidate in candidates.items():
        if candidate is None:
            continue
        area = pixel_count(candidate)
        if area > target_area * max_area_ratio:
            conflicts.append(
                _conflict(
                    "area_ratio_exceeded",
                    area_px=area,
                    severity="reject",
                    layers=[layer_id] if layer_id else [],
                    candidate_types=[candidate_type],
                    reason=f"candidate area exceeds {max_area_ratio:.3f} of target area",
                )
            )
            conflict_mask = mask_union(conflict_mask, candidate)
        small = _small_components(
            candidate,
            threshold=1,
            maximum_size=min_island_area_px - 1,
        )
        if small:
            small_mask = Image.new("L", candidate.size, 0)
            small_pixels = small_mask.load()
            assert small_pixels is not None
            for component in small:
                for x, y in component:
                    small_pixels[x, y] = 255
            conflicts.append(
                _conflict(
                    "thin_isolated_region",
                    area_px=sum(len(component) for component in small),
                    layers=[layer_id] if layer_id else [],
                    candidate_types=[candidate_type],
                    reason="candidate contains a component below the minimum island area",
                )
            )
            conflict_mask = mask_union(conflict_mask, small_mask)
        if candidate_type in {"edge_extension", "inpaint"}:
            adjacency = mask_intersection(binary(candidate), binary(target_neighborhood))
            if pixel_count(adjacency) == 0 and area:
                conflicts.append(
                    _conflict(
                        "detached_from_target",
                        area_px=area,
                        severity="reject",
                        layers=[layer_id] if layer_id else [],
                        candidate_types=[candidate_type],
                        reason="candidate is a detached island with no target adjacency",
                    )
                )
                conflict_mask = mask_union(conflict_mask, candidate)
        if source_alpha is not None and candidate_type == "edge_extension":
            outside = mask_intersection(candidate, ImageOps.invert(binary(source_alpha)))
            outside_area = pixel_count(outside)
            if outside_area:
                conflicts.append(
                    _conflict(
                        "source_alpha_abnormal_extension",
                        area_px=outside_area,
                        severity="reject",
                        layers=[layer_id] if layer_id else [],
                        candidate_types=[candidate_type],
                        reason="edge extension reaches pixels outside source alpha",
                    )
                )
                conflict_mask = mask_union(conflict_mask, outside)
    return conflicts, conflict_mask


def symmetry_warning(
    left: Image.Image,
    right: Image.Image,
    *,
    tolerance: float = 0.35,
) -> dict[str, Any] | None:
    if left.size != right.size:
        raise ValueError("symmetry masks must share a canvas")
    left_area = pixel_count(left)
    right_area = pixel_count(right)
    largest = max(left_area, right_area)
    if largest == 0:
        return None
    ratio = abs(left_area - right_area) / largest
    mirrored_left = ImageOps.mirror(binary(left))
    union_area = pixel_count(mask_union(mirrored_left, binary(right)))
    overlap_area, _ = _overlap(mirrored_left, right)
    shape_difference = 1.0 - (overlap_area / union_area if union_area else 1.0)
    if max(ratio, shape_difference) <= tolerance:
        return None
    return _conflict(
        "left_right_asymmetry",
        area_px=abs(left_area - right_area),
        reason="paired left/right candidate masks differ beyond tolerance",
    ) | {"area_difference_ratio": round(ratio, 6), "shape_difference": round(shape_difference, 6)}
