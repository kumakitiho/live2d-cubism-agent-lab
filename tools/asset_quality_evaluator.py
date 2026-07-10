from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from tools.asset_pipeline_common import (
    atomic_save_png,
    import_parts,
    load_and_validate_mask_manifest,
    load_binary_mask,
    load_rgba,
    mask_manifest_protected_paths,
    require_manifest_canvas,
    require_output_suffix,
    resolve_inside_base,
    validate_asset_quality,
    write_yaml,
)

DEFAULT_QUALITY_THRESHOLDS: dict[str, int | float] = {
    "max_white_halo_px": 0,
    "max_transparent_hole_px": 0,
    "max_overlap_deficit_px": 0,
    "max_preserve_region_difference_score": 0.0,
    "max_edge_extension_difference_score": 0.01,
    "max_inpaint_outside_difference_score": 0.0,
    "max_edge_continuity_score": 0.1,
    "max_boundary_color_difference_score": 0.15,
    "max_visual_reconstruction_difference_score": 0.01,
}


def _binary_alpha(image: Image.Image, *, alpha_threshold: int = 1) -> Image.Image:
    if not 1 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be from 1 to 255")
    return image.convert("RGBA").getchannel("A").point(
        lambda value: 255 if value >= alpha_threshold else 0,
        mode="L",
    )


def _union_masks(*masks: Image.Image) -> Image.Image:
    if not masks:
        raise ValueError("at least one mask is required")
    result = masks[0].convert("L")
    for mask in masks[1:]:
        converted = mask.convert("L")
        if converted.size != result.size:
            raise ValueError("all masks must use the same canvas")
        result = ImageChops.lighter(result, converted)
    return result


def allowed_change_region_mask(
    edge_extension_mask: Image.Image,
    inpaint_mask: Image.Image,
    *,
    protect_mask: Image.Image | None = None,
) -> Image.Image:
    allowed = _union_masks(edge_extension_mask, inpaint_mask)
    if protect_mask is None:
        return allowed
    protect = protect_mask.convert("L")
    if protect.size != allowed.size:
        raise ValueError("allowed-change and protect masks must use the same canvas")
    return ImageChops.subtract(allowed, protect)


def desired_coverage_mask(
    target_mask: Image.Image,
    edge_extension_mask: Image.Image | None = None,
) -> Image.Image:
    target = target_mask.convert("L")
    if edge_extension_mask is None:
        return target
    extension = edge_extension_mask.convert("L")
    if target.size != extension.size:
        raise ValueError("target and edge-extension masks must use the same canvas")
    return ImageChops.lighter(target, extension)


def count_transparent_holes(part: Image.Image, target_mask: Image.Image) -> int:
    if part.size != target_mask.size:
        raise ValueError("part and target mask must use the same canvas")
    alpha = _binary_alpha(part)
    missing = ImageChops.subtract(target_mask.convert("L"), alpha)
    return sum(missing.histogram()[1:])


def count_overlap_deficit(
    part: Image.Image,
    target_mask: Image.Image,
    overlap_margin_px: int,
    *,
    edge_extension_mask: Image.Image | None = None,
) -> int:
    if part.size != target_mask.size:
        raise ValueError("part and target mask must use the same canvas")
    if overlap_margin_px < 0:
        raise ValueError("overlap_margin_px must be non-negative")
    expected = desired_coverage_mask(target_mask, edge_extension_mask)
    missing = ImageChops.subtract(expected, _binary_alpha(part))
    return sum(missing.histogram()[1:])


def count_white_halo(part: Image.Image, source: Image.Image) -> int:
    part_rgba = part.convert("RGBA")
    source_rgba = source.convert("RGBA")
    if part_rgba.size != source_rgba.size:
        raise ValueError("part and source must use the same canvas")
    alpha = _binary_alpha(part_rgba)
    eroded = alpha.filter(ImageFilter.MinFilter(3))
    boundary = ImageChops.subtract(alpha, eroded)

    def white_opaque_mask(image: Image.Image) -> Image.Image:
        red, green, blue, image_alpha = image.split()
        white = ImageChops.darker(
            ImageChops.darker(
                red.point(lambda value: 255 if value >= 245 else 0, mode="L"),
                green.point(lambda value: 255 if value >= 245 else 0, mode="L"),
            ),
            blue.point(lambda value: 255 if value >= 245 else 0, mode="L"),
        )
        opaque = image_alpha.point(lambda value: 255 if value else 0, mode="L")
        return ImageChops.darker(white, opaque)

    part_white = white_opaque_mask(part_rgba)
    source_not_white = ImageChops.invert(white_opaque_mask(source_rgba))
    halo = ImageChops.darker(ImageChops.darker(boundary, part_white), source_not_white)
    return sum(halo.histogram()[1:])


def _premultiply_rgba(image: Image.Image) -> Image.Image:
    red, green, blue, alpha = image.convert("RGBA").split()
    return Image.merge(
        "RGBA",
        (
            ImageChops.multiply(red, alpha),
            ImageChops.multiply(green, alpha),
            ImageChops.multiply(blue, alpha),
            alpha,
        ),
    )


def premultiplied_difference_image(
    reference: Image.Image,
    candidate: Image.Image,
    mask: Image.Image | None = None,
) -> Image.Image:
    reference_rgba = reference.convert("RGBA")
    candidate_rgba = candidate.convert("RGBA")
    if reference_rgba.size != candidate_rgba.size:
        raise ValueError("images must use the same canvas")

    difference = ImageChops.difference(
        _premultiply_rgba(reference_rgba),
        _premultiply_rgba(candidate_rgba),
    )
    if mask is not None:
        mask_l = mask.convert("L")
        if mask_l.size != reference_rgba.size:
            raise ValueError("difference mask must use the same canvas")
        difference = Image.merge(
            "RGBA",
            tuple(ImageChops.multiply(channel, mask_l) for channel in difference.split()),
        )
    return difference


def difference_score(
    reference: Image.Image,
    candidate: Image.Image,
    mask: Image.Image | None = None,
) -> float:
    difference = premultiplied_difference_image(reference, candidate, mask)
    pixel_count = reference.width * reference.height
    if mask is not None:
        pixel_count = sum(mask.convert("L").histogram()[1:])
    total = sum(
        value * count
        for channel in difference.split()
        for value, count in enumerate(channel.histogram())
    )
    maximum = pixel_count * 4 * 255
    return total / maximum if maximum else 0.0


def inpaint_outside_difference_score(
    part: Image.Image,
    source: Image.Image,
    target_mask: Image.Image,
    protect_mask: Image.Image,
    edge_extension_mask: Image.Image,
    inpaint_mask: Image.Image,
) -> float:
    target_and_protect = _union_masks(target_mask, protect_mask)
    allowed_generated_region = _union_masks(edge_extension_mask, inpaint_mask)
    source_guard = ImageChops.subtract(target_and_protect, allowed_generated_region)
    source_difference = difference_score(source, part, source_guard)

    declared_region = _union_masks(target_and_protect, allowed_generated_region)
    outside_region = ImageChops.invert(declared_region)
    transparent = Image.new("RGBA", part.size, (0, 0, 0, 0))
    leak_difference = difference_score(transparent, part, outside_region)
    return max(source_difference, leak_difference)


def _shift_for_neighbor(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Return an image where pixel (x, y) reads the original (x + dx, y + dy)."""
    width, height = image.size
    source_left = max(0, dx)
    source_top = max(0, dy)
    source_right = min(width, width + dx)
    source_bottom = min(height, height + dy)
    result = Image.new(image.mode, image.size, 0)
    if source_left >= source_right or source_top >= source_bottom:
        return result
    result.paste(
        image.crop((source_left, source_top, source_right, source_bottom)),
        (max(0, -dx), max(0, -dy)),
    )
    return result


def _masked_channel_difference_total(
    channel: Image.Image,
    shifted_channel: Image.Image,
    mask: Image.Image,
) -> int:
    difference = ImageChops.difference(channel, shifted_channel)
    masked = ImageChops.multiply(difference, mask)
    return sum(value * count for value, count in enumerate(masked.histogram()))


def inpaint_boundary_metrics(
    part: Image.Image,
    inpaint_mask: Image.Image,
    seam_support_mask: Image.Image,
) -> tuple[float, float]:
    inpaint = inpaint_mask.convert("L")
    seam_support = seam_support_mask.convert("L")
    if part.size != inpaint.size or seam_support.size != inpaint.size:
        raise ValueError("part and inpaint boundary masks must use the same canvas")
    bbox = inpaint.getbbox()
    if bbox is None:
        return 0.0, 0.0

    left, top, right, bottom = bbox
    region = (
        max(0, left - 1),
        max(0, top - 1),
        min(inpaint.width, right + 1),
        min(inpaint.height, bottom + 1),
    )
    inpaint = inpaint.crop(region)
    seam_support = seam_support.crop(region)
    part = part.crop(region)

    inner = ImageChops.subtract(inpaint, inpaint.filter(ImageFilter.MinFilter(3)))
    premultiplied = _premultiply_rgba(part)
    red, green, blue, alpha = premultiplied.split()
    pair_count = 0
    alpha_difference_total = 0
    color_difference_total = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted_support = _shift_for_neighbor(seam_support, dx, dy)
            pair_mask = ImageChops.darker(inner, shifted_support)
            current_pair_count = sum(pair_mask.histogram()[1:])
            if current_pair_count == 0:
                continue
            pair_count += current_pair_count
            alpha_difference_total += _masked_channel_difference_total(
                alpha,
                _shift_for_neighbor(alpha, dx, dy),
                pair_mask,
            )
            for channel in (red, green, blue):
                color_difference_total += _masked_channel_difference_total(
                    channel,
                    _shift_for_neighbor(channel, dx, dy),
                    pair_mask,
                )
    if pair_count == 0:
        return 0.0, 0.0
    edge_continuity_score = alpha_difference_total / (pair_count * 255)
    boundary_color_difference_score = color_difference_total / (pair_count * 3 * 255)
    return edge_continuity_score, boundary_color_difference_score


def foreground_reconstruction_mask(
    part_regions: list[Image.Image],
    reconstructed: Image.Image,
) -> Image.Image:
    if not part_regions:
        return _binary_alpha(reconstructed)
    foreground = _union_masks(*part_regions)
    if foreground.size != reconstructed.size:
        raise ValueError("foreground regions and reconstruction must use the same canvas")
    return ImageChops.lighter(foreground, _binary_alpha(reconstructed))


def _validated_thresholds(
    thresholds: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    result = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(DEFAULT_QUALITY_THRESHOLDS)
        if unknown:
            raise ValueError(f"unknown quality thresholds: {sorted(unknown)}")
        result.update(thresholds)
    for key in ("max_white_halo_px", "max_transparent_hole_px", "max_overlap_deficit_px"):
        value = result.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in (
        "max_preserve_region_difference_score",
        "max_edge_extension_difference_score",
        "max_inpaint_outside_difference_score",
        "max_edge_continuity_score",
        "max_boundary_color_difference_score",
        "max_visual_reconstruction_difference_score",
    ):
        value = result.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"{key} must be between 0 and 1")
    if float(result["max_preserve_region_difference_score"]) != 0.0:
        raise ValueError("max_preserve_region_difference_score must equal 0")
    if float(result["max_inpaint_outside_difference_score"]) != 0.0:
        raise ValueError("max_inpaint_outside_difference_score must equal 0")
    return result


def evaluate_part(
    part: Image.Image,
    source: Image.Image,
    target_mask: Image.Image,
    *,
    overlap_margin_px: int,
    protect_mask: Image.Image | None = None,
    edge_extension_mask: Image.Image | None = None,
    inpaint_mask: Image.Image | None = None,
    reconstructed: Image.Image | None = None,
    thresholds: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    config = _validated_thresholds(thresholds)
    empty = Image.new("L", target_mask.size, 0)
    protect = protect_mask or target_mask
    edge_extension = edge_extension_mask or empty
    inpaint = inpaint_mask or empty
    reconstructed_image = reconstructed or source
    expected_canvas = source.size
    canvases = {
        "part": part.size,
        "target_mask": target_mask.size,
        "protect_mask": protect.size,
        "edge_extension_mask": edge_extension.size,
        "inpaint_mask": inpaint.size,
        "reconstructed": reconstructed_image.size,
    }
    mismatched = {name: size for name, size in canvases.items() if size != expected_canvas}
    if mismatched:
        raise ValueError(
            f"canvas/origin mismatch: expected={expected_canvas}, mismatched={mismatched}"
        )
    allowed_change = allowed_change_region_mask(
        edge_extension,
        inpaint,
        protect_mask=protect,
    )
    attribution_region = _union_masks(target_mask, edge_extension, inpaint)
    candidate_inpaint = ImageChops.darker(inpaint, _binary_alpha(part))
    seam_support = ImageChops.subtract(
        _union_masks(target_mask, protect, edge_extension),
        inpaint,
    )
    edge_continuity_score, boundary_color_difference_score = inpaint_boundary_metrics(
        part,
        candidate_inpaint,
        seam_support,
    )
    metrics: dict[str, int | float] = {
        "white_halo_px": count_white_halo(part, source),
        "transparent_hole_px": count_transparent_holes(part, target_mask),
        "overlap_deficit_px": count_overlap_deficit(
            part,
            target_mask,
            overlap_margin_px,
            edge_extension_mask=edge_extension,
        ),
        "preserve_region_difference_score": difference_score(source, part, protect),
        "edge_extension_difference_score": difference_score(
            source,
            part,
            edge_extension,
        ),
        "inpaint_region_source_difference_score": difference_score(
            source,
            part,
            inpaint,
        ),
        "inpaint_outside_difference_score": inpaint_outside_difference_score(
            part,
            source,
            target_mask,
            protect,
            edge_extension,
            inpaint,
        ),
        "edge_continuity_score": edge_continuity_score,
        "boundary_color_difference_score": boundary_color_difference_score,
        "visual_reconstruction_difference_score": difference_score(
            source,
            reconstructed_image,
            attribution_region,
        ),
    }
    threshold_by_metric = {
        "white_halo_px": "max_white_halo_px",
        "transparent_hole_px": "max_transparent_hole_px",
        "overlap_deficit_px": "max_overlap_deficit_px",
        "preserve_region_difference_score": "max_preserve_region_difference_score",
        "edge_extension_difference_score": "max_edge_extension_difference_score",
        "inpaint_outside_difference_score": "max_inpaint_outside_difference_score",
        "edge_continuity_score": "max_edge_continuity_score",
        "boundary_color_difference_score": "max_boundary_color_difference_score",
        "visual_reconstruction_difference_score": (
            "max_visual_reconstruction_difference_score"
        ),
    }
    failed_checks = [
        metric
        for metric, threshold in threshold_by_metric.items()
        if metrics[metric] > config[threshold]
    ]
    return {
        "quality_status": "fail" if failed_checks else "pass",
        "allowed_change_region": {
            "pixel_count": sum(allowed_change.histogram()[1:]),
        },
        "metrics": metrics,
        "failed_checks": failed_checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated Live2D part pixels before Cubism import."
    )
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--difference-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-white-halo-px", type=int, default=0)
    parser.add_argument("--max-transparent-hole-px", type=int, default=0)
    parser.add_argument("--max-overlap-deficit-px", type=int, default=0)
    parser.add_argument(
        "--max-edge-extension-difference-score",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-inpaint-outside-difference-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-edge-continuity-score",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-boundary-color-difference-score",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--max-visual-reconstruction-difference-score",
        type=float,
        default=0.01,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _path_value(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        manifest = load_and_validate_mask_manifest(args.mask_manifest, base_dir=base_dir)
        source_data = manifest.get("source_image")
        parts_data = manifest.get("parts")
        if not isinstance(source_data, Mapping) or not isinstance(parts_data, list):
            raise ValueError("mask manifest source_image and parts are required")
        quality_parts = import_parts(manifest)
        source_value = source_data.get("path")
        if not isinstance(source_value, str):
            raise ValueError("source_image.path is required")
        source_path = resolve_inside_base(base_dir, source_value, "source_image.path")
        reconstructed_path = resolve_inside_base(
            base_dir, str(args.reconstructed), "reconstructed image"
        )
        difference_path = resolve_inside_base(
            base_dir, str(args.difference_output), "difference image"
        )
        output_path = resolve_inside_base(base_dir, str(args.output), "quality output")
        require_output_suffix(difference_path, {".png"}, "difference image")
        require_output_suffix(output_path, {".yaml", ".yml"}, "quality output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        if output_path == difference_path:
            raise ValueError("quality YAML and difference image outputs must be different")
        protected_inputs = mask_manifest_protected_paths(
            manifest,
            base_dir,
            manifest_path=manifest_path,
        )
        protected_inputs.add(reconstructed_path)
        if output_path in protected_inputs or difference_path in protected_inputs:
            raise ValueError(
                "quality outputs must not overwrite source, reconstruction, part, mask, "
                "manifest, queue, or canonical derivatives"
            )
        thresholds = _validated_thresholds(
            {
                "max_white_halo_px": args.max_white_halo_px,
                "max_transparent_hole_px": args.max_transparent_hole_px,
                "max_overlap_deficit_px": args.max_overlap_deficit_px,
                "max_preserve_region_difference_score": 0.0,
                "max_edge_extension_difference_score": (
                    args.max_edge_extension_difference_score
                ),
                "max_inpaint_outside_difference_score": (
                    args.max_inpaint_outside_difference_score
                ),
                "max_edge_continuity_score": args.max_edge_continuity_score,
                "max_boundary_color_difference_score": (
                    args.max_boundary_color_difference_score
                ),
                "max_visual_reconstruction_difference_score": (
                    args.max_visual_reconstruction_difference_score
                ),
            }
        )
        report_parts: list[dict[str, Any]] = []
        visual_reconstruction_difference_score = 0.0
        if args.execute:
            existing = [path for path in (difference_path, output_path) if path.exists()]
            if existing and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {existing}")
            source = load_rgba(source_path)
            require_manifest_canvas(source, manifest, "source image")
            reconstructed = load_rgba(reconstructed_path)
            if source.size != reconstructed.size:
                raise ValueError("source and reconstructed images must use the same canvas")
            foreground_regions: list[Image.Image] = []
            for index, raw_part in enumerate(quality_parts):
                layer_id = raw_part.get("layer_id")
                part_value = raw_part.get("output_file")
                target_value = raw_part.get("target_mask")
                protect_value = raw_part.get("protect_mask")
                edge_extension_value = raw_part.get("edge_extension_mask")
                inpaint_value = raw_part.get("inpaint_mask")
                margin = raw_part.get("overlap_margin_px")
                if not isinstance(layer_id, str) or not isinstance(part_value, str):
                    raise ValueError(f"parts[{index}] requires layer_id and output_file")
                if not all(
                    isinstance(value, str)
                    for value in (
                        target_value,
                        protect_value,
                        edge_extension_value,
                        inpaint_value,
                    )
                ):
                    raise ValueError(
                        f"import parts[{index}] requires target/protect/edge-extension/inpaint "
                        "masks"
                    )
                if not isinstance(margin, int):
                    raise ValueError(f"parts[{index}] requires overlap_margin_px")
                part = load_rgba(resolve_inside_base(base_dir, part_value, "part output_file"))
                require_manifest_canvas(part, manifest, f"part {layer_id}")
                assert isinstance(target_value, str)
                assert isinstance(protect_value, str)
                assert isinstance(edge_extension_value, str)
                assert isinstance(inpaint_value, str)
                target = load_binary_mask(
                    resolve_inside_base(base_dir, target_value, "target_mask"), source.size
                )
                protect = load_binary_mask(
                    resolve_inside_base(base_dir, protect_value, "protect_mask"), source.size
                )
                edge_extension = load_binary_mask(
                    resolve_inside_base(
                        base_dir,
                        edge_extension_value,
                        "edge_extension_mask",
                    ),
                    source.size,
                )
                inpaint = load_binary_mask(
                    resolve_inside_base(base_dir, inpaint_value, "inpaint_mask"), source.size
                )
                foreground_regions.append(_union_masks(target, edge_extension, inpaint))
                evaluation = evaluate_part(
                    part,
                    source,
                    target,
                    overlap_margin_px=margin,
                    protect_mask=protect,
                    edge_extension_mask=edge_extension,
                    inpaint_mask=inpaint,
                    reconstructed=reconstructed,
                    thresholds=thresholds,
                )
                allowed_change = evaluation.get("allowed_change_region")
                if isinstance(allowed_change, dict):
                    allowed_change.update(
                        {
                            "edge_extension_mask": edge_extension_value,
                            "inpaint_mask": inpaint_value,
                        }
                    )
                report_parts.append({"layer_id": layer_id, **evaluation})
            foreground = foreground_reconstruction_mask(foreground_regions, reconstructed)
            difference = premultiplied_difference_image(source, reconstructed, foreground)
            atomic_save_png(difference, difference_path, force=args.force)
            visual_reconstruction_difference_score = difference_score(
                source,
                reconstructed,
                foreground,
            )
        else:
            for raw_part in quality_parts:
                edge_extension_value = raw_part.get("edge_extension_mask")
                inpaint_value = raw_part.get("inpaint_mask")
                report_parts.append(
                    {
                        "layer_id": raw_part.get("layer_id"),
                        "quality_status": "pass",
                        "allowed_change_region": {
                            "edge_extension_mask": edge_extension_value,
                            "inpaint_mask": inpaint_value,
                            "pixel_count": 0,
                        },
                        "metrics": {
                            "white_halo_px": 0,
                            "transparent_hole_px": 0,
                            "overlap_deficit_px": 0,
                            "preserve_region_difference_score": 0.0,
                            "edge_extension_difference_score": 0.0,
                            "inpaint_region_source_difference_score": 0.0,
                            "inpaint_outside_difference_score": 0.0,
                            "edge_continuity_score": 0.0,
                            "boundary_color_difference_score": 0.0,
                            "visual_reconstruction_difference_score": 0.0,
                        },
                        "failed_checks": [],
                    }
                )
        failed_parts = sum(
            1 for part in report_parts if part.get("quality_status") == "fail"
        )
        visual_threshold = float(thresholds["max_visual_reconstruction_difference_score"])
        computed_result = (
            "fail"
            if failed_parts or visual_reconstruction_difference_score > visual_threshold
            else "pass"
        )
        report = {
            "schema_version": 2,
            "project": manifest.get("project"),
            "derived_from": {
                "mask_manifest": _path_value(manifest_path, base_dir),
                "asset_generation_queue": (
                    manifest.get("derived_from", {}).get("asset_generation_queue")
                    if isinstance(manifest.get("derived_from"), Mapping)
                    else "<fixture>"
                ),
            },
            "source_image": _path_value(source_path, base_dir),
            "reconstructed_source": _path_value(reconstructed_path, base_dir),
            "difference_image": _path_value(difference_path, base_dir),
            "parts": report_parts,
            "thresholds": thresholds,
            "summary": {
                "total_parts": len(report_parts),
                "failed_parts": failed_parts,
                "result": computed_result,
                "visual_reconstruction_difference_score": (
                    visual_reconstruction_difference_score
                ),
            },
        }
        issues = validate_asset_quality(report)
        if issues:
            raise ValueError("; ".join(issue.format() for issue in issues))
        if args.execute:
            write_yaml(output_path, report, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "output": str(output_path),
        "difference_image": str(difference_path),
        "quality_result": computed_result if args.execute else "not_run",
        "evaluated_parts": len(report_parts) if args.execute else 0,
        "failed_parts": failed_parts if args.execute else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
