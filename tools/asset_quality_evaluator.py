from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from tools.asset_pipeline_common import (
    load_and_validate_mask_manifest,
    load_mask,
    load_rgba,
    require_manifest_canvas,
    resolve_inside_base,
    validate_asset_quality,
    write_yaml,
)
from tools.asset_recomposer import difference_image


def _binary_alpha(image: Image.Image) -> Image.Image:
    return image.convert("RGBA").getchannel("A").point(
        lambda value: 255 if value > 0 else 0,
        mode="L",
    )


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
) -> int:
    if part.size != target_mask.size:
        raise ValueError("part and target mask must use the same canvas")
    if overlap_margin_px < 0:
        raise ValueError("overlap_margin_px must be non-negative")
    expected = target_mask.convert("L")
    if overlap_margin_px:
        expected = expected.filter(ImageFilter.MaxFilter(overlap_margin_px * 2 + 1))
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
    part_pixels: Any = part_rgba.load()
    source_pixels: Any = source_rgba.load()
    boundary_pixels: Any = boundary.load()
    count = 0
    for y in range(part_rgba.height):
        for x in range(part_rgba.width):
            part_pixel = part_pixels[x, y]
            source_pixel = source_pixels[x, y]
            if boundary_pixels[x, y] == 0 or part_pixel[3] == 0:
                continue
            part_is_white = min(part_pixel[:3]) >= 245
            source_is_white = min(source_pixel[:3]) >= 245 and source_pixel[3] > 0
            if part_is_white and not source_is_white:
                count += 1
    return count


def difference_score(
    reference: Image.Image,
    candidate: Image.Image,
    mask: Image.Image | None = None,
) -> float:
    reference_rgba = reference.convert("RGBA")
    candidate_rgba = candidate.convert("RGBA")
    if reference_rgba.size != candidate_rgba.size:
        raise ValueError("images must use the same canvas")
    difference = ImageChops.difference(reference_rgba, candidate_rgba)
    pixel_count = reference_rgba.width * reference_rgba.height
    if mask is not None:
        mask_l = mask.convert("L")
        if mask_l.size != reference_rgba.size:
            raise ValueError("difference mask must use the same canvas")
        difference = Image.merge(
            "RGBA",
            tuple(ImageChops.multiply(channel, mask_l) for channel in difference.split()),
        )
        pixel_count = sum(mask_l.histogram()[1:])
    total = sum(
        value * count
        for channel in difference.split()
        for value, count in enumerate(channel.histogram())
    )
    maximum = pixel_count * 4 * 255
    return round(total / maximum, 8) if maximum else 0.0


def evaluate_part(
    part: Image.Image,
    source: Image.Image,
    target_mask: Image.Image,
    *,
    overlap_margin_px: int,
    protect_mask: Image.Image | None = None,
) -> dict[str, Any]:
    metrics: dict[str, int | float] = {
        "white_halo_px": count_white_halo(part, source),
        "transparent_hole_px": count_transparent_holes(part, target_mask),
        "overlap_deficit_px": count_overlap_deficit(part, target_mask, overlap_margin_px),
        "difference_score": difference_score(source, part, protect_mask or target_mask),
    }
    failed_checks = [
        name
        for name in ("white_halo_px", "transparent_hole_px", "overlap_deficit_px")
        if metrics[name] > 0
    ]
    if metrics["difference_score"] > 0:
        failed_checks.append("source_pixel_difference")
    return {
        "quality_status": "fail" if failed_checks else "pass",
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
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        if output_path == difference_path:
            raise ValueError("quality YAML and difference image outputs must be different")
        protected_inputs = {source_path, reconstructed_path, manifest_path}
        for raw_part in parts_data:
            if not isinstance(raw_part, Mapping):
                continue
            for field in ("output_file", "target_mask", "protect_mask", "inpaint_mask"):
                value = raw_part.get(field)
                if isinstance(value, str):
                    protected_inputs.add(resolve_inside_base(base_dir, value, field))
        if output_path in protected_inputs or difference_path in protected_inputs:
            raise ValueError(
                "quality outputs must not overwrite source, reconstruction, or manifest"
            )
        report_parts: list[dict[str, Any]] = []
        global_difference_score = 0.0
        if args.execute:
            existing = [path for path in (difference_path, output_path) if path.exists()]
            if existing and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {existing}")
            source = load_rgba(source_path)
            require_manifest_canvas(source, manifest, "source image")
            reconstructed = load_rgba(reconstructed_path)
            if source.size != reconstructed.size:
                raise ValueError("source and reconstructed images must use the same canvas")
            difference = difference_image(source, reconstructed)
            difference_path.parent.mkdir(parents=True, exist_ok=True)
            difference.save(difference_path, format="PNG")
            global_difference_score = difference_score(source, reconstructed)
            for index, raw_part in enumerate(parts_data):
                if not isinstance(raw_part, Mapping):
                    raise ValueError(f"parts[{index}] must be a mapping")
                layer_id = raw_part.get("layer_id")
                part_value = raw_part.get("output_file")
                target_value = raw_part.get("target_mask")
                protect_value = raw_part.get("protect_mask")
                margin = raw_part.get("overlap_margin_px")
                if not isinstance(layer_id, str) or not isinstance(part_value, str):
                    raise ValueError(f"parts[{index}] requires layer_id and output_file")
                if not isinstance(target_value, str) or not isinstance(protect_value, str):
                    raise ValueError(f"parts[{index}] requires target_mask and protect_mask")
                if not isinstance(margin, int):
                    raise ValueError(f"parts[{index}] requires overlap_margin_px")
                part = load_rgba(resolve_inside_base(base_dir, part_value, "part output_file"))
                target = load_mask(
                    resolve_inside_base(base_dir, target_value, "target_mask"), source.size
                )
                protect = load_mask(
                    resolve_inside_base(base_dir, protect_value, "protect_mask"), source.size
                )
                evaluation = evaluate_part(
                    part,
                    source,
                    target,
                    overlap_margin_px=margin,
                    protect_mask=protect,
                )
                report_parts.append({"layer_id": layer_id, **evaluation})
        else:
            for raw_part in parts_data:
                if isinstance(raw_part, Mapping):
                    report_parts.append(
                        {
                            "layer_id": raw_part.get("layer_id"),
                            "quality_status": "pass",
                            "metrics": {
                                "white_halo_px": 0,
                                "transparent_hole_px": 0,
                                "overlap_deficit_px": 0,
                                "difference_score": 0.0,
                            },
                            "failed_checks": [],
                        }
                    )
        failed_parts = sum(
            1 for part in report_parts if part.get("quality_status") == "fail"
        )
        computed_result = "fail" if failed_parts or global_difference_score > 0.0 else "pass"
        report = {
            "schema_version": 1,
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
            "thresholds": {"max_global_difference_score": 0.0},
            "summary": {
                "total_parts": len(report_parts),
                "failed_parts": failed_parts,
                "result": computed_result,
                "global_difference_score": global_difference_score,
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
