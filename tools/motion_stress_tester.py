from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image

from tools.asset_pipeline_common import (
    atomic_save_png,
    import_parts,
    load_and_validate_mask_manifest,
    load_rgba,
    mask_manifest_protected_paths,
    require_manifest_canvas,
    require_output_suffix,
    resolve_inside_base,
)
from tools.asset_recomposer import recompose_parts


def shift_part(part: Image.Image, dx: int, dy: int) -> Image.Image:
    source = part.convert("RGBA")
    result = Image.new("RGBA", source.size, (0, 0, 0, 0))
    left = max(0, -dx)
    top = max(0, -dy)
    right = min(source.width, source.width - dx)
    bottom = min(source.height, source.height - dy)
    if right <= left or bottom <= top:
        return result
    cropped = source.crop((left, top, right, bottom))
    result.alpha_composite(cropped, (max(0, dx), max(0, dy)))
    return result


def create_part_motion_debug_preview(part: Image.Image, distance_px: int) -> Image.Image:
    if distance_px <= 0:
        raise ValueError("distance_px must be positive")
    rgba = part.convert("RGBA")
    offsets = ((-distance_px, 0), (0, 0), (distance_px, 0))
    preview = Image.new("RGBA", (rgba.width * len(offsets), rgba.height), (32, 32, 32, 255))
    for index, (dx, dy) in enumerate(offsets):
        frame = shift_part(rgba, dx, dy)
        preview.alpha_composite(frame, (index * rgba.width, 0))
    return preview


def create_motion_stress_preview(
    canvas: tuple[int, int],
    parts: Sequence[tuple[int, str, Image.Image]],
    target_layer_id: str,
    distance_px: int,
) -> Image.Image:
    if distance_px <= 0:
        raise ValueError("distance_px must be positive")
    if target_layer_id not in {layer_id for _order, layer_id, _image in parts}:
        raise ValueError(f"target part is not an import part: {target_layer_id}")
    offsets = (-distance_px, 0, distance_px)
    preview = Image.new("RGBA", (canvas[0] * len(offsets), canvas[1]), (32, 32, 32, 255))
    for index, dx in enumerate(offsets):
        frame_parts = [
            (
                draw_order,
                shift_part(image, dx, 0) if layer_id == target_layer_id else image,
            )
            for draw_order, layer_id, image in parts
        ]
        frame = recompose_parts(canvas, frame_parts)
        rendered = Image.new("RGBA", canvas, (32, 32, 32, 255))
        rendered.alpha_composite(frame)
        preview.alpha_composite(rendered, (index * canvas[0], 0))
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompose all import parts while translating one part for motion stress."
    )
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-part-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        if args.distance <= 0:
            raise ValueError("--distance must be positive")
        manifest = load_and_validate_mask_manifest(args.mask_manifest, base_dir=base_dir)
        parts = import_parts(manifest)
        all_parts = manifest.get("parts")
        source_data = manifest.get("source_image")
        if not isinstance(all_parts, list) or not isinstance(source_data, Mapping):
            raise ValueError("mask manifest source_image and parts are required")
        part = next((item for item in parts if item.get("layer_id") == args.part), None)
        if part is None:
            raise ValueError(f"target part is not an import part: {args.part}")
        part_value = part.get("output_file")
        if not isinstance(part_value, str):
            raise ValueError("part output_file is required")
        input_path = resolve_inside_base(base_dir, part_value, "part output_file")
        output_path = resolve_inside_base(base_dir, str(args.output), "preview output")
        require_output_suffix(output_path, {".png"}, "preview output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        protected_inputs = mask_manifest_protected_paths(
            manifest,
            base_dir,
            manifest_path=manifest_path,
        )
        if output_path in protected_inputs:
            raise ValueError(
                "preview output must not overwrite source, part, mask, manifest, queue, "
                "or canonical derivatives"
            )
        if args.execute:
            if output_path.exists() and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
            if args.debug_part_only:
                part_image = load_rgba(input_path)
                require_manifest_canvas(part_image, manifest, "part image")
                preview = create_part_motion_debug_preview(part_image, args.distance)
            else:
                part_images: list[tuple[int, str, Image.Image]] = []
                for index, raw_part in enumerate(parts):
                    layer_id = raw_part.get("layer_id")
                    draw_order = raw_part.get("draw_order")
                    output_value = raw_part.get("output_file")
                    if (
                        not isinstance(layer_id, str)
                        or not isinstance(draw_order, int)
                        or not isinstance(output_value, str)
                    ):
                        raise ValueError(f"import parts[{index}] has invalid render metadata")
                    image = load_rgba(
                        resolve_inside_base(base_dir, output_value, "part output_file")
                    )
                    require_manifest_canvas(image, manifest, f"part {layer_id}")
                    part_images.append((draw_order, layer_id, image))
                preview = create_motion_stress_preview(
                    part_images[0][2].size,
                    part_images,
                    args.part,
                    args.distance,
                )
            atomic_save_png(preview, output_path, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "part": args.part,
        "distance_px": args.distance,
        "mode": "part_debug" if args.debug_part_only else "full_recomposition",
        "output": str(output_path),
        "note": "translation-only full-model preview; deformation quality is evaluated in Cubism",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
