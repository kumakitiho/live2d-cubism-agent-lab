from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from tools.asset_pipeline_common import (
    find_part,
    load_and_validate_mask_manifest,
    load_rgba,
    require_manifest_canvas,
    resolve_inside_base,
)


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


def create_motion_stress_preview(part: Image.Image, distance_px: int) -> Image.Image:
    if distance_px <= 0:
        raise ValueError("distance_px must be positive")
    rgba = part.convert("RGBA")
    offsets = ((-distance_px, 0), (0, 0), (distance_px, 0))
    preview = Image.new("RGBA", (rgba.width * len(offsets), rgba.height), (32, 32, 32, 255))
    for index, (dx, dy) in enumerate(offsets):
        frame = shift_part(rgba, dx, dy)
        preview.alpha_composite(frame, (index * rgba.width, 0))
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a simple pre-Cubism translated-part motion stress preview."
    )
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
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
        part = find_part(manifest, args.part)
        part_value = part.get("output_file")
        if not isinstance(part_value, str):
            raise ValueError("part output_file is required")
        input_path = resolve_inside_base(base_dir, part_value, "part output_file")
        output_path = resolve_inside_base(base_dir, str(args.output), "preview output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        protected_inputs = {input_path, manifest_path}
        for field in ("target_mask", "protect_mask", "inpaint_mask"):
            value = part.get(field)
            if isinstance(value, str):
                protected_inputs.add(resolve_inside_base(base_dir, value, field))
        if output_path in protected_inputs:
            raise ValueError("preview output must not overwrite part or manifest inputs")
        if args.execute:
            if output_path.exists() and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
            part_image = load_rgba(input_path)
            require_manifest_canvas(part_image, manifest, "part image")
            preview = create_motion_stress_preview(part_image, args.distance)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            preview.save(output_path, format="PNG")
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "part": args.part,
        "distance_px": args.distance,
        "output": str(output_path),
        "note": "translation-only preview; deformation quality is evaluated in Cubism",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
