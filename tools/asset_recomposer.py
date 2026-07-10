from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import reduce
from pathlib import Path

from PIL import Image, ImageChops

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


def recompose_parts(
    canvas: tuple[int, int],
    parts: Sequence[tuple[int, Image.Image]],
) -> Image.Image:
    result = Image.new("RGBA", canvas, (0, 0, 0, 0))
    for _draw_order, part in sorted(parts, key=lambda item: item[0]):
        rgba = part.convert("RGBA")
        if rgba.size != canvas:
            raise ValueError(f"part canvas mismatch: {rgba.size} != {canvas}")
        result.alpha_composite(rgba)
    return result


def difference_image(source: Image.Image, reconstructed: Image.Image) -> Image.Image:
    source_rgba = source.convert("RGBA")
    reconstructed_rgba = reconstructed.convert("RGBA")
    if source_rgba.size != reconstructed_rgba.size:
        raise ValueError("source and reconstructed images must use the same canvas")
    difference = ImageChops.difference(source_rgba, reconstructed_rgba)
    magnitude = reduce(ImageChops.lighter, difference.split())
    difference.putalpha(magnitude)
    return difference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompose generated parts by draw order and compare with the source image."
    )
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difference-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        manifest = load_and_validate_mask_manifest(args.mask_manifest, base_dir=base_dir)
        source_data = manifest.get("source_image")
        parts_data = manifest.get("parts")
        if not isinstance(source_data, dict) or not isinstance(parts_data, list):
            raise ValueError("mask manifest source_image and parts are required")
        import_part_data = import_parts(manifest)
        source_value = source_data.get("path")
        if not isinstance(source_value, str):
            raise ValueError("source_image.path is required")
        source_path = resolve_inside_base(base_dir, source_value, "source_image.path")
        output_path = resolve_inside_base(base_dir, str(args.output), "reconstructed output")
        difference_path = resolve_inside_base(
            base_dir,
            str(args.difference_output),
            "difference output",
        )
        require_output_suffix(output_path, {".png"}, "reconstructed output")
        require_output_suffix(difference_path, {".png"}, "difference output")
        if output_path == difference_path:
            raise ValueError("reconstructed and difference outputs must use different paths")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        protected_inputs = mask_manifest_protected_paths(
            manifest,
            base_dir,
            manifest_path=manifest_path,
        )
        if output_path in protected_inputs or difference_path in protected_inputs:
            raise ValueError(
                "recomposition outputs must not overwrite source, part, mask, manifest, "
                "queue, or canonical derivatives"
            )
        if args.execute:
            existing = [path for path in (output_path, difference_path) if path.exists()]
            if existing and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {existing}")
            source = load_rgba(source_path)
            require_manifest_canvas(source, manifest, "source image")
            part_images: list[tuple[int, Image.Image]] = []
            for part in import_part_data:
                output_value = part.get("output_file")
                draw_order = part.get("draw_order")
                if not isinstance(output_value, str) or not isinstance(draw_order, int):
                    raise ValueError("part output_file and draw_order are required")
                part_path = resolve_inside_base(base_dir, output_value, "part output_file")
                part_images.append((draw_order, load_rgba(part_path)))
            reconstructed = recompose_parts(source.size, part_images)
            difference = difference_image(source, reconstructed)
            atomic_save_png(reconstructed, output_path, force=args.force)
            atomic_save_png(difference, difference_path, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "reconstructed_source": str(output_path),
        "difference_image": str(difference_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
