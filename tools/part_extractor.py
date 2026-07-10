from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from tools.asset_pipeline_common import (
    atomic_save_png,
    find_part,
    load_and_validate_mask_manifest,
    load_rgba,
    load_soft_mask,
    mask_manifest_protected_paths,
    require_manifest_canvas,
    require_output_suffix,
    resolve_inside_base,
)


def extract_rgba(source: Image.Image, target_mask: Image.Image) -> Image.Image:
    source_rgba = source.convert("RGBA")
    mask = target_mask.convert("L")
    if source_rgba.size != mask.size:
        raise ValueError(f"canvas mismatch: source={source_rgba.size}, mask={mask.size}")
    result = source_rgba.copy()
    result.putalpha(ImageChops.multiply(source_rgba.getchannel("A"), mask))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract source RGBA pixels with a target mask.")
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        manifest = load_and_validate_mask_manifest(args.mask_manifest, base_dir=base_dir)
        part = find_part(manifest, args.part)
        source_value = part.get("source_file")
        target_mask_value = part.get("target_mask")
        output_value = str(args.output) if args.output is not None else part.get("output_file")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError("part source_file and target_mask are required")
        if not isinstance(target_mask_value, str) or not target_mask_value:
            raise ValueError("part source_file and target_mask are required")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("part output_file or --output is required")
        source_path = resolve_inside_base(base_dir, source_value, "source_file")
        target_mask_path = resolve_inside_base(base_dir, target_mask_value, "target_mask")
        output_path = resolve_inside_base(base_dir, output_value, "output")
        require_output_suffix(output_path, {".png"}, "part output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        protected_inputs = mask_manifest_protected_paths(
            manifest,
            base_dir,
            manifest_path=manifest_path,
        )
        designated_output = part.get("output_file")
        if isinstance(designated_output, str):
            protected_inputs.discard(
                resolve_inside_base(base_dir, designated_output, "part output_file")
            )
        if output_path in protected_inputs:
            raise ValueError(
                "part output must not overwrite source, another part, mask, manifest, "
                "queue, or canonical derivatives"
            )
        if args.execute:
            if output_path.exists() and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
            source = load_rgba(source_path)
            require_manifest_canvas(source, manifest, "source image")
            target_mask = load_soft_mask(target_mask_path, source.size)
            result_image = extract_rgba(source, target_mask)
            atomic_save_png(result_image, output_path, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "part": args.part,
        "output": str(output_path),
        "operation": "target_mask_rgba_extract",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
