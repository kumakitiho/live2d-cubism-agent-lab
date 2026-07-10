from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance

from tools.asset_pipeline_common import (
    atomic_save_png,
    load_rgba,
    load_soft_mask,
    require_output_suffix,
    resolve_inside_base,
)


def build_inpainting_preview(
    reference: Image.Image,
    candidate: Image.Image,
    inpaint_mask: Image.Image,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    reference_rgba = reference.convert("RGBA")
    candidate_rgba = candidate.convert("RGBA")
    mask = inpaint_mask.convert("L")
    if reference_rgba.size != candidate_rgba.size or mask.size != reference_rgba.size:
        raise ValueError("preview inputs must use the same canvas")
    box = crop_box or (0, 0, reference_rgba.width, reference_rgba.height)
    left, top, right, bottom = box
    horizontal_ok = 0 <= left < right <= reference_rgba.width
    vertical_ok = 0 <= top < bottom <= reference_rgba.height
    if not horizontal_ok or not vertical_ok:
        raise ValueError("preview crop_box must stay inside the canvas")
    before = reference_rgba.crop(box)
    after = candidate_rgba.crop(box)
    crop_mask = mask.crop(box)
    difference = ImageEnhance.Contrast(ImageChops.difference(before, after)).enhance(3.0)
    overlay = after.copy()
    tint = Image.new("RGBA", after.size, (255, 0, 180, 112))
    overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", after.size), crop_mask))
    preview = Image.new("RGBA", (before.width * 4, before.height), (0, 0, 0, 0))
    for index, panel in enumerate((before, after, difference, overlay)):
        preview.paste(panel, (index * before.width, 0))
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a before/candidate/difference/mask preview."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("inpaint_mask", type=Path)
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
        reference_path = resolve_inside_base(base_dir, str(args.reference), "reference")
        candidate_path = resolve_inside_base(base_dir, str(args.candidate), "candidate")
        mask_path = resolve_inside_base(base_dir, str(args.inpaint_mask), "inpaint_mask")
        output_path = resolve_inside_base(base_dir, str(args.output), "output")
        require_output_suffix(output_path, {".png"}, "preview output")
        if output_path in {reference_path, candidate_path, mask_path}:
            raise ValueError("preview output must not overwrite an input")
        if args.execute:
            reference = load_rgba(reference_path)
            candidate = load_rgba(candidate_path)
            mask = load_soft_mask(mask_path, reference.size)
            atomic_save_png(
                build_inpainting_preview(reference, candidate, mask),
                output_path,
                force=args.force,
            )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    result = {"status": "written" if args.execute else "planned", "output": str(output_path)}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
