from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from tools.asset_pipeline_common import atomic_save_png, require_same_canvas

COLORS = {
    "target": (255, 70, 90),
    "protect": (40, 220, 100),
    "edge_extension": (60, 130, 255),
    "inpaint": (190, 70, 255),
    "conflict": (255, 210, 20),
    "adjacent": (30, 220, 220),
}


def _overlay(
    base: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(mask.convert("L").point(lambda value: round(value * opacity), mode="L"))
    return Image.alpha_composite(base, layer)


def build_mask_derivation_preview(
    source: Image.Image,
    target_mask: Image.Image,
    candidate_mask: Image.Image,
    *,
    conflict_mask: Image.Image | None = None,
    adjacent_mask: Image.Image | None = None,
    mask_type: str,
) -> Image.Image:
    if mask_type not in {"protect", "edge_extension", "inpaint"}:
        raise ValueError(f"unsupported mask type: {mask_type}")
    conflict_mask = conflict_mask or Image.new("L", source.size, 0)
    adjacent_mask = adjacent_mask or Image.new("L", source.size, 0)
    require_same_canvas(
        {
            "source": source,
            "target": target_mask,
            "candidate": candidate_mask,
            "conflict": conflict_mask,
            "adjacent": adjacent_mask,
        }
    )
    result = source.convert("RGBA")
    result = _overlay(result, adjacent_mask, COLORS["adjacent"], 0.22)
    result = _overlay(result, target_mask, COLORS["target"], 0.22)
    result = _overlay(result, candidate_mask, COLORS[mask_type], 0.55)
    result = _overlay(result, conflict_mask, COLORS["conflict"], 0.72)
    draw = ImageDraw.Draw(result)
    label = f"mask: {mask_type}  target:red candidate:{mask_type} conflict:yellow adjacent:cyan"
    box = draw.textbbox((0, 0), label)
    draw.rectangle((0, 0, min(result.width - 1, box[2] + 8), box[3] + 6), fill=(0, 0, 0, 190))
    draw.text((4, 3), label, fill=(255, 255, 255, 255))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a color-coded mask derivation preview.")
    parser.add_argument("source", type=Path)
    parser.add_argument("target_mask", type=Path)
    parser.add_argument("candidate_mask", type=Path)
    parser.add_argument(
        "--mask-type",
        choices=("protect", "edge_extension", "inpaint"),
        required=True,
    )
    parser.add_argument("--conflict-mask", type=Path)
    parser.add_argument("--adjacent-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _load(path: Path, mode: str) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as opened:
        return opened.convert(mode)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = {args.source.resolve(), args.target_mask.resolve(), args.candidate_mask.resolve()}
        for optional in (args.conflict_mask, args.adjacent_mask):
            if optional is not None:
                inputs.add(optional.resolve())
        if args.output.resolve() in inputs:
            raise ValueError("preview output must not overwrite an input image")
        source = _load(args.source, "RGBA")
        target = _load(args.target_mask, "L")
        candidate = _load(args.candidate_mask, "L")
        conflict = _load(args.conflict_mask, "L") if args.conflict_mask else None
        adjacent = _load(args.adjacent_mask, "L") if args.adjacent_mask else None
        preview = build_mask_derivation_preview(
            source,
            target,
            candidate,
            conflict_mask=conflict,
            adjacent_mask=adjacent,
            mask_type=args.mask_type,
        )
        if args.execute:
            atomic_save_png(preview, args.output, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"status: {'written' if args.execute else 'planned'}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
