from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from tools.asset_pipeline_common import atomic_save_png


def build_segmentation_preview(
    source: Image.Image,
    soft_mask: Image.Image,
    *,
    color: tuple[int, int, int] = (255, 64, 96),
    opacity: float = 0.45,
) -> Image.Image:
    if source.size != soft_mask.size:
        raise ValueError(f"preview canvas mismatch: {source.size} != {soft_mask.size}")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("preview opacity must be between 0 and 1")
    overlay = Image.new("RGBA", source.size, (*color, 0))
    overlay.putalpha(
        soft_mask.convert("L").point(
            lambda value: round(value * opacity),
            mode="L",
        )
    )
    return Image.alpha_composite(source.convert("RGBA"), overlay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay a soft segmentation mask on its source.")
    parser.add_argument("source", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = {args.source.resolve(), args.mask.resolve(), args.output.resolve()}
        if len(paths) != 3:
            raise ValueError("preview output must not overwrite source or mask")
        if not args.source.is_file():
            raise FileNotFoundError(f"source image not found: {args.source}")
        if not args.mask.is_file():
            raise FileNotFoundError(f"soft mask not found: {args.mask}")
        with Image.open(args.source) as opened_source:
            source = opened_source.convert("RGBA")
        with Image.open(args.mask) as opened_mask:
            mask = opened_mask.convert("L")
        preview = build_segmentation_preview(source, mask, opacity=args.opacity)
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
