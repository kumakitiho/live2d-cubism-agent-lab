from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from tools.asset_pipeline_common import (
    find_part,
    load_and_validate_mask_manifest,
    load_mask,
    load_rgba,
    require_manifest_canvas,
    resolve_inside_base,
)

LOCAL_COMPLETION_METHODS = {"extract_and_edge_repair", "transparency_fill"}


def transparency_fill(
    part: Image.Image,
    inpaint_mask: Image.Image,
    protect_mask: Image.Image,
    *,
    iterations: int = 8,
) -> Image.Image:
    rgba = part.convert("RGBA")
    inpaint = inpaint_mask.convert("L")
    protect = protect_mask.convert("L")
    if rgba.size != inpaint.size or rgba.size != protect.size:
        raise ValueError("part and masks must use the same canvas")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    width, height = rgba.size
    current = rgba.copy()
    inpaint_pixels: Any = inpaint.load()
    protect_pixels: Any = protect.load()
    for _ in range(iterations):
        source_pixels: Any = current.load()
        updated = current.copy()
        updated_pixels: Any = updated.load()
        changed = False
        for y in range(height):
            for x in range(width):
                if inpaint_pixels[x, y] == 0 or protect_pixels[x, y] > 0:
                    continue
                if source_pixels[x, y][3] > 0:
                    continue
                neighbors: list[tuple[int, int, int, int]] = []
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < width and 0 <= ny < height:
                        candidate = source_pixels[nx, ny]
                        if candidate[3] > 0:
                            neighbors.append(candidate)
                if neighbors:
                    count = len(neighbors)
                    updated_pixels[x, y] = (
                        sum(pixel[0] for pixel in neighbors) // count,
                        sum(pixel[1] for pixel in neighbors) // count,
                        sum(pixel[2] for pixel in neighbors) // count,
                        max(pixel[3] for pixel in neighbors),
                    )
                    changed = True
        if not changed:
            break
        current = updated
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Complete transparent edge/hidden pixels without changing protected source pixels."
        )
    )
    parser.add_argument("mask_manifest", type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=8)
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
        method = part.get("generation_method")
        input_value = part.get("output_file")
        inpaint_value = part.get("inpaint_mask")
        protect_value = part.get("protect_mask")
        output_value = str(args.output) if args.output is not None else input_value
        if not isinstance(input_value, str) or not input_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        if not isinstance(inpaint_value, str) or not inpaint_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        if not isinstance(protect_value, str) or not protect_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        input_path = resolve_inside_base(base_dir, input_value, "part input")
        inpaint_path = resolve_inside_base(base_dir, inpaint_value, "inpaint_mask")
        protect_path = resolve_inside_base(base_dir, protect_value, "protect_mask")
        output_path = resolve_inside_base(base_dir, output_value, "output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        if output_path in {inpaint_path, protect_path, manifest_path}:
            raise ValueError("completion output must not overwrite mask or manifest inputs")
        if args.execute:
            if method not in LOCAL_COMPLETION_METHODS:
                raise ValueError(f"{method} backend is not connected; only plan this operation")
            if output_path.exists() and output_path != input_path and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
            if output_path == input_path and not args.force:
                raise FileExistsError("overwriting the input part requires --force")
            image = load_rgba(input_path)
            require_manifest_canvas(image, manifest, "part image")
            inpaint_mask = load_mask(inpaint_path, image.size)
            protect_mask = load_mask(protect_path, image.size)
            completed = transparency_fill(
                image,
                inpaint_mask,
                protect_mask,
                iterations=args.iterations,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            completed.save(output_path, format="PNG")
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "part": args.part,
        "generation_method": method,
        "output": str(output_path),
        "backend_connected": method in LOCAL_COMPLETION_METHODS,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
