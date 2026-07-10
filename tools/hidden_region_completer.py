from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from tools.asset_pipeline_common import (
    atomic_save_png,
    find_part,
    load_and_validate_mask_manifest,
    load_binary_mask,
    load_rgba,
    load_soft_mask,
    mask_manifest_protected_paths,
    require_manifest_canvas,
    require_output_suffix,
    resolve_inside_base,
)

LOCAL_COMPLETION_METHODS = {"extract_and_edge_repair", "transparency_fill"}


def extract_and_edge_repair(
    part: Image.Image,
    target_mask: Image.Image,
    protect_mask: Image.Image,
) -> Image.Image:
    rgba = part.convert("RGBA")
    target = target_mask.convert("L")
    protect = protect_mask.convert("L")
    if rgba.size != target.size or rgba.size != protect.size:
        raise ValueError("part and masks must use the same canvas")

    binary_target = target.point(lambda value: 255 if value > 0 else 0, mode="L")
    boundary = ImageChops.subtract(binary_target, binary_target.filter(ImageFilter.MinFilter(3)))
    antialiased = target.point(
        lambda value: 255 if 0 < value < 255 else 0,
        mode="L",
    )
    repair_region = ImageChops.subtract(ImageChops.lighter(boundary, antialiased), protect)
    bounds = repair_region.getbbox()
    if bounds is None:
        return rgba.copy()

    source_pixels: Any = rgba.load()
    repair_pixels: Any = repair_region.load()
    result = rgba.copy()
    result_pixels: Any = result.load()
    left, top, right, bottom = bounds
    for y in range(top, bottom):
        for x in range(left, right):
            if repair_pixels[x, y] == 0:
                continue
            pixel = source_pixels[x, y]
            alpha = pixel[3]
            if alpha == 0:
                continue
            neighbors: list[tuple[int, int, int, int]] = []
            for ny in range(max(0, y - 1), min(rgba.height, y + 2)):
                for nx in range(max(0, x - 1), min(rgba.width, x + 2)):
                    candidate = source_pixels[nx, ny]
                    if candidate[3] > alpha:
                        neighbors.append(candidate)
            if neighbors:
                count = len(neighbors)
                result_pixels[x, y] = (
                    sum(candidate[0] for candidate in neighbors) // count,
                    sum(candidate[1] for candidate in neighbors) // count,
                    sum(candidate[2] for candidate in neighbors) // count,
                    alpha,
                )
    return result


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

    current = rgba.copy()
    inpaint_pixels: Any = inpaint.load()
    protect_pixels: Any = protect.load()
    bounds = inpaint.getbbox()
    if bounds is None:
        return current
    left, top, right, bottom = bounds
    for _ in range(iterations):
        source_pixels: Any = current.load()
        updated = current.copy()
        updated_pixels: Any = updated.load()
        changed = False
        for y in range(top, bottom):
            for x in range(left, right):
                if inpaint_pixels[x, y] == 0 or protect_pixels[x, y] > 0:
                    continue
                if source_pixels[x, y][3] > 0:
                    continue
                neighbors: list[tuple[int, int, int, int]] = []
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < rgba.width and 0 <= ny < rgba.height:
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
        target_value = part.get("target_mask")
        inpaint_value = part.get("inpaint_mask")
        protect_value = part.get("protect_mask")
        output_value = str(args.output) if args.output is not None else input_value
        if not isinstance(input_value, str) or not input_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        if not isinstance(target_value, str) or not target_value:
            raise ValueError("part output_file and all masks are required")
        if not isinstance(inpaint_value, str) or not inpaint_value:
            raise ValueError("part output_file and all masks are required")
        if not isinstance(protect_value, str) or not protect_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("part output_file, inpaint_mask, and protect_mask are required")
        input_path = resolve_inside_base(base_dir, input_value, "part input")
        target_path = resolve_inside_base(base_dir, target_value, "target_mask")
        inpaint_path = resolve_inside_base(base_dir, inpaint_value, "inpaint_mask")
        protect_path = resolve_inside_base(base_dir, protect_value, "protect_mask")
        output_path = resolve_inside_base(base_dir, output_value, "output")
        require_output_suffix(output_path, {".png"}, "completion output")
        manifest_path = resolve_inside_base(base_dir, str(args.mask_manifest), "mask manifest")
        protected_inputs = mask_manifest_protected_paths(
            manifest,
            base_dir,
            manifest_path=manifest_path,
        )
        protected_inputs.discard(input_path)
        if output_path in protected_inputs:
            raise ValueError(
                "completion output must not overwrite source, another part, mask, manifest, "
                "queue, or canonical derivatives"
            )
        if args.execute:
            if method not in LOCAL_COMPLETION_METHODS:
                raise ValueError(f"{method} backend is not connected; only plan this operation")
            if output_path.exists() and output_path != input_path and not args.force:
                raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
            if output_path == input_path and not args.force:
                raise FileExistsError("overwriting the input part requires --force")
            image = load_rgba(input_path)
            require_manifest_canvas(image, manifest, "part image")
            protect_mask = load_binary_mask(protect_path, image.size)
            if method == "extract_and_edge_repair":
                target_mask = load_soft_mask(target_path, image.size)
                completed = extract_and_edge_repair(image, target_mask, protect_mask)
            else:
                inpaint_mask = load_binary_mask(inpaint_path, image.size)
                completed = transparency_fill(
                    image,
                    inpaint_mask,
                    protect_mask,
                    iterations=args.iterations,
                )
            atomic_save_png(completed, output_path, force=args.force)
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
