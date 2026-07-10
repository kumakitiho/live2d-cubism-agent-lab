from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import resolve_inside_base, write_yaml

NEGATIVE_CONSTRAINTS = (
    "full character regeneration",
    "pose change",
    "face identity change",
    "eye color change",
    "line thickness change",
    "lighting direction change",
    "unrelated accessories",
    "opaque background",
    "canvas resize",
    "unmasked region modification",
)


def _find_asset(queue: Mapping[str, Any], layer_id: str) -> Mapping[str, Any]:
    assets = queue.get("assets")
    if not isinstance(assets, list):
        raise ValueError("asset queue assets must be a list")
    for asset in assets:
        if isinstance(asset, Mapping) and asset.get("layer_id") == layer_id:
            return asset
    raise ValueError(f"unknown queue layer_id: {layer_id}")


def _text(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value:
        return ", ".join(str(item) for item in value)
    return fallback


def build_inpainting_prompt(
    character_spec: Mapping[str, Any],
    asset_queue: Mapping[str, Any],
    layer_id: str,
) -> dict[str, Any]:
    if character_spec.get("project") != asset_queue.get("project"):
        raise ValueError("character spec and asset queue project must match")
    asset = _find_asset(asset_queue, layer_id)
    appearance = character_spec.get("appearance")
    observed = appearance.get("observed") if isinstance(appearance, Mapping) else {}
    if not isinstance(observed, Mapping):
        observed = {}
    identity = "; ".join(
        (
            f"hair: {_text(observed.get('hair'), 'match source character hair')}",
            f"eyes: {_text(observed.get('eyes'), 'match source character eyes')}",
            f"outfit: {_text(observed.get('outfit'), 'match source character outfit')}",
        )
    )
    role = _text(asset.get("role"), layer_id)
    side = _text(asset.get("side"), "C")
    dependencies = _text(asset.get("dependencies"), "adjacent source geometry")
    components = {
        "character_identity": identity,
        "line_style": _text(observed.get("line_style"), "exactly match source line style"),
        "palette": _text(observed.get("palette"), "sample and match the surrounding palette"),
        "lighting_direction": _text(
            observed.get("lighting_direction"), "preserve source lighting direction"
        ),
        "part_role": role,
        "side": side,
        "surrounding_geometry": _text(asset.get("surrounding_geometry"), dependencies),
        "hidden_region_purpose": _text(
            asset.get("hidden_region_purpose"),
            f"complete only the occluded pixels required by {role}",
        ),
        "background": "transparent background",
        "alignment": "same canvas size and origin",
        "edit_scope": "only modify the masked region",
    }
    prompt = ". ".join(f"{key.replace('_', ' ')}: {value}" for key, value in components.items())
    return {
        "schema_version": 1,
        "project": asset_queue.get("project"),
        "layer_id": layer_id,
        "prompt": prompt,
        "negative_prompt": ", ".join(NEGATIVE_CONSTRAINTS),
        "components": components,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-consistent part inpainting prompt."
    )
    parser.add_argument("character_spec", type=Path)
    parser.add_argument("asset_queue", type=Path)
    parser.add_argument("--layer-id", required=True)
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
        spec_path = resolve_inside_base(base_dir, str(args.character_spec), "character spec")
        queue_path = resolve_inside_base(base_dir, str(args.asset_queue), "asset queue")
        output_path = resolve_inside_base(base_dir, str(args.output), "output")
        if output_path in {spec_path, queue_path}:
            raise ValueError("prompt output must not overwrite an input")
        result = build_inpainting_prompt(
            load_yaml_mapping(spec_path), load_yaml_mapping(queue_path), args.layer_id
        )
        if args.execute:
            write_yaml(output_path, result, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    report = {
        "status": "written" if args.execute else "planned",
        "output": str(output_path),
        "prompt": result,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
