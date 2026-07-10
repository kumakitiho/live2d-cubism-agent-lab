from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    is_positive_int,
    referenced_artifact_paths,
    require_output_suffix,
    resolve_inside_base,
    validate_mask_manifest,
    write_yaml,
)
from tools.asset_queue_builder import normalize_queue_ref


def build_mask_manifest(
    queue: Mapping[str, Any],
    *,
    queue_ref: str,
) -> dict[str, Any]:
    source_image = queue.get("source_image")
    canvas = queue.get("canvas")
    assets = queue.get("assets")
    if not isinstance(source_image, Mapping) or not isinstance(canvas, Mapping):
        raise ValueError("queue source_image and canvas must be mappings")
    if not isinstance(assets, list) or not assets:
        raise ValueError("queue assets must be a non-empty list")
    source_path = source_image.get("path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("queue source_image.path is required")

    parts: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"queue assets[{index}] must be a mapping")
        if not is_positive_int(asset.get("draw_order")):
            raise ValueError(f"queue assets[{index}].draw_order must be a positive integer")
        parts.append(
            {
                "layer_id": asset.get("layer_id"),
                "target_mask": asset.get("target_mask"),
                "protect_mask": asset.get("protect_mask"),
                "inpaint_mask": asset.get("inpaint_mask"),
                "source_file": source_path,
                "output_file": asset.get("source_file"),
                "generation_method": asset.get("generation_method"),
                "dependencies": deepcopy(asset.get("dependencies")),
                "draw_order": asset.get("draw_order"),
                "overlap_margin_px": asset.get("overlap_margin_px"),
                "quality_status": asset.get("quality_status"),
                "refinement_attempts": asset.get("refinement_attempts"),
                "include_in_import": asset.get("include_in_import"),
            }
        )
    parts.sort(key=lambda part: int(part["draw_order"]))
    result = {
        "schema_version": 1,
        "project": queue.get("project"),
        "derived_from": {
            "asset_generation_queue": queue_ref,
            "queue_schema_version": queue.get("schema_version"),
        },
        "source_image": {"path": source_path},
        "canvas": deepcopy(dict(canvas)),
        "parts": parts,
    }
    issues = validate_mask_manifest(result)
    if issues:
        raise ValueError("; ".join(issue.format() for issue in issues))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a mask candidate manifest from the canonical asset queue."
    )
    parser.add_argument("queue", type=Path)
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
        queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
        queue = load_yaml_mapping(queue_path)
        queue_ref = normalize_queue_ref(queue_path, base_dir)
        manifest = build_mask_manifest(queue, queue_ref=queue_ref)
        output_value: str
        if args.output is not None:
            output_value = str(args.output)
        else:
            derivatives = queue.get("derivatives")
            configured_output = (
                derivatives.get("mask_manifest") if isinstance(derivatives, Mapping) else None
            )
            if not isinstance(configured_output, str):
                raise ValueError("queue derivatives.mask_manifest or --output is required")
            output_value = configured_output
        output = resolve_inside_base(base_dir, output_value, "mask manifest output")
        require_output_suffix(output, {".yaml", ".yml"}, "mask manifest output")
        protected_paths = referenced_artifact_paths(
            queue,
            base_dir,
            document_path=queue_path,
        )
        derivatives = queue.get("derivatives")
        owned_output = (
            derivatives.get("mask_manifest") if isinstance(derivatives, Mapping) else None
        )
        if isinstance(owned_output, str):
            protected_paths.discard(
                resolve_inside_base(base_dir, owned_output, "derivatives.mask_manifest")
            )
        if output in protected_paths:
            raise ValueError(
                "mask manifest output must not overwrite source, part, mask, queue, "
                "or another canonical derivative"
            )
        if args.execute:
            write_yaml(output, manifest, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "output": str(output),
        "part_count": len(manifest["parts"]),
        "backend": "mask_generation_not_connected",
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
