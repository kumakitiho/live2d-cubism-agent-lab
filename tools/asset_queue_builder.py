from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping, validate_layer_map
from tools.asset_manifest_validator import validate_asset_manifest

MANIFEST_PART_FIELDS = (
    "layer_id",
    "layer_name",
    "role",
    "side",
    "source_file",
    "mask_file",
    "prompt_id",
    "generation_method",
    "order",
    "required",
    "inferred",
    "review_required",
    "readiness",
    "include_in_import",
)


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"queue {key} must be a mapping")
    return value


def _require_assets(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("queue assets must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"queue assets[{index}] must be a mapping")
        result.append(asset)
    return result


def _derived_from(queue: Mapping[str, Any], queue_ref: str | None) -> dict[str, Any]:
    return {
        "asset_generation_queue": queue_ref or "<in-memory>",
        "queue_schema_version": queue.get("schema_version"),
    }


def derive_asset_manifest(
    queue: Mapping[str, Any],
    *,
    queue_ref: str | None = None,
) -> dict[str, Any]:
    source_image = _require_mapping(queue, "source_image")
    canvas = _require_mapping(queue, "canvas")
    derivatives = _require_mapping(queue, "derivatives")
    constraints = _require_mapping(queue, "import_constraints")
    assets = _require_assets(queue)

    parts: list[dict[str, Any]] = []
    for asset in assets:
        parts.append(
            {key: deepcopy(asset.get(key)) for key in MANIFEST_PART_FIELDS if key in asset}
        )

    return {
        "schema_version": 1,
        "validation_mode": queue.get("validation_mode", "strict"),
        "project": queue.get("project"),
        "derived_from": _derived_from(queue, queue_ref),
        "source_image": deepcopy(dict(source_image)),
        "canvas": deepcopy(dict(canvas)),
        "output": {
            "model_import_psd": derivatives.get("model_import_psd"),
            "layer_map": derivatives.get("layer_map"),
        },
        "import_constraints": deepcopy(dict(constraints)),
        "parts": parts,
    }


def derive_layer_map(
    queue: Mapping[str, Any],
    *,
    queue_ref: str | None = None,
) -> dict[str, Any]:
    canvas = _require_mapping(queue, "canvas")
    derivatives = _require_mapping(queue, "derivatives")
    assets = _require_assets(queue)

    layers: list[dict[str, Any]] = []
    for asset in assets:
        layers.append(
            {
                "path": asset.get("layer_path"),
                "name": asset.get("layer_name"),
                "layer_id": asset.get("layer_id"),
                "role": asset.get("role"),
                "side": asset.get("side"),
                "source": asset.get("source_file"),
                "order": asset.get("order"),
                "inferred": asset.get("inferred"),
                "review_required": asset.get("review_required"),
                "readiness": asset.get("readiness"),
                "required": asset.get("required"),
            }
        )
    layers.sort(key=lambda layer: int(layer.get("order", 0)))

    return {
        "schema_version": 1,
        "validation_mode": queue.get("validation_mode", "strict"),
        "project": queue.get("project"),
        "derived_from": _derived_from(queue, queue_ref),
        "source_psd": derivatives.get("model_import_psd"),
        "canvas": {
            "width": canvas.get("width"),
            "height": canvas.get("height"),
            "color_mode": canvas.get("color_mode"),
        },
        "layers": layers,
    }


def _resolve_path(base_dir: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"queue derivatives.{field} must be a non-empty path")
    path = Path(value)
    resolved = (path if path.is_absolute() else base_dir / path).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"queue derivatives.{field} must stay inside base-dir") from exc
    return resolved


def _write_yaml_pair(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    layer_map_path: Path,
    layer_map: Mapping[str, Any],
    *,
    force: bool,
) -> None:
    if manifest_path == layer_map_path:
        raise ValueError("manifest and layer map outputs must use different paths")
    outputs = {
        manifest_path: yaml.safe_dump(dict(manifest), allow_unicode=True, sort_keys=False),
        layer_map_path: yaml.safe_dump(dict(layer_map), allow_unicode=True, sort_keys=False),
    }
    existing = {path for path in outputs if path.exists()}
    if existing and not force:
        paths = ", ".join(str(path) for path in sorted(existing))
        raise FileExistsError(f"refusing to overwrite without --force: {paths}")

    token = uuid.uuid4().hex
    temporary = {path: path.with_name(f".{path.name}.{token}.tmp") for path in outputs}
    backups = {path: path.with_name(f".{path.name}.{token}.bak") for path in existing}
    committed = False
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[path].write_text(content, encoding="utf-8")
        for path, backup in backups.items():
            path.replace(backup)
        for path in outputs:
            temporary[path].replace(path)
        committed = True
    except OSError:
        for path in outputs:
            backup_path = backups.get(path)
            if backup_path is not None and backup_path.exists():
                path.unlink(missing_ok=True)
                backup_path.replace(path)
            elif path not in existing:
                path.unlink(missing_ok=True)
        raise
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)
        if committed:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive asset_manifest.yaml and layer_map.yaml from the canonical queue."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        queue = load_yaml_mapping(args.queue)
        from tools.asset_generation_queue_validator import (
            load_feedback_documents,
            validate_asset_generation_queue,
        )

        feedback_documents = load_feedback_documents(queue, base_dir=base_dir)
        queue_report = validate_asset_generation_queue(
            queue,
            feedback_documents=feedback_documents,
            base_dir=base_dir,
        )
        if not queue_report.valid:
            details = "; ".join(issue.format() for issue in queue_report.issues)
            raise ValueError(f"queue is invalid: {details}")
        queue_ref = args.queue.as_posix()
        manifest = derive_asset_manifest(queue, queue_ref=queue_ref)
        layer_map = derive_layer_map(queue, queue_ref=queue_ref)
        manifest_report = validate_asset_manifest(manifest)
        layer_issues = validate_layer_map(layer_map)
        if not manifest_report.valid:
            details = "; ".join(issue.format() for issue in manifest_report.errors)
            raise ValueError(f"derived manifest is invalid: {details}")
        if layer_issues:
            details = "; ".join(issue.format() for issue in layer_issues)
            raise ValueError(f"derived layer map is invalid: {details}")
        derivatives = _require_mapping(queue, "derivatives")
        manifest_path = _resolve_path(base_dir, derivatives.get("asset_manifest"), "asset_manifest")
        layer_map_path = _resolve_path(base_dir, derivatives.get("layer_map"), "layer_map")
    except (FileNotFoundError, FileExistsError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "queue": queue_ref,
        "manifest": str(manifest_path),
        "layer_map": str(layer_map_path),
        "asset_count": len(manifest["parts"]),
    }
    if args.execute:
        try:
            _write_yaml_pair(
                manifest_path,
                manifest,
                layer_map_path,
                layer_map,
                force=args.force,
            )
        except (FileExistsError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
