from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping
from pathlib import Path

import yaml

from tools.asset_pipeline_common import (
    require_output_suffix,
    resolve_inside_base,
    write_yaml,
)
from tools.backends.segmentation.integrity import file_sha256
from tools.mask_derivation.pipeline import DerivationConfig, derive_masks, load_queue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive deterministic review-only protect, edge-extension, and inpaint masks."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--retry-failed-from", type=Path)
    parser.add_argument("--protect-radius", type=int, default=2)
    parser.add_argument("--edge-radius", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=2.0)
    parser.add_argument("--min-island-area", type=int, default=2)
    parser.add_argument("--binary-threshold", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _retry_layers(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("retry result YAML root must be a mapping")
    layers = data.get("layers")
    if not isinstance(layers, list):
        raise ValueError("retry result layers must be a list")
    result = {
        str(layer.get("layer_id"))
        for layer in layers
        if isinstance(layer, Mapping) and layer.get("status") == "failed"
    }
    if not result:
        raise ValueError("retry result contains no failed layers")
    return result


def _selected_layers(args: argparse.Namespace, base_dir: Path) -> set[str] | None:
    explicit = {
        value.strip() for item in args.layer for value in str(item).split(",") if value.strip()
    }
    retry: set[str] = set()
    if args.retry_failed_from:
        retry_path = resolve_inside_base(
            base_dir,
            str(args.retry_failed_from),
            "retry mask derivation result",
        )
        if not retry_path.is_file():
            raise FileNotFoundError(f"retry result not found: {retry_path}")
        retry = _retry_layers(retry_path)
    if explicit and retry:
        selected = explicit & retry
        if not selected:
            raise ValueError("--layer selection contains none of the failed retry layers")
        return selected
    return explicit or retry or None


def _preflight(
    *,
    output: Path,
    image_outputs: set[Path],
    input_paths: set[Path],
    force: bool,
) -> None:
    all_outputs = {output.resolve(), *(path.resolve() for path in image_outputs)}
    if len(all_outputs) != len(image_outputs) + 1:
        raise ValueError("mask derivation outputs contain a path collision")
    collision = all_outputs & {path.resolve() for path in input_paths}
    if collision:
        raise ValueError(
            "mask derivation output must not overwrite canonical inputs: "
            + ", ".join(str(path) for path in sorted(collision))
        )
    existing = [path for path in all_outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing mask derivation outputs without --force: "
            + ", ".join(str(path) for path in sorted(existing))
        )


def _verify_inputs(document: Mapping[str, object], base_dir: Path) -> None:
    queue = resolve_inside_base(base_dir, str(document["canonical_queue"]), "canonical queue")
    source = resolve_inside_base(base_dir, str(document["source_image"]), "source image")
    if file_sha256(queue) != document.get("canonical_queue_sha256"):
        raise ValueError("canonical queue changed during mask derivation")
    if file_sha256(source) != document.get("source_image_sha256"):
        raise ValueError("source image changed during mask derivation")
    input_masks = document.get("input_masks")
    if not isinstance(input_masks, list) or not input_masks:
        raise ValueError("mask derivation result input_masks must be a non-empty list")
    for item in input_masks:
        if not isinstance(item, Mapping):
            raise ValueError("mask derivation input_masks entries must be mappings")
        path = resolve_inside_base(base_dir, str(item.get("path")), "input target/context mask")
        if file_sha256(path) != item.get("sha256"):
            raise ValueError(f"input mask changed during derivation: {item.get('layer_id')}")


def _verify_output_artifacts(document: Mapping[str, object], base_dir: Path) -> None:
    fields = (
        ("soft_mask_file", "soft_mask_sha256"),
        ("binary_mask_file", "binary_mask_sha256"),
        ("preview_file", "preview_sha256"),
    )
    layers = document.get("layers")
    if not isinstance(layers, list):
        raise ValueError("mask derivation result layers must be a list")
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        candidates = layer.get("candidates")
        if not isinstance(candidates, Mapping):
            continue
        for candidate in candidates.values():
            if not isinstance(candidate, Mapping) or "candidate_id" not in candidate:
                continue
            for path_field, digest_field in fields:
                path = resolve_inside_base(
                    base_dir,
                    str(candidate.get(path_field)),
                    f"candidate {path_field}",
                )
                if file_sha256(path) != candidate.get(digest_field):
                    raise ValueError(
                        f"written candidate artifact hash mismatch: {candidate.get('candidate_id')}"
                    )


def _atomic_write_png_bytes(path: Path, content: bytes, *, force: bool) -> None:
    require_output_suffix(path, {".png"}, "PNG output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base_dir = args.base_dir.resolve()
        queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
        output = resolve_inside_base(base_dir, str(args.output), "mask derivation result")
        require_output_suffix(output, {".yaml", ".yml"}, "mask derivation result")
        output_dir_value = args.output_dir or output.parent / "mask-derivation"
        output_dir = resolve_inside_base(
            base_dir,
            str(output_dir_value),
            "mask derivation output directory",
        )
        if output.resolve() == queue_path.resolve():
            raise ValueError("mask derivation result must not overwrite the canonical queue")
        queue = load_queue(queue_path)
        selected = _selected_layers(args, base_dir)
        config = DerivationConfig(
            protect_radius_px=args.protect_radius,
            edge_radius_px=args.edge_radius,
            fine_part_min_area_px=args.min_area,
            max_candidate_area_ratio=args.max_area_ratio,
            min_island_area_px=args.min_island_area,
            binary_threshold=args.binary_threshold,
        )
        config.validate()
        document, artifacts = derive_masks(
            queue,
            queue_path=queue_path,
            base_dir=base_dir,
            output_dir=output_dir,
            config=config,
            run_id=args.run_id,
            layer_ids=selected,
            retain_artifacts=args.execute,
        )
        _preflight(
            output=output,
            image_outputs=set(artifacts.output_paths),
            input_paths=set(artifacts.input_paths),
            force=args.force,
        )
        _verify_inputs(document, base_dir)
        if args.execute:
            try:
                for path in sorted(artifacts.payloads, key=str):
                    _atomic_write_png_bytes(path, artifacts.read_bytes(path), force=args.force)
                _verify_output_artifacts(document, base_dir)
                write_yaml(output, document, force=args.force)
            finally:
                artifacts.close()
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = {
        "status": "written" if args.execute else "planned",
        "output": str(output),
        "run_id": document.get("run_id"),
        "layers_processed": document.get("summary", {}).get("layers_processed", 0)
        if isinstance(document.get("summary"), Mapping)
        else 0,
        "canonical_queue_modified": False,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
