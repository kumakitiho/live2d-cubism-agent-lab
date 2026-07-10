from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import require_output_suffix, resolve_inside_base
from tools.backends.segmentation.integrity import canonical_mapping_sha256, file_sha256
from tools.mask_derivation_ranker import verify_derivation_inputs
from tools.segmentation_assignment_planner import render_queue_with_selected_updates

MASK_FIELDS = {
    "protect": "protect_mask",
    "edge_extension": "edge_extension_mask",
    "inpaint": "inpaint_mask",
}
ALLOWED_UPDATES = {
    *MASK_FIELDS.values(),
    "mask_derivation_run_id",
    "mask_derivation_confidence",
    "mask_derivation_status",
}


def _load_queue_snapshot(path: Path) -> tuple[dict[str, Any], bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"queue not found: {path}")
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("queue YAML root must be a mapping")
    return data, raw, file_sha256(path)


def _candidate_index(result: Mapping[str, Any]) -> dict[str, tuple[str, str, Mapping[str, Any]]]:
    raw_layers = result.get("layers")
    if not isinstance(raw_layers, list):
        raise ValueError("mask derivation result layers must be a list")
    index: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
    for layer in raw_layers:
        if not isinstance(layer, Mapping):
            continue
        layer_id = str(layer.get("layer_id"))
        candidates = layer.get("candidates")
        if not isinstance(candidates, Mapping):
            continue
        for mask_type, raw_candidate in candidates.items():
            if mask_type not in MASK_FIELDS or not isinstance(raw_candidate, Mapping):
                continue
            candidate_id = raw_candidate.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            if candidate_id in index:
                raise ValueError(f"duplicate mask candidate ID: {candidate_id}")
            index[candidate_id] = (layer_id, str(mask_type), raw_candidate)
    return index


def _approved_candidate_ids(review: Mapping[str, Any]) -> set[str]:
    layers = review.get("layers")
    if not isinstance(layers, list):
        return set()
    return {
        str(candidate_id)
        for layer in layers
        if isinstance(layer, Mapping) and layer.get("status") == "approved"
        for candidate_id in (
            layer.get("selected", {}).values() if isinstance(layer.get("selected"), Mapping) else []
        )
        if candidate_id is not None
    }


def apply_review_plan(
    queue: Mapping[str, Any],
    review: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    if review.get("review_status") != "approved":
        raise ValueError("review plan must have review_status: approved before apply")
    if queue.get("project") != review.get("project") or queue.get("project") != result.get(
        "project"
    ):
        raise ValueError("queue, review plan, and derivation result projects must match")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip() or review.get("run_id") != run_id:
        raise ValueError("review plan and derivation result run IDs must match")
    derived_from = review.get("derived_from")
    if not isinstance(derived_from, Mapping):
        raise ValueError("review plan derived_from must be a mapping")
    if canonical_mapping_sha256(queue) != derived_from.get("canonical_queue_content_sha256"):
        raise ValueError("review plan was approved for different canonical queue content")
    raw_assets = queue.get("assets")
    if not isinstance(raw_assets, list):
        raise ValueError("queue assets must be a list")
    updated = deepcopy(dict(queue))
    updated_assets = updated.get("assets")
    assert isinstance(updated_assets, list)
    asset_by_id = {
        str(asset.get("layer_id")): asset for asset in updated_assets if isinstance(asset, dict)
    }
    candidates = _candidate_index(result)
    raw_layers = review.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("review plan layers must be a non-empty list")
    approved_layers: set[str] = set()
    selected_ids: set[str] = set()
    for index, layer in enumerate(raw_layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"review layers[{index}] must be a mapping")
        layer_id = layer.get("layer_id")
        if not isinstance(layer_id, str) or layer_id not in asset_by_id:
            raise ValueError(f"review layers[{index}] references an unknown layer")
        if layer.get("status") != "approved":
            continue
        if layer.get("requires_review") is not False:
            raise ValueError(f"approved layer {layer_id} must set requires_review: false")
        if layer_id in approved_layers:
            raise ValueError(f"duplicate approved review layer: {layer_id}")
        selected = layer.get("selected")
        if not isinstance(selected, Mapping):
            raise ValueError(f"approved layer {layer_id} selected must be a mapping")
        if not selected.get("protect_candidate_id") or not selected.get(
            "edge_extension_candidate_id"
        ):
            raise ValueError(f"approved layer {layer_id} requires protect and edge candidates")
        target = asset_by_id[layer_id]
        before = deepcopy(target)
        for mask_type, queue_field in MASK_FIELDS.items():
            candidate_id = selected.get(f"{mask_type}_candidate_id")
            if candidate_id is None:
                continue
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(f"approved layer {layer_id} has invalid {mask_type} candidate")
            if candidate_id in selected_ids:
                raise ValueError(f"mask candidate selected more than once: {candidate_id}")
            selected_ids.add(candidate_id)
            found = candidates.get(candidate_id)
            if found is None:
                raise ValueError(f"unknown selected mask candidate: {candidate_id}")
            candidate_layer, candidate_type, candidate = found
            if candidate_layer != layer_id or candidate_type != mask_type:
                raise ValueError(f"candidate {candidate_id} does not match layer/type selection")
            if candidate.get("run_id") != run_id:
                raise ValueError(f"candidate {candidate_id} belongs to a different run ID")
            if candidate.get("status") != "candidate":
                raise ValueError(f"candidate {candidate_id} is unavailable or rejected")
            soft_path = candidate.get("soft_mask_file")
            if not isinstance(soft_path, str) or not soft_path.strip():
                raise ValueError(f"candidate {candidate_id} soft_mask_file is required")
            target[queue_field] = soft_path
        confidence = layer.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"approved layer {layer_id} confidence must be between 0 and 1")
        target["mask_derivation_run_id"] = run_id
        target["mask_derivation_confidence"] = float(confidence)
        target["mask_derivation_status"] = "approved"
        changed = {key for key in set(before) | set(target) if before.get(key) != target.get(key)}
        unexpected = changed - ALLOWED_UPDATES
        if unexpected:
            raise ValueError(f"mask assignment changed unsupported fields: {sorted(unexpected)}")
        if target.get("target_mask") != before.get("target_mask"):
            raise ValueError("mask assignment must not change target_mask")
        if target.get("source_file") != before.get("source_file"):
            raise ValueError("mask assignment must not change source_file")
        approved_layers.add(layer_id)
    if not approved_layers:
        raise ValueError("review plan contains no approved layers to apply")
    return updated, approved_layers


def _atomic_write(path: Path, text: str, *, force: bool) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "mask-assigned queue output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply only approved mask review selections to a new queue candidate."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "apply":
        raw_args = raw_args[1:]
    args = build_parser().parse_args(raw_args)
    try:
        base_dir = args.base_dir.resolve()
        queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
        review_path = resolve_inside_base(base_dir, str(args.review), "mask derivation review")
        output = resolve_inside_base(base_dir, str(args.output), "mask-assigned queue output")
        if output in {queue_path, review_path}:
            raise ValueError("mask-assigned queue output must not overwrite either input")
        queue, raw_queue, queue_sha256 = _load_queue_snapshot(queue_path)
        review = load_yaml_mapping(review_path)
        derived_from = review.get("derived_from")
        if not isinstance(derived_from, Mapping):
            raise ValueError("review plan derived_from must be a mapping")
        if queue_sha256 != derived_from.get("canonical_queue_sha256"):
            raise ValueError("review plan was approved for different canonical queue bytes")
        result_path = resolve_inside_base(
            base_dir,
            str(derived_from.get("mask_derivation_result")),
            "mask derivation result",
        )
        if output == result_path:
            raise ValueError("mask-assigned queue output must not overwrite the derivation result")
        if file_sha256(result_path) != derived_from.get("mask_derivation_result_sha256"):
            raise ValueError("mask derivation result changed after review plan creation")
        result = load_yaml_mapping(result_path)
        selected_candidate_ids = _approved_candidate_ids(review)
        verify_derivation_inputs(
            result,
            base_dir=base_dir,
            candidate_ids=selected_candidate_ids,
        )
        if result.get("canonical_queue") != queue_path.relative_to(base_dir).as_posix():
            raise ValueError("mask derivation result references a different queue path")
        updated, approved_layers = apply_review_plan(queue, review, result)
        rendered = render_queue_with_selected_updates(
            raw_queue.decode("utf-8"),
            updated,
            selected_layer_ids=approved_layers,
        )
        if file_sha256(queue_path) != queue_sha256:
            raise ValueError("canonical queue changed during mask assignment")
        candidate_index = _candidate_index(result)
        candidate_paths = {
            resolve_inside_base(
                base_dir,
                str(candidate_index[candidate_id][2].get("soft_mask_file")),
                "selected candidate mask",
            )
            for candidate_id in selected_candidate_ids
        }
        missing_candidates = sorted(path for path in candidate_paths if not path.is_file())
        if missing_candidates:
            raise FileNotFoundError(
                "selected result candidate artifacts are missing: "
                + ", ".join(str(path) for path in missing_candidates)
            )
        protected = {queue_path, review_path, result_path, *candidate_paths}
        if output.resolve() in {path.resolve() for path in protected}:
            raise ValueError("mask-assigned queue output must not overwrite an input artifact")
        if args.execute:
            _atomic_write(output, rendered, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = {
        "status": "written" if args.execute else "planned",
        "output": str(output),
        "approved_layers": sorted(approved_layers),
        "canonical_queue_modified": False,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
