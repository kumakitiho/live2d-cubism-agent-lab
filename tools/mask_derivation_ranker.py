from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import resolve_inside_base, write_yaml
from tools.backends.segmentation.integrity import file_sha256

MASK_TYPES = ("protect", "edge_extension", "inpaint")


def _required_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def verify_derivation_inputs(
    result: Mapping[str, Any],
    *,
    base_dir: Path,
    candidate_ids: set[str] | None = None,
) -> None:
    queue = resolve_inside_base(
        base_dir,
        str(result.get("canonical_queue")),
        "canonical queue",
    )
    source = resolve_inside_base(
        base_dir,
        str(result.get("source_image")),
        "source image",
    )
    if file_sha256(queue) != _required_digest(
        result.get("canonical_queue_sha256"), "canonical_queue_sha256"
    ):
        raise ValueError("mask derivation result references changed canonical queue bytes")
    if file_sha256(source) != _required_digest(
        result.get("source_image_sha256"), "source_image_sha256"
    ):
        raise ValueError("mask derivation result references changed source image bytes")
    input_masks = result.get("input_masks")
    if not isinstance(input_masks, list) or not input_masks:
        raise ValueError("mask derivation result input_masks must be a non-empty list")
    for index, item in enumerate(input_masks):
        if not isinstance(item, Mapping):
            raise ValueError(f"input_masks[{index}] must be a mapping")
        path = resolve_inside_base(base_dir, str(item.get("path")), "input target/context mask")
        expected = _required_digest(item.get("sha256"), "input mask sha256")
        if file_sha256(path) != expected:
            raise ValueError(f"input target/context mask changed: {item.get('layer_id')}")
    layers = result.get("layers")
    if not isinstance(layers, list):
        raise ValueError("mask derivation result layers must be a list")
    artifact_fields = (
        ("soft_mask_file", "soft_mask_sha256"),
        ("binary_mask_file", "binary_mask_sha256"),
        ("preview_file", "preview_sha256"),
    )
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("mask derivation layers entries must be mappings")
        candidates = layer.get("candidates")
        if not isinstance(candidates, Mapping):
            continue
        for candidate in candidates.values():
            if not isinstance(candidate, Mapping) or "candidate_id" not in candidate:
                continue
            candidate_id = str(candidate["candidate_id"])
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            for path_field, digest_field in artifact_fields:
                value = candidate.get(path_field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"candidate {path_field} must be a non-empty string")
                path = resolve_inside_base(base_dir, value, f"candidate {path_field}")
                expected = _required_digest(candidate.get(digest_field), digest_field)
                if file_sha256(path) != expected:
                    raise ValueError(f"candidate artifact changed after derivation: {candidate_id}")


def _selected_candidate(candidate: object) -> Mapping[str, Any] | None:
    if not isinstance(candidate, Mapping) or "candidate_id" not in candidate:
        return None
    if candidate.get("status") == "rejected":
        return None
    return candidate


def build_review_plan(
    result: Mapping[str, Any],
    *,
    result_ref: str,
    result_sha256: str,
) -> dict[str, Any]:
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("mask derivation run_id must be a non-empty string")
    layers = result.get("layers")
    if not isinstance(layers, list):
        raise ValueError("mask derivation layers must be a list")
    review_layers: list[dict[str, Any]] = []
    for index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, Mapping):
            raise ValueError(f"layers[{index}] must be a mapping")
        layer_id = raw_layer.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"layers[{index}].layer_id must be a non-empty string")
        candidates = raw_layer.get("candidates")
        if not isinstance(candidates, Mapping):
            candidates = {}
        selected: dict[str, str | None] = {}
        confidences: list[float] = []
        derivation: dict[str, Any] = {}
        review_reasons = ["human_mask_approval_required"]
        for mask_type in MASK_TYPES:
            candidate = _selected_candidate(candidates.get(mask_type))
            selected[f"{mask_type}_candidate_id"] = (
                str(candidate["candidate_id"]) if candidate else None
            )
            if candidate:
                value = candidate.get("confidence")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    confidences.append(float(value))
                derivation[mask_type] = {
                    "method": candidate.get("method"),
                    "parameters": candidate.get("parameters", {}),
                    "reasons": candidate.get("derivation_reasons", []),
                    "adjacent_layers": candidate.get("adjacent_layers", []),
                }
                review_reasons.extend(str(value) for value in candidate.get("review_reasons", []))
            else:
                unavailable = candidates.get(mask_type)
                if isinstance(unavailable, Mapping):
                    reason = unavailable.get("reason")
                    if isinstance(reason, str):
                        review_reasons.append(reason)
                derivation[mask_type] = {"status": "unavailable_or_rejected"}
        conflicts = raw_layer.get("conflicts", [])
        if isinstance(conflicts, list):
            review_reasons.extend(
                str(conflict.get("type")) for conflict in conflicts if isinstance(conflict, Mapping)
            )
        review_layers.append(
            {
                "layer_id": layer_id,
                "selected": selected,
                "confidence": round(min(confidences), 6) if confidences else 0.0,
                "status": "needs_review",
                "requires_review": True,
                "review_reasons": list(dict.fromkeys(review_reasons)),
                "derivation": derivation,
            }
        )
    return {
        "schema_version": 1,
        "project": result.get("project"),
        "run_id": run_id,
        "review_status": "pending",
        "derived_from": {
            "mask_derivation_result": result_ref,
            "mask_derivation_result_sha256": result_sha256,
            "canonical_queue": result.get("canonical_queue"),
            "canonical_queue_sha256": result.get("canonical_queue_sha256"),
            "canonical_queue_content_sha256": result.get("canonical_queue_content_sha256"),
            "source_image": result.get("source_image"),
            "source_image_sha256": result.get("source_image_sha256"),
        },
        "layers": review_layers,
        "summary": {
            "layers": len(review_layers),
            "selected_candidates": sum(
                value is not None for layer in review_layers for value in layer["selected"].values()
            ),
            "all_layers_require_review": True,
            "canonical_queue_modified": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank mask candidates into a human review plan.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base_dir = args.base_dir.resolve()
        result_path = resolve_inside_base(base_dir, str(args.result), "mask derivation result")
        output = resolve_inside_base(base_dir, str(args.output), "mask derivation review plan")
        if output == result_path:
            raise ValueError("review output must not overwrite the derivation result")
        result = load_yaml_mapping(result_path)
        verify_derivation_inputs(result, base_dir=base_dir)
        result_ref = result_path.relative_to(base_dir).as_posix()
        plan = build_review_plan(
            result,
            result_ref=result_ref,
            result_sha256=file_sha256(result_path),
        )
        protected = {
            result_path,
            resolve_inside_base(base_dir, str(result.get("canonical_queue")), "canonical queue"),
            resolve_inside_base(base_dir, str(result.get("source_image")), "source image"),
        }
        if output.resolve() in {path.resolve() for path in protected}:
            raise ValueError("review output must not overwrite a derivation input")
        if args.execute:
            write_yaml(output, plan, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = {
        "status": "written" if args.execute else "planned",
        "output": str(output),
        "run_id": plan["run_id"],
        "layers": plan["summary"]["layers"],
        "requires_review": True,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
