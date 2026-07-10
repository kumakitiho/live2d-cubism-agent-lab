from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    GENERATION_METHODS,
    referenced_artifact_paths,
    require_output_suffix,
    resolve_inside_base,
    validate_asset_quality,
)
from tools.asset_queue_builder import normalize_queue_ref, write_yaml_pair


def next_generation_method(current: str) -> str:
    if current not in GENERATION_METHODS:
        raise ValueError(f"unknown generation method: {current}")
    index = GENERATION_METHODS.index(current)
    return GENERATION_METHODS[min(index + 1, len(GENERATION_METHODS) - 1)]


FAILED_CHECK_METHOD_HINTS = {
    "white_halo_px": "extract_and_edge_repair",
    "preserve_region_difference_score": "extract",
    "edge_extension_difference_score": "extract_and_edge_repair",
    "edge_continuity_score": "inpaint",
    "boundary_color_difference_score": "inpaint",
    "visual_reconstruction_difference_score": "inpaint",
    "transparent_hole_px": "transparency_fill",
    "overlap_deficit_px": "extract_and_edge_repair",
}

INPAINT_RETRY_CHECKS = {
    "inpaint_outside_difference_score",
}
INPAINT_RERANK_CHECKS = {
    "edge_continuity_score",
    "boundary_color_difference_score",
}


def select_generation_method(current: str, failed_checks: list[str]) -> str:
    if current not in GENERATION_METHODS:
        raise ValueError(f"unknown generation method: {current}")
    if "preserve_region_difference_score" in failed_checks:
        return "extract"
    if INPAINT_RETRY_CHECKS.intersection(failed_checks):
        return current if current in {"inpaint", "redraw"} else "inpaint"
    if "transparent_hole_px" in failed_checks:
        return "transparency_fill"
    if "white_halo_px" in failed_checks and current == "extract":
        return "extract_and_edge_repair"
    if "edge_extension_difference_score" in failed_checks or "overlap_deficit_px" in failed_checks:
        return "extract_and_edge_repair"
    if INPAINT_RERANK_CHECKS.intersection(failed_checks):
        return current if current in {"inpaint", "redraw"} else "inpaint"
    current_index = GENERATION_METHODS.index(current)
    hinted_methods = {
        FAILED_CHECK_METHOD_HINTS[check]
        for check in failed_checks
        if check in FAILED_CHECK_METHOD_HINTS
    }
    forward_hints = [
        method for method in hinted_methods if GENERATION_METHODS.index(method) > current_index
    ]
    if forward_hints:
        return max(forward_hints, key=GENERATION_METHODS.index)
    return next_generation_method(current)


def requested_refinement_action(
    current: str,
    target: str,
    failed_checks: list[str],
) -> str:
    if "preserve_region_difference_score" in failed_checks:
        return "reset_from_source_and_reextract"
    if "inpaint_outside_difference_score" in failed_checks:
        return (
            "retry_same_inpaint_with_mask_compositing"
            if current == target == "inpaint"
            else f"run_{target}_with_corrected_mask_compositing"
        )
    if "transparent_hole_px" in failed_checks:
        return "fill_required_target_transparency"
    if "edge_extension_difference_score" in failed_checks or "overlap_deficit_px" in failed_checks:
        return "rerun_extract_and_edge_repair"
    if "white_halo_px" in failed_checks:
        return (
            "rerun_extract_and_edge_repair"
            if target == "extract_and_edge_repair"
            else f"advance_to_{target}_after_edge_repair_exhausted"
        )
    if INPAINT_RERANK_CHECKS.intersection(failed_checks):
        return f"regenerate_or_rerank_{target}_candidate"
    return "regenerate_failed_part_only"


def failed_layer_ids(quality: Mapping[str, Any]) -> set[str]:
    parts = quality.get("parts")
    if not isinstance(parts, list):
        raise ValueError("asset quality parts must be a list")
    return {
        str(part["layer_id"])
        for part in parts
        if isinstance(part, Mapping) and part.get("quality_status") == "fail"
    }


def build_refinement_plan(
    queue: Mapping[str, Any],
    quality: Mapping[str, Any],
    *,
    quality_ref: str,
    queue_ref: str | None = None,
) -> dict[str, Any]:
    if quality.get("project") != queue.get("project"):
        raise ValueError("asset quality project must match queue project")
    quality_derived = quality.get("derived_from")
    if queue_ref is not None:
        if not isinstance(quality_derived, Mapping):
            raise ValueError("asset quality derived_from is required")
        if quality_derived.get("asset_generation_queue") != queue_ref:
            raise ValueError("asset quality must reference the queue being refined")
    failed = failed_layer_ids(quality)
    summary = quality.get("summary")
    if isinstance(summary, Mapping) and summary.get("result") == "fail" and not failed:
        raise ValueError(
            "global reconstruction failed without an attributable part; manual review required"
        )
    assets = queue.get("assets")
    quality_parts = quality.get("parts")
    if not isinstance(assets, list) or not isinstance(quality_parts, list):
        raise ValueError("queue assets and quality parts are required")
    asset_by_id = {
        str(asset.get("layer_id")): asset for asset in assets if isinstance(asset, Mapping)
    }
    quality_by_id = {
        str(part.get("layer_id")): part for part in quality_parts if isinstance(part, Mapping)
    }
    expected_quality_ids = {
        layer_id
        for layer_id, asset in asset_by_id.items()
        if asset.get("include_in_import") is True
    }
    actual_quality_ids = set(quality_by_id)
    if actual_quality_ids != expected_quality_ids:
        missing = sorted(expected_quality_ids - actual_quality_ids)
        extra = sorted(actual_quality_ids - expected_quality_ids)
        raise ValueError(
            f"asset quality must cover every import asset exactly; missing={missing}, extra={extra}"
        )
    unknown = failed - set(asset_by_id)
    if unknown:
        raise ValueError(f"quality report contains unknown failed parts: {sorted(unknown)}")
    jobs: list[dict[str, Any]] = []
    for layer_id in sorted(failed):
        asset = asset_by_id[layer_id]
        quality_part = quality_by_id[layer_id]
        current = asset.get("generation_method")
        attempts = asset.get("refinement_attempts")
        if not isinstance(current, str) or not isinstance(attempts, int):
            raise ValueError(f"asset {layer_id} requires generation_method/refinement_attempts")
        if attempts >= 3:
            raise ValueError(
                f"asset {layer_id} reached the automatic refinement limit; manual review required"
            )
        failed_checks = deepcopy(quality_part.get("failed_checks", []))
        if not isinstance(failed_checks, list) or not all(
            isinstance(check, str) for check in failed_checks
        ):
            raise ValueError(f"quality part {layer_id} requires failed_checks")
        target_method = select_generation_method(current, failed_checks)
        jobs.append(
            {
                "layer_id": layer_id,
                "from_generation_method": current,
                "to_generation_method": target_method,
                "refinement_attempt": attempts + 1,
                "failed_checks": failed_checks,
                "requested_action": requested_refinement_action(
                    current,
                    target_method,
                    failed_checks,
                ),
            }
        )
    return {
        "schema_version": 1,
        "project": queue.get("project"),
        "asset_quality": quality_ref,
        "generation_priority": list(GENERATION_METHODS),
        "jobs": jobs,
        "summary": {"failed_parts": len(jobs), "requeue_only_failed_parts": True},
    }


def apply_refinement_plan(
    queue: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(queue))
    jobs = plan.get("jobs")
    assets = result.get("assets")
    queue_jobs = result.get("jobs")
    if not isinstance(jobs, list) or not isinstance(assets, list):
        raise ValueError("refinement jobs and queue assets are required")
    refinements = {str(job.get("layer_id")): job for job in jobs if isinstance(job, Mapping)}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        layer_id = asset.get("layer_id")
        refinement = refinements.get(str(layer_id))
        if refinement is None:
            continue
        asset["generation_method"] = refinement["to_generation_method"]
        if asset["generation_method"] == "inpaint":
            asset["inferred"] = True
            asset["review_required"] = True
        elif asset["generation_method"] == "redraw":
            asset["review_required"] = True
        asset["quality_status"] = "pending"
        asset["refinement_attempts"] = refinement["refinement_attempt"]
        asset["readiness"] = "planned"
    if isinstance(queue_jobs, list):
        failed = set(refinements)
        asset_methods = {
            str(asset.get("layer_id")): asset.get("generation_method")
            for asset in assets
            if isinstance(asset, Mapping)
        }
        generation_operation_names = set(GENERATION_METHODS) | {"mask_extract"}
        for job in queue_jobs:
            if not isinstance(job, dict):
                continue
            targets = job.get("targets")
            if not isinstance(targets, list) or not (failed & set(targets)):
                continue
            operations = job.get("operations")
            preserved_operations = (
                [
                    operation
                    for operation in operations
                    if isinstance(operation, str) and operation not in generation_operation_names
                ]
                if isinstance(operations, list)
                else []
            )
            target_methods = [
                asset_methods[str(target)]
                for target in targets
                if asset_methods.get(str(target)) in GENERATION_METHODS
            ]
            job["operations"] = list(dict.fromkeys([*preserved_operations, *target_methods]))
            job["status"] = "planned"
            validation = job.get("validation")
            if isinstance(validation, dict):
                for key in validation:
                    validation[key] = False
    merge_gate = result.get("merge_gate")
    if refinements and isinstance(merge_gate, dict):
        validation = merge_gate.get("validation")
        if isinstance(validation, dict):
            for key in validation:
                validation[key] = False
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Requeue only parts that failed the pre-Cubism asset quality gate."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("quality", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refined-queue-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        queue_path = resolve_inside_base(base_dir, str(args.queue), "queue")
        quality_path = resolve_inside_base(base_dir, str(args.quality), "asset quality")
        output_path = resolve_inside_base(base_dir, str(args.output), "refinement plan output")
        refined_queue_path = resolve_inside_base(
            base_dir,
            str(args.refined_queue_output),
            "refined queue output",
        )
        require_output_suffix(output_path, {".yaml", ".yml"}, "refinement plan output")
        require_output_suffix(refined_queue_path, {".yaml", ".yml"}, "refined queue output")
        queue = load_yaml_mapping(queue_path)
        quality = load_yaml_mapping(quality_path)
        protected_inputs = referenced_artifact_paths(
            queue,
            base_dir,
            document_path=queue_path,
        )
        protected_inputs.add(quality_path)
        if output_path == refined_queue_path:
            raise ValueError("refinement outputs must use different paths")
        if output_path in protected_inputs or refined_queue_path in protected_inputs:
            raise ValueError(
                "refinement outputs must not overwrite queue, quality, source, character spec, "
                "feedback, part, mask, or canonical derivatives"
            )
        quality_issues = validate_asset_quality(quality)
        if quality_issues:
            raise ValueError("; ".join(issue.format() for issue in quality_issues))
        queue_ref = normalize_queue_ref(queue_path, base_dir)
        quality_ref = normalize_queue_ref(quality_path, base_dir)
        plan = build_refinement_plan(
            queue,
            quality,
            quality_ref=quality_ref,
            queue_ref=queue_ref,
        )
        refined_queue = apply_refinement_plan(queue, plan)
        if args.execute:
            write_yaml_pair(
                output_path,
                plan,
                refined_queue_path,
                refined_queue,
                force=args.force,
            )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    result = {
        "status": "written" if args.execute else "planned",
        "failed_parts": plan["summary"]["failed_parts"],
        "requeued_layers": [job["layer_id"] for job in plan["jobs"]],
        "output": str(output_path),
        "refined_queue_output": str(refined_queue_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
