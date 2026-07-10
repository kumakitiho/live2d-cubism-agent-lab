from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import load_rgba, resolve_inside_base
from tools.generative_inpainter import (
    atomic_write_yaml,
    file_sha256,
    validate_inpainting_result,
)


def _number(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"quality_metrics.{key} must be numeric")
    return float(value)


def candidate_rank_key(candidate: Mapping[str, Any]) -> tuple[float | int | str, ...]:
    metrics = candidate.get("quality_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("candidate quality_metrics must be a mapping")
    seed = candidate.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("candidate seed must be an integer")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a string")
    return (
        _number(metrics, "edge_continuity_score"),
        _number(metrics, "boundary_color_difference_score"),
        _number(metrics, "white_halo_px"),
        _number(metrics, "alpha_continuity_score"),
        _number(metrics, "surrounding_palette_consistency_score"),
        _number(metrics, "visual_reconstruction_score"),
        seed,
        candidate_id,
    )


def strict_candidate_rejections(candidate: Mapping[str, Any]) -> list[str]:
    metrics = candidate.get("quality_metrics")
    if not isinstance(metrics, Mapping):
        return ["schema_invalid"]
    reasons: list[str] = []
    exact_zero_checks = {
        "preserve_region_difference_score": "protect_region_changed",
        "protect_difference_px": "protect_region_changed",
        "inpaint_outside_difference_score": "inpaint_mask_outside_changed",
        "inpaint_outside_difference_px": "inpaint_mask_outside_changed",
        "transparent_hole_px": "required_target_coverage_missing",
        "overlap_deficit_px": "required_target_coverage_missing",
    }
    for metric, reason in exact_zero_checks.items():
        if _number(metrics, metric) != 0.0:
            reasons.append(reason)
    for metric, reason in (
        ("canvas_match", "canvas_mismatch"),
        ("origin_match", "origin_mismatch"),
        ("alpha_valid", "alpha_invalid"),
    ):
        if metrics.get(metric) is not True:
            reasons.append(reason)
    return sorted(set(reasons))


def rank_candidates(
    result: Mapping[str, Any],
    *,
    result_ref: str = "<in-memory>",
    result_sha256: str | None = None,
) -> dict[str, Any]:
    errors = validate_inpainting_result(result)
    if errors:
        raise ValueError("invalid inpainting result: " + "; ".join(errors))
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    passing: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        assert isinstance(candidate, Mapping)
        reasons = candidate.get("rejection_reasons")
        strict_reasons = strict_candidate_rejections(candidate)
        if candidate.get("quality_status") == "pass" and reasons == [] and not strict_reasons:
            passing.append(candidate)
        else:
            rejected.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "rejection_reasons": sorted(
                        set(deepcopy(reasons) if isinstance(reasons, list) else [])
                        | set(strict_reasons)
                    ),
                }
            )
    if not passing:
        raise ValueError("all inpainting candidates failed; no selection was created")
    ranked = sorted(passing, key=candidate_rank_key)
    best = deepcopy(dict(ranked[0]))
    effective_result_sha256 = result_sha256 or hashlib.sha256(
        json.dumps(dict(result), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    ranking = [
        {
            "rank": index,
            "candidate_id": candidate["candidate_id"],
            "seed": candidate["seed"],
            "rank_key": list(candidate_rank_key(candidate)),
        }
        for index, candidate in enumerate(ranked, start=1)
    ]
    return {
        "schema_version": 1,
        "project": result["project"],
        "run_id": result["run_id"],
        "layer_id": result["layer_id"],
        "status": "selected",
        "derived_from_result": result_ref,
        "derived_result_sha256": effective_result_sha256,
        "review_required": True,
        "review": {
            "status": "pending",
            "reviewer": None,
            "notes": "Set status to approved only after visual review of the preview.",
        },
        "selected_candidate": best,
        "ranking": ranking,
        "rejected_candidates": rejected,
        "queue_update_plan": {
            "canonical_queue_modified": False,
            "layer_id": result["layer_id"],
            "allowed_fields": [
                "source_file",
                "generation_method",
                "inferred",
                "review_required",
                "quality_status",
                "readiness",
                "generation_backend",
                "model_id",
                "model_revision",
                "seed",
                "inpainting_run_id",
            ],
        },
    }


def apply_selection_to_queue(
    queue: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    if selection.get("status") != "selected" or selection.get("review_required") is not True:
        raise ValueError("selection must be selected and review_required")
    review = selection.get("review")
    if not isinstance(review, Mapping) or review.get("status") != "approved":
        raise ValueError("selection review.status must be approved before queue application")
    if not isinstance(review.get("reviewer"), str) or not review.get("reviewer", "").strip():
        raise ValueError("approved selection review.reviewer is required")
    if not isinstance(review.get("notes"), str) or not review.get("notes", "").strip():
        raise ValueError("approved selection review.notes is required")
    if queue.get("project") != selection.get("project"):
        raise ValueError("selection project must match canonical queue project")
    layer_id = selection.get("layer_id")
    if not isinstance(layer_id, str) or not layer_id:
        raise ValueError("selection layer_id is required")
    selected = selection.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("selection selected_candidate must be a mapping")
    if selected.get("requires_review") is not True:
        raise ValueError("selected candidate must remain review-required")
    if selected.get("quality_status") != "pass" or selected.get("rejection_reasons") != []:
        raise ValueError("selected candidate must pass all automatic quality gates")
    if strict_candidate_rejections(selected):
        raise ValueError("selected candidate violates a strict preservation gate")
    if not isinstance(selected.get("backend"), str) or not selected.get("backend"):
        raise ValueError("selected candidate backend is required")
    if not isinstance(selected.get("seed"), int) or isinstance(selected.get("seed"), bool):
        raise ValueError("selected candidate seed must be an integer")
    source_file = selected.get("output_file")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("selected candidate output_file is required")
    source_path = Path(source_file)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError("selected candidate output_file must be a safe relative path")
    assets = queue.get("assets")
    if not isinstance(assets, list):
        raise ValueError("canonical queue assets must be a list")
    updated = deepcopy(dict(queue))
    updated_assets = updated.get("assets")
    assert isinstance(updated_assets, list)
    matches = 0
    for asset in updated_assets:
        if not isinstance(asset, dict) or asset.get("layer_id") != layer_id:
            continue
        matches += 1
        asset.update(
            {
                "source_file": source_file,
                "generation_method": "inpaint",
                "inferred": True,
                "review_required": True,
                "quality_status": "pass",
                "readiness": "generated",
                "generation_backend": selected.get("backend"),
                "model_id": selected.get("model_id"),
                "model_revision": selected.get("model_revision"),
                "seed": selected.get("seed"),
                "inpainting_run_id": selection.get("run_id"),
            }
        )
    if matches != 1:
        raise ValueError(f"selection layer_id must match exactly one queue asset: {layer_id}")
    return updated


def verify_selection_against_result(
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    result_sha256: str,
) -> None:
    result_errors = validate_inpainting_result(result)
    if result_errors:
        raise ValueError("invalid derived inpainting result: " + "; ".join(result_errors))
    if selection.get("derived_result_sha256") != result_sha256:
        raise ValueError("selection derived_result_sha256 does not match its result file")
    for field in ("project", "run_id", "layer_id"):
        if selection.get(field) != result.get(field):
            raise ValueError(f"selection {field} must match its derived result")
    update_plan = selection.get("queue_update_plan")
    if (
        not isinstance(update_plan, Mapping)
        or update_plan.get("canonical_queue_modified") is not False
        or update_plan.get("layer_id") != selection.get("layer_id")
    ):
        raise ValueError("selection queue_update_plan is invalid")
    selected = selection.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("selection selected_candidate must be a mapping")
    candidates = result.get("candidates")
    assert isinstance(candidates, list)
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == selected.get("candidate_id")
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(selected):
        raise ValueError("selection candidate must exactly match its derived result")


def verify_candidate_artifacts(result: Mapping[str, Any], base_dir: Path) -> None:
    canvas = result.get("canvas")
    if not isinstance(canvas, Mapping):
        raise ValueError("result canvas must be a mapping")
    expected_size = (int(canvas["width"]), int(canvas["height"]))
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("result candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("result candidate must be a mapping")
        candidate_id = candidate.get("candidate_id")
        for path_field, digest_field in (
            ("output_file", "output_sha256"),
            ("preview_file", "preview_sha256"),
        ):
            value = candidate.get(path_field)
            if not isinstance(value, str):
                raise ValueError(f"candidate {candidate_id} {path_field} is required")
            path = resolve_inside_base(base_dir, value, f"candidate {path_field}")
            if file_sha256(path) != candidate.get(digest_field):
                raise ValueError(f"candidate {candidate_id} {path_field} digest mismatch")
            image = load_rgba(path)
            if path_field == "output_file" and image.size != expected_size:
                raise ValueError(f"candidate {candidate_id} output canvas mismatch")


def _rank_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank passing inpainting candidates (dry-run by default)."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an approved selection to a new queue candidate (never in place)."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _print_report(report: Mapping[str, Any], use_json: bool) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2) if use_json else dict(report))


def _rank_main(argv: Sequence[str]) -> int:
    args = _rank_parser().parse_args(list(argv))
    base_dir = args.base_dir.resolve()
    try:
        result_path = resolve_inside_base(base_dir, str(args.result), "inpainting result")
        output_path = resolve_inside_base(base_dir, str(args.output), "selection output")
        if output_path == result_path:
            raise ValueError("selection output must not overwrite the inpainting result")
        result = load_yaml_mapping(result_path)
        request_ref = result.get("derived_from_request")
        if not isinstance(request_ref, str) or not request_ref.strip():
            raise ValueError("inpainting result derived_from_request is required")
        request_path = resolve_inside_base(base_dir, request_ref, "derived inpainting request")
        if output_path == request_path:
            raise ValueError("selection output must not overwrite the inpainting request")
        selection = rank_candidates(
            result,
            result_ref=result_path.resolve().relative_to(base_dir).as_posix(),
            result_sha256=file_sha256(result_path),
        )
        verify_candidate_artifacts(result, base_dir)
        if args.execute:
            atomic_write_yaml(output_path, selection, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    _print_report(
        {
            "status": "written" if args.execute else "planned",
            "output": str(output_path),
            "selected_candidate": selection["selected_candidate"]["candidate_id"],
            "review_required": True,
            "review_status": "pending",
        },
        args.json,
    )
    return 0


def _apply_main(argv: Sequence[str]) -> int:
    args = _apply_parser().parse_args(list(argv))
    base_dir = args.base_dir.resolve()
    try:
        queue_path = resolve_inside_base(base_dir, str(args.queue), "canonical queue")
        selection_path = resolve_inside_base(base_dir, str(args.selection), "selection")
        output_path = resolve_inside_base(base_dir, str(args.output), "queue candidate output")
        if output_path in {queue_path, selection_path}:
            raise ValueError("queue candidate output must not overwrite queue or selection")
        queue = load_yaml_mapping(queue_path)
        selection = load_yaml_mapping(selection_path)
        result_ref = selection.get("derived_from_result")
        if not isinstance(result_ref, str) or not result_ref.strip():
            raise ValueError("selection derived_from_result is required")
        result_path = resolve_inside_base(base_dir, result_ref, "derived inpainting result")
        result = load_yaml_mapping(result_path)
        request_ref = result.get("derived_from_request")
        if not isinstance(request_ref, str) or not request_ref.strip():
            raise ValueError("inpainting result derived_from_request is required")
        request_path = resolve_inside_base(base_dir, request_ref, "derived inpainting request")
        if output_path in {result_path, request_path}:
            raise ValueError(
                "queue candidate output must not overwrite inpainting request or result"
            )
        verify_selection_against_result(
            selection,
            result,
            result_sha256=file_sha256(result_path),
        )
        verify_candidate_artifacts(result, base_dir)
        updated = apply_selection_to_queue(queue, selection)
        if args.execute:
            atomic_write_yaml(output_path, updated, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    _print_report(
        {
            "status": "written" if args.execute else "planned",
            "canonical_queue_modified": False,
            "output": str(output_path),
            "layer_id": selection.get("layer_id"),
        },
        args.json,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    values = list(argv) if argv is not None else None
    if values is None:
        import sys

        values = sys.argv[1:]
    if values and values[0] == "apply":
        return _apply_main(values[1:])
    return _rank_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
