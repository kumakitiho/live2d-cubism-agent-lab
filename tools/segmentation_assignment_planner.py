from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    referenced_artifact_paths,
    require_output_suffix,
    resolve_inside_base,
)
from tools.asset_queue_builder import normalize_queue_ref
from tools.backends.segmentation.integrity import (
    bytes_sha256,
    canonical_mapping_sha256,
    file_sha256,
)

QUEUE_UPDATE_FIELDS = (
    "target_mask",
    "protect_mask",
    "edge_extension_mask",
    "inpaint_mask",
    "segmentation_backend",
    "segmentation_model_id",
    "segmentation_model_revision",
    "segmentation_run_id",
    "segmentation_confidence",
)


def _required_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_file(queue: Mapping[str, Any], base_dir: Path) -> Path:
    source = queue.get("source_image")
    if not isinstance(source, Mapping):
        raise ValueError("queue source_image must be a mapping")
    value = source.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("queue source_image.path must be a non-empty string")
    return resolve_inside_base(base_dir, value, "queue source_image.path")


def _load_queue_snapshot(path: Path) -> tuple[dict[str, Any], bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"queue not found: {path}")
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"queue YAML root must be a mapping: {path}")
    return data, raw_bytes, bytes_sha256(raw_bytes)


def _assets(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_assets = queue.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("queue assets must be a non-empty list")
    assets: list[Mapping[str, Any]] = []
    layer_ids: set[str] = set()
    for index, asset in enumerate(raw_assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"queue assets[{index}] must be a mapping")
        layer_id = asset.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"queue assets[{index}].layer_id must be a non-empty string")
        if layer_id in layer_ids:
            raise ValueError(f"duplicate queue layer_id: {layer_id}")
        layer_ids.add(layer_id)
        assets.append(asset)
    return assets


def _ranked_candidates(ranked: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if ranked.get("status") != "ranked":
        raise ValueError("segmentation candidates must be ranked before assignment planning")
    raw_candidates = ranked.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("ranked segmentation candidates must be a non-empty list")
    candidates: list[Mapping[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"ranked candidates[{index}] must be a mapping")
        candidate_id = candidate.get("candidate_id")
        layer_id = candidate.get("layer_id")
        rank = candidate.get("rank")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"ranked candidates[{index}].candidate_id is required")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        candidate_ids.add(candidate_id)
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"ranked candidates[{index}].layer_id is required")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError(f"candidate {candidate_id} rank must be positive")
        candidates.append(candidate)
    return candidates


def build_assignment_plan(
    queue: Mapping[str, Any],
    ranked: Mapping[str, Any],
    *,
    queue_ref: str = "<in-memory>",
    ranked_ref: str = "<in-memory>",
) -> dict[str, Any]:
    if queue.get("project") != ranked.get("project"):
        raise ValueError("ranked segmentation project must match the queue project")
    ranked_queue_ref = ranked.get("asset_generation_queue")
    if (
        queue_ref != "<in-memory>"
        and isinstance(ranked_queue_ref, str)
        and ranked_queue_ref != queue_ref
    ):
        raise ValueError("ranked segmentation must reference the queue being planned")
    queue_content_sha256 = canonical_mapping_sha256(queue)
    ranked_content_sha256 = _required_sha256(
        ranked.get("asset_generation_queue_content_sha256"),
        "ranked asset_generation_queue_content_sha256",
    )
    if ranked_content_sha256 != queue_content_sha256:
        raise ValueError("ranked segmentation was produced from different queue content")
    queue_file_sha256 = _required_sha256(
        ranked.get("asset_generation_queue_sha256"),
        "ranked asset_generation_queue_sha256",
    )
    source_sha256 = _required_sha256(
        ranked.get("source_image_sha256"),
        "ranked source_image_sha256",
    )
    run_id = ranked.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("ranked segmentation run_id must be a non-empty string")
    assets = _assets(queue)
    asset_by_id = {str(asset["layer_id"]): asset for asset in assets}
    by_layer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in _ranked_candidates(ranked):
        layer_id = str(candidate["layer_id"])
        if layer_id not in asset_by_id:
            raise ValueError(f"ranked segmentation contains unknown layer: {layer_id}")
        by_layer[layer_id].append(candidate)

    assignments: list[dict[str, Any]] = []
    for asset in assets:
        layer_id = str(asset["layer_id"])
        candidates = sorted(
            by_layer.get(layer_id, []),
            key=lambda candidate: (int(candidate["rank"]), str(candidate["candidate_id"])),
        )
        if not candidates:
            continue
        selected = candidates[0]
        if selected.get("rank") != 1:
            raise ValueError(f"layer {layer_id} has no rank-1 candidate")
        confidence = selected.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"candidate {selected['candidate_id']} confidence is required")
        soft_mask = selected.get("soft_mask_file")
        if not isinstance(soft_mask, str) or not soft_mask.strip():
            raise ValueError(f"candidate {selected['candidate_id']} soft_mask_file is required")
        reasons = [str(reason) for reason in selected.get("rejection_reasons", [])]
        assignments.append(
            {
                "layer_id": layer_id,
                "selected_candidate_id": selected["candidate_id"],
                "alternative_candidate_ids": [
                    candidate["candidate_id"] for candidate in candidates[1:]
                ],
                "target_mask": soft_mask,
                "protect_mask": asset.get("protect_mask"),
                "edge_extension_mask": asset.get("edge_extension_mask"),
                "inpaint_mask": asset.get("inpaint_mask"),
                "confidence": float(confidence),
                "status": "needs_review",
                "requires_review": True,
                "review_reasons": list(
                    dict.fromkeys([*reasons, "human_assignment_approval_required"])
                ),
                "derivation": {
                    "target_mask": "selected_observed_soft_segmentation_candidate",
                    "protect_mask": (
                        "retained_existing; derive separately from source-preserve policy"
                    ),
                    "edge_extension_mask": (
                        "retained_existing; derive separately from source boundary overlap"
                    ),
                    "inpaint_mask": (
                        "retained_existing; derive separately from hidden-region generation policy"
                    ),
                    "edge_extension_is_inpaint": False,
                },
                "segmentation_backend": selected.get("source_backend"),
                "segmentation_model_id": selected.get("model_id"),
                "segmentation_model_revision": selected.get("model_revision"),
                "segmentation_run_id": ranked.get("run_id"),
                "segmentation_confidence": float(confidence),
            }
        )
    return {
        "schema_version": 1,
        "project": queue.get("project"),
        "derived_from": {
            "asset_generation_queue": queue_ref,
            "segmentation_ranked": ranked_ref,
            "asset_generation_queue_sha256": queue_file_sha256,
            "asset_generation_queue_content_sha256": queue_content_sha256,
            "source_image_sha256": source_sha256,
        },
        "segmentation_run_id": run_id,
        "review_status": "pending",
        "assignments": assignments,
        "summary": {
            "assignment_count": len(assignments),
            "all_assignments_require_review": True,
            "canonical_queue_modified": False,
        },
    }


def apply_assignment_plan(
    queue: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    if queue.get("project") != plan.get("project"):
        raise ValueError("assignment plan project must match the queue project")
    if plan.get("review_status") != "approved":
        raise ValueError("assignment plan must have review_status: approved before apply")
    derived_from = plan.get("derived_from")
    if not isinstance(derived_from, Mapping):
        raise ValueError("assignment plan derived_from must be a mapping")
    expected_content_sha256 = _required_sha256(
        derived_from.get("asset_generation_queue_content_sha256"),
        "assignment derived_from.asset_generation_queue_content_sha256",
    )
    if canonical_mapping_sha256(queue) != expected_content_sha256:
        raise ValueError("assignment plan was reviewed for different queue content")
    expected_source_sha256 = _required_sha256(
        derived_from.get("source_image_sha256"),
        "assignment derived_from.source_image_sha256",
    )
    if source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("assignment plan was reviewed for different source image bytes")
    plan_run_id = plan.get("segmentation_run_id")
    if not isinstance(plan_run_id, str) or not plan_run_id.strip():
        raise ValueError("assignment plan segmentation_run_id must be a non-empty string")
    raw_assignments = plan.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("assignment plan assignments must be a non-empty list")

    result = deepcopy(dict(queue))
    result_assets = result.get("assets")
    if not isinstance(result_assets, list):
        raise ValueError("queue assets must be a list")
    asset_by_id = {
        str(asset.get("layer_id")): asset for asset in result_assets if isinstance(asset, dict)
    }
    seen_layers: set[str] = set()
    seen_candidates: set[str] = set()
    applied = 0
    for index, assignment in enumerate(raw_assignments):
        if not isinstance(assignment, Mapping):
            raise ValueError(f"assignments[{index}] must be a mapping")
        layer_id = assignment.get("layer_id")
        candidate_id = assignment.get("selected_candidate_id")
        if not isinstance(layer_id, str) or layer_id not in asset_by_id:
            raise ValueError(f"assignments[{index}] references an unknown layer")
        if layer_id in seen_layers:
            raise ValueError(f"duplicate assignment layer_id: {layer_id}")
        seen_layers.add(layer_id)
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"assignments[{index}].selected_candidate_id is required")
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate selected candidate ID: {candidate_id}")
        seen_candidates.add(candidate_id)
        status = assignment.get("status")
        if assignment.get("segmentation_run_id") != plan_run_id:
            raise ValueError(f"assignment {layer_id} segmentation_run_id does not match plan")
        if status != "approved":
            continue
        if assignment.get("requires_review") is not False:
            raise ValueError(
                f"assignment {layer_id} must set requires_review: false after human approval"
            )
        target = asset_by_id[layer_id]
        edge_extension = assignment.get(
            "edge_extension_mask",
            target.get("edge_extension_mask"),
        )
        inpaint = assignment.get("inpaint_mask", target.get("inpaint_mask"))
        if edge_extension == inpaint:
            raise ValueError(
                f"assignment {layer_id} edge_extension_mask must differ from inpaint_mask"
            )
        for field in QUEUE_UPDATE_FIELDS:
            value = assignment.get(field)
            if value is None:
                if field in {"segmentation_model_id", "segmentation_model_revision"}:
                    target[field] = None
                continue
            if field in {
                "target_mask",
                "protect_mask",
                "edge_extension_mask",
                "inpaint_mask",
                "segmentation_backend",
                "segmentation_run_id",
            } and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"assignment {layer_id} {field} must be a non-empty string")
            if field == "segmentation_confidence" and (
                not isinstance(value, (int, float)) or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"assignment {layer_id} segmentation_confidence must be between 0 and 1"
                )
            target[field] = deepcopy(value)
        applied += 1
    if applied == 0:
        raise ValueError("assignment plan contains no approved assignments to apply")
    return result


def _atomic_write_yaml(path: Path, data: Mapping[str, Any], *, force: bool) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "segmentation assignment output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _asset_block_spans(
    queue_text: str,
) -> tuple[list[str], dict[str, tuple[int, int, str]], str]:
    lines = queue_text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in queue_text else "\n"
    assets_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"assets:\s*(?:#.*)?", line.rstrip("\r\n"))
        ),
        None,
    )
    if assets_index is None:
        raise ValueError("queue must use a root block-style assets: list")
    block_end = len(lines)
    for index in range(assets_index + 1, len(lines)):
        content = lines[index].rstrip("\r\n")
        if re.match(r"^[^ \t#-][^:]*:", content):
            block_end = index
            break
    first_item = next(
        (
            (index, match.group(1))
            for index in range(assets_index + 1, block_end)
            if (match := re.match(r"^([ \t]*)-(?:\s|$)", lines[index])) is not None
        ),
        None,
    )
    if first_item is None:
        raise ValueError("queue assets must use a non-empty block-style sequence")
    item_indent = first_item[1]
    item_starts = [
        index
        for index in range(first_item[0], block_end)
        if re.match(rf"^{re.escape(item_indent)}-(?:\s|$)", lines[index]) is not None
    ]
    spans: dict[str, tuple[int, int, str]] = {}
    for position, start in enumerate(item_starts):
        end = item_starts[position + 1] if position + 1 < len(item_starts) else block_end
        snippet = f"assets:{newline}{''.join(lines[start:end])}"
        parsed = yaml.safe_load(snippet)
        if not isinstance(parsed, Mapping):
            raise ValueError("unable to parse an assets item while preserving queue bytes")
        assets = parsed.get("assets")
        if not isinstance(assets, list) or len(assets) != 1 or not isinstance(assets[0], Mapping):
            raise ValueError("each preserved assets block must contain exactly one mapping")
        layer_id = assets[0].get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError("each preserved assets block requires layer_id")
        if layer_id in spans:
            raise ValueError(f"duplicate queue layer_id while preserving bytes: {layer_id}")
        spans[layer_id] = (start, end, item_indent)
    return lines, spans, newline


def render_queue_with_selected_updates(
    queue_text: str,
    updated_queue: Mapping[str, Any],
    *,
    selected_layer_ids: set[str],
) -> str:
    if not selected_layer_ids:
        raise ValueError("at least one selected layer is required")
    lines, spans, newline = _asset_block_spans(queue_text)
    missing = selected_layer_ids - set(spans)
    if missing:
        raise ValueError(f"selected queue layers are missing from source YAML: {sorted(missing)}")
    updated_assets = updated_queue.get("assets")
    if not isinstance(updated_assets, list):
        raise ValueError("updated queue assets must be a list")
    updated_by_id = {
        str(asset.get("layer_id")): asset for asset in updated_assets if isinstance(asset, Mapping)
    }
    replacements: list[tuple[int, int, str]] = []
    for layer_id in selected_layer_ids:
        asset = updated_by_id.get(layer_id)
        if asset is None:
            raise ValueError(f"updated queue is missing selected layer: {layer_id}")
        start, end, indent = spans[layer_id]
        dumped_lines = yaml.safe_dump(
            [dict(asset)],
            allow_unicode=True,
            sort_keys=False,
        ).splitlines()
        rendered = "".join(f"{indent}{line}{newline}" for line in dumped_lines)
        replacements.append((start, end, rendered))
    for start, end, rendered in sorted(replacements, reverse=True):
        lines[start:end] = [rendered]
    result = "".join(lines)
    reparsed = yaml.safe_load(result)
    if reparsed != dict(updated_queue):
        raise ValueError("selected-only queue patch did not reproduce the approved queue content")
    return result


def _atomic_write_queue_text(path: Path, text: str, *, force: bool) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "segmented queue output")
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


def _plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose review-required assignments from ranked segmentation candidates."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("ranked", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply only human-approved assignments to a new queue candidate."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("assignment", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _run_plan(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    base_dir = args.base_dir.resolve()
    queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
    ranked_path = resolve_inside_base(base_dir, str(args.ranked), "ranked segmentation")
    output = resolve_inside_base(base_dir, str(args.output), "segmentation assignment output")
    if output in {queue_path, ranked_path}:
        raise ValueError("assignment output must not overwrite either input")
    queue, _queue_bytes, queue_sha256 = _load_queue_snapshot(queue_path)
    ranked = load_yaml_mapping(ranked_path)
    protected = referenced_artifact_paths(queue, base_dir, document_path=queue_path)
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("assignment output must not overwrite a canonical queue artifact")
    ranked_queue_sha256 = _required_sha256(
        ranked.get("asset_generation_queue_sha256"),
        "ranked asset_generation_queue_sha256",
    )
    if queue_sha256 != ranked_queue_sha256:
        raise ValueError("ranked segmentation was produced from different queue bytes")
    source_path = _source_file(queue, base_dir)
    if source_path.is_file():
        ranked_source_sha256 = _required_sha256(
            ranked.get("source_image_sha256"),
            "ranked source_image_sha256",
        )
        if file_sha256(source_path) != ranked_source_sha256:
            raise ValueError("ranked segmentation was produced from different source bytes")
    plan = build_assignment_plan(
        queue,
        ranked,
        queue_ref=normalize_queue_ref(queue_path, base_dir),
        ranked_ref=normalize_queue_ref(ranked_path, base_dir),
    )
    if args.execute:
        _atomic_write_yaml(output, plan, force=args.force)
    return plan, output


def _run_apply(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    base_dir = args.base_dir.resolve()
    queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
    assignment_path = resolve_inside_base(
        base_dir,
        str(args.assignment),
        "segmentation assignment",
    )
    output = resolve_inside_base(base_dir, str(args.output), "segmented queue output")
    if output in {queue_path, assignment_path}:
        raise ValueError("segmented queue output must not overwrite either input")
    queue, raw_queue_bytes, queue_sha256 = _load_queue_snapshot(queue_path)
    assignment = load_yaml_mapping(assignment_path)
    protected = referenced_artifact_paths(queue, base_dir, document_path=queue_path)
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("segmented queue output must not overwrite a canonical queue artifact")
    derived_from = assignment.get("derived_from")
    if not isinstance(derived_from, Mapping):
        raise ValueError("assignment derived_from must be a mapping")
    if derived_from.get("asset_generation_queue") != normalize_queue_ref(
        queue_path,
        base_dir,
    ):
        raise ValueError("assignment plan references a different queue path")
    expected_queue_sha256 = _required_sha256(
        derived_from.get("asset_generation_queue_sha256"),
        "assignment derived_from.asset_generation_queue_sha256",
    )
    if queue_sha256 != expected_queue_sha256:
        raise ValueError("assignment plan was reviewed for different queue bytes")
    source_path = _source_file(queue, base_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"source image not found: {source_path}")
    current_source_sha256 = file_sha256(source_path)
    updated_queue = apply_assignment_plan(
        queue,
        assignment,
        source_sha256=current_source_sha256,
    )
    raw_queue_text = raw_queue_bytes.decode("utf-8")
    selected_layer_ids = {
        str(item.get("layer_id"))
        for item in assignment.get("assignments", [])
        if isinstance(item, Mapping) and item.get("status") == "approved"
    }
    rendered_queue = render_queue_with_selected_updates(
        raw_queue_text,
        updated_queue,
        selected_layer_ids=selected_layer_ids,
    )
    if file_sha256(queue_path) != queue_sha256:
        raise ValueError("canonical queue changed during assignment apply")
    if file_sha256(source_path) != current_source_sha256:
        raise ValueError("source image changed during assignment apply")
    if args.execute:
        _atomic_write_queue_text(output, rendered_queue, force=args.force)
    return updated_queue, output


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    apply_mode = bool(raw_args and raw_args[0] == "apply")
    parser = _apply_parser() if apply_mode else _plan_parser()
    args = parser.parse_args(raw_args[1:] if apply_mode else raw_args)
    try:
        if apply_mode:
            document, output = _run_apply(args)
            count = sum(
                1
                for asset in document.get("assets", [])
                if isinstance(asset, Mapping) and "segmentation_run_id" in asset
            )
        else:
            document, output = _run_plan(args)
            assignments = document.get("assignments", [])
            count = len(assignments) if isinstance(assignments, list) else 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = {
        "status": "written" if args.execute else "planned",
        "mode": "apply" if apply_mode else "plan",
        "output": str(output),
        "assignment_count": count,
        "canonical_queue_modified": False,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
