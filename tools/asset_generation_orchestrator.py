from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    atomic_save_png,
    load_rgba,
    load_soft_mask,
    resolve_inside_base,
)
from tools.asset_quality_evaluator import main as quality_main
from tools.asset_recomposer import main as recompose_main
from tools.asset_refinement_planner import main as refinement_main
from tools.automatic_segmenter import main as segmentation_main
from tools.backend_registry import registry
from tools.backends.segmentation.integrity import canonical_mapping_sha256
from tools.generative_inpainter import file_sha256
from tools.generative_inpainter import main as inpainting_main
from tools.inpainting_candidate_ranker import (
    apply_selection_to_queue,
    rank_candidates,
    verify_candidate_artifacts,
    verify_selection_against_result,
)
from tools.mask_candidate_generator import build_mask_manifest
from tools.part_extractor import extract_rgba
from tools.resource_scheduler import (
    ResourceLimits,
    ResourceScheduler,
    ScheduledTask,
)
from tools.segmentation_assignment_planner import apply_assignment_plan
from tools.segmentation_assignment_planner import main as assignment_main
from tools.segmentation_candidate_ranker import main as segmentation_ranker_main

STAGE_ORDER = (
    "segmentation",
    "assignment",
    "extraction",
    "inpainting",
    "quality",
    "refinement",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "run.yaml"

    def path(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"run artifact path escaped run root: {value}") from exc
        return candidate

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in (
            "segmentation",
            "assignments",
            "masks",
            "extracted-parts",
            "inpainting",
            "quality",
            "previews",
            "refinement",
            "queue-candidates",
        ):
            (self.root / name).mkdir(exist_ok=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact path is outside base-dir: {path}") from exc


def _atomic_yaml(path: Path, data: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symbolic YAML output: {path}")
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


def _source_path(queue: Mapping[str, Any], base_dir: Path) -> Path:
    source = queue.get("source_image")
    if not isinstance(source, Mapping):
        raise ValueError("queue source_image must be a mapping")
    value = source.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("queue source_image.path is required")
    return resolve_inside_base(base_dir, value, "source_image.path")


def _queue_assets(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    assets = queue.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("queue assets must be a non-empty list")
    if not all(isinstance(asset, Mapping) for asset in assets):
        raise ValueError("queue assets must contain mappings")
    layer_ids = [asset.get("layer_id") for asset in assets]
    if not all(
        isinstance(layer_id, str) and RUN_ID_PATTERN.fullmatch(layer_id) for layer_id in layer_ids
    ):
        raise ValueError("queue layer_id values must be safe artifact identifiers")
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("queue layer_id values must be unique")
    return assets


def _validate_queue_artifact_safety(value: object, field: str = "queue") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if "token" in lowered or "credential" in lowered or "password" in lowered:
                raise ValueError(f"{field}.{key} must not contain credentials")
            _validate_queue_artifact_safety(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_queue_artifact_safety(item, f"{field}[{index}]")
        return
    if isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"{field} must not contain a local absolute path")


def _sanitize_provenance(value: object, base_dir: Path) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if "token" in lowered or "credential" in lowered or "password" in lowered:
                continue
            result[str(key)] = _sanitize_provenance(item, base_dir)
        return result
    if isinstance(value, list):
        return [_sanitize_provenance(item, base_dir) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provenance(item, base_dir) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return Path(value).resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            return "<redacted-local-path>"
    return value


def _stage(state: dict[str, Any], name: str) -> dict[str, Any]:
    stages = state.get("stages")
    assert isinstance(stages, dict)
    value = stages[name]
    assert isinstance(value, dict)
    return value


def _block_after(state: dict[str, Any], failed_stage: str) -> None:
    after = False
    for name in STAGE_ORDER:
        if name == failed_stage:
            after = True
            continue
        if not after:
            continue
        stage = _stage(state, name)
        if stage.get("status") not in {"completed", "skipped"}:
            stage["status"] = "blocked"


def _initial_state(
    *,
    queue: Mapping[str, Any],
    queue_ref: str,
    queue_sha256: str,
    source_sha256: str,
    run_id: str,
    segmentation_backend: str,
    inpainting_backend: str,
    limits: ResourceLimits,
    configuration_sha256: str,
) -> dict[str, Any]:
    segmentation_status = "skipped" if segmentation_backend == "disabled" else "planned"
    assignment_status = "skipped" if segmentation_backend == "disabled" else "planned"
    return {
        "schema_version": 1,
        "project": queue.get("project"),
        "run_id": run_id,
        "outcome": "running",
        "canonical_queue": queue_ref,
        "canonical_queue_sha256": queue_sha256,
        "canonical_queue_content_sha256": canonical_mapping_sha256(queue),
        "source_image_sha256": source_sha256,
        "configuration_sha256": configuration_sha256,
        "queue_candidate": None,
        "stages": {
            "segmentation": {
                "backend": segmentation_backend,
                "status": segmentation_status,
                "request": None,
                "result": None,
            },
            "assignment": {"status": assignment_status, "plan": None},
            "extraction": {"status": "planned", "request": None, "result": None},
            "inpainting": {
                "backend": inpainting_backend,
                "status": "skipped" if inpainting_backend == "disabled" else "planned",
                "requests": [],
                "results": [],
                "selections": [],
            },
            "quality": {"status": "planned", "request": None, "result": None},
            "refinement": {"status": "planned", "request": None, "result": None},
        },
        "resources": limits.to_dict(),
    }


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    run_id: str,
    queue: Mapping[str, Any],
    queue_ref: str,
    queue_sha256: str,
    source_sha256: str,
    configuration_sha256: str,
) -> None:
    if state.get("schema_version") != 1:
        raise ValueError("unsupported run state schema_version")
    if state.get("run_id") != run_id:
        raise ValueError("run ID does not match the existing run state")
    if state.get("project") != queue.get("project"):
        raise ValueError("run state project does not match the canonical queue")
    if state.get("canonical_queue") != queue_ref:
        raise ValueError("run state references a different canonical queue")
    if state.get("canonical_queue_sha256") != queue_sha256:
        raise ValueError("stale run: canonical queue bytes changed; start a new run")
    if state.get("canonical_queue_content_sha256") != canonical_mapping_sha256(queue):
        raise ValueError("stale run: canonical queue content changed; start a new run")
    if state.get("source_image_sha256") != source_sha256:
        raise ValueError("stale run: source image changed; start a new run")
    if state.get("configuration_sha256") != configuration_sha256:
        raise ValueError("resume configuration differs from the original run")


def _configuration_sha256(args: argparse.Namespace, limits: ResourceLimits) -> str:
    configuration = {
        "segmentation_backend": args.segmentation_backend,
        "inpainting_backend": args.inpainting_backend,
        "segmentation_model_id": args.segmentation_model_id,
        "segmentation_model_revision": args.segmentation_model_revision,
        "segmentation_checkpoint": (
            str(args.segmentation_checkpoint) if args.segmentation_checkpoint is not None else None
        ),
        "segmentation_model_config": args.segmentation_model_config,
        "grounding_model": str(args.grounding_model) if args.grounding_model is not None else None,
        "grounding_model_revision": args.grounding_model_revision,
        "inpainting_model_id": args.inpainting_model_id,
        "inpainting_model_revision": args.inpainting_model_revision,
        "inpainting_dtype": args.inpainting_dtype,
        "device": args.device,
        "resources": limits.to_dict(),
    }
    encoded = json.dumps(configuration, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(encoded)


def _load_run_artifact(
    path: Path,
    *,
    run_id: str,
    run_id_field: str = "run_id",
) -> dict[str, Any]:
    data = load_yaml_mapping(path)
    if data.get(run_id_field) != run_id:
        raise ValueError(f"artifact run ID mismatch: {path}")
    return data


def _approve_assignment_for_mock(plan: dict[str, Any]) -> dict[str, Any]:
    approved = deepcopy(plan)
    approved["review_status"] = "approved"
    assignments = approved.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("mock assignment plan contains no assignments")
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("assignment entries must be mappings")
        assignment["status"] = "approved"
        assignment["requires_review"] = False
        reasons = assignment.get("review_reasons")
        if isinstance(reasons, list):
            assignment["review_reasons"] = [
                reason for reason in reasons if reason != "human_assignment_approval_required"
            ]
    return approved


def _approve_selection_for_mock(selection: dict[str, Any]) -> dict[str, Any]:
    approved = deepcopy(selection)
    approved["review"] = {
        "status": "approved",
        "reviewer": "mock-auto-approval",
        "notes": "Approved by explicit --auto-approve-mock for deterministic CI only.",
    }
    return approved


def _asset_diff(
    canonical: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    before = {str(asset.get("layer_id")): asset for asset in _queue_assets(canonical)}
    after = {str(asset.get("layer_id")): asset for asset in _queue_assets(candidate)}
    changed: list[dict[str, Any]] = []
    for layer_id in sorted(set(before) | set(after)):
        old = before.get(layer_id, {})
        new = after.get(layer_id, {})
        fields = [key for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)]
        if fields:
            changed.append({"layer_id": layer_id, "fields": fields})
    return {
        "schema_version": 1,
        "canonical_queue_modified": False,
        "changed_parts": changed,
    }


def _segmentation_artifact_digests(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    ranked_path: Path,
    requests_path: Path,
    base_dir: Path,
    paths: RunPaths,
) -> dict[str, str]:
    segmentation_root = paths.path("segmentation")
    artifact_paths = [result_path, ranked_path, requests_path]
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            raise ValueError("segmentation candidates must be mappings")
        for field in ("soft_mask_file", "binary_mask_file", "preview_file"):
            value = candidate.get(field)
            if not isinstance(value, str):
                raise ValueError(f"segmentation candidate {field} is required")
            artifact = resolve_inside_base(base_dir, value, f"segmentation candidate {field}")
            try:
                artifact.resolve().relative_to(segmentation_root.resolve())
            except ValueError as exc:
                raise ValueError("segmentation candidate artifact escaped its run") from exc
            artifact_paths.append(artifact)
    digests: dict[str, str] = {}
    for artifact in artifact_paths:
        if not artifact.is_file():
            raise FileNotFoundError(f"segmentation artifact not found: {artifact}")
        reference = _relative(artifact, base_dir)
        if reference in digests:
            continue
        digests[reference] = file_sha256(artifact)
    return digests


def _verify_segmentation_artifacts(
    stage: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    result_path: Path,
    ranked_path: Path,
    requests_path: Path,
    base_dir: Path,
    paths: RunPaths,
) -> None:
    expected = stage.get("artifact_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("segmentation stage has no artifact digest manifest")
    actual = _segmentation_artifact_digests(
        result,
        result_path=result_path,
        ranked_path=ranked_path,
        requests_path=requests_path,
        base_dir=base_dir,
        paths=paths,
    )
    if dict(expected) != actual:
        raise ValueError("segmentation artifact digest mismatch or mixed run artifacts")


def _verify_assignment_matches_segmentation(
    queue: Mapping[str, Any],
    result: Mapping[str, Any],
    ranked: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    ranked_ref: str,
    run_id: str,
) -> None:
    for document in (result, ranked):
        if document.get("run_id") != run_id or document.get("project") != queue.get("project"):
            raise ValueError("segmentation artifact run/project mismatch")
    if assignment.get("segmentation_run_id") != run_id:
        raise ValueError("assignment run ID mismatch")
    derived_from = assignment.get("derived_from")
    if (
        not isinstance(derived_from, Mapping)
        or derived_from.get("segmentation_ranked") != ranked_ref
    ):
        raise ValueError("assignment references a different ranked segmentation artifact")
    result_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in result.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    ranked_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in ranked.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    if set(result_by_id) != set(ranked_by_id):
        raise ValueError("ranked segmentation candidate set differs from its result")
    immutable_candidate_fields = (
        "layer_id",
        "soft_mask_file",
        "binary_mask_file",
        "preview_file",
        "confidence",
        "source_backend",
        "model_id",
        "model_revision",
    )
    for candidate_id, candidate in ranked_by_id.items():
        source_candidate = result_by_id[candidate_id]
        if any(
            candidate.get(field) != source_candidate.get(field)
            for field in immutable_candidate_fields
        ):
            raise ValueError(f"ranked candidate was altered: {candidate_id}")
    assets_by_id = {str(asset.get("layer_id")): asset for asset in _queue_assets(queue)}
    assignments = assignment.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("assignment entries must be a list")
    assigned_layers: set[str] = set()
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("assignment entries must be mappings")
        layer_id = item.get("layer_id")
        selected_id = item.get("selected_candidate_id")
        if not isinstance(layer_id, str) or not isinstance(selected_id, str):
            raise ValueError("assignment layer and selected candidate are required")
        if layer_id in assigned_layers:
            raise ValueError(f"duplicate assignment layer: {layer_id}")
        assigned_layers.add(layer_id)
        selected = ranked_by_id.get(selected_id)
        asset = assets_by_id.get(layer_id)
        if selected is None or selected.get("layer_id") != layer_id or asset is None:
            raise ValueError(f"assignment selected candidate mismatch: {layer_id}")
        expected_alternatives = {
            candidate_id
            for candidate_id, candidate in ranked_by_id.items()
            if candidate.get("layer_id") == layer_id and candidate_id != selected_id
        }
        alternatives = item.get("alternative_candidate_ids")
        if not isinstance(alternatives, list) or set(alternatives) != expected_alternatives:
            raise ValueError(f"assignment alternatives mismatch: {layer_id}")
        expected_fields = {
            "target_mask": selected.get("soft_mask_file"),
            "protect_mask": asset.get("protect_mask"),
            "edge_extension_mask": asset.get("edge_extension_mask"),
            "inpaint_mask": asset.get("inpaint_mask"),
            "confidence": selected.get("confidence"),
            "segmentation_backend": selected.get("source_backend"),
            "segmentation_model_id": selected.get("model_id"),
            "segmentation_model_revision": selected.get("model_revision"),
            "segmentation_run_id": run_id,
            "segmentation_confidence": selected.get("confidence"),
        }
        if any(item.get(field) != value for field, value in expected_fields.items()):
            raise ValueError(f"assignment fields differ from selected candidate: {layer_id}")
    candidate_layers = {str(candidate.get("layer_id")) for candidate in ranked_by_id.values()}
    if assigned_layers != candidate_layers:
        raise ValueError("assignment layers differ from ranked segmentation layers")


def _extract_parts(
    queue: Mapping[str, Any],
    *,
    source: Image.Image,
    base_dir: Path,
    paths: RunPaths,
    scheduler: ResourceScheduler,
    resume: bool,
) -> dict[str, Any]:
    tasks: list[ScheduledTask] = []
    output_by_layer: dict[str, Path] = {}
    for asset in _queue_assets(queue):
        layer_id = asset.get("layer_id")
        target_value = asset.get("target_mask")
        if not isinstance(layer_id, str) or not isinstance(target_value, str):
            raise ValueError("each asset requires layer_id and target_mask")
        target_path = resolve_inside_base(base_dir, target_value, f"{layer_id}.target_mask")
        output = paths.path(f"extracted-parts/{layer_id}.png")
        output_by_layer[layer_id] = output

        def operation(
            target_path: Path = target_path,
            output: Path = output,
        ) -> str:
            target = load_soft_mask(target_path, source.size)
            atomic_save_png(extract_rgba(source, target), output, force=resume)
            return str(output)

        tasks.append(ScheduledTask(f"extract:{layer_id}", operation))
    results = scheduler.run(tasks)
    failures = [result for result in results.values() if result.status != "completed"]
    if failures:
        raise RuntimeError(
            "part extraction failed: " + "; ".join(str(item.error) for item in failures)
        )
    candidate = deepcopy(dict(queue))
    assets = candidate.get("assets")
    assert isinstance(assets, list)
    for asset in assets:
        assert isinstance(asset, dict)
        layer_id = str(asset["layer_id"])
        asset["source_file"] = _relative(output_by_layer[layer_id], base_dir)
    return candidate


def _inpainting_request(
    queue: Mapping[str, Any],
    asset: Mapping[str, Any],
    *,
    run_id: str,
    backend: str,
    model_id: str | None,
    model_revision: str | None,
    device: str,
    dtype: str,
    base_dir: Path,
    paths: RunPaths,
) -> dict[str, Any]:
    layer_id = asset.get("layer_id")
    if not isinstance(layer_id, str):
        raise ValueError("inpainting asset layer_id is required")
    source_value = queue.get("source_image")
    if not isinstance(source_value, Mapping) or not isinstance(source_value.get("path"), str):
        raise ValueError("queue source_image.path is required")
    required_paths = {
        key: asset.get(key)
        for key in (
            "source_file",
            "target_mask",
            "protect_mask",
            "edge_extension_mask",
            "inpaint_mask",
        )
    }
    if not all(isinstance(value, str) and value for value in required_paths.values()):
        raise ValueError(f"inpainting asset {layer_id} has incomplete source/mask paths")
    role = str(asset.get("role", layer_id))
    return {
        "schema_version": 1,
        "project": queue.get("project"),
        "run_id": run_id,
        "layer_id": layer_id,
        "source_image": source_value["path"],
        "current_part": required_paths["source_file"],
        "target_mask": required_paths["target_mask"],
        "protect_mask": required_paths["protect_mask"],
        "edge_extension_mask": required_paths["edge_extension_mask"],
        "inpaint_mask": required_paths["inpaint_mask"],
        "prompt": (
            f"Preserve character identity, line style, palette, lighting, canvas, and origin. "
            f"Complete only {role} ({layer_id}) inside the inpaint mask."
        ),
        "negative_prompt": (
            "full character regeneration, pose change, identity change, palette change, "
            "lighting change, background, canvas resize, unmasked region modification"
        ),
        "backend": backend,
        "backend_config": {
            "padding": 2,
            "model_size": [64, 64],
            "local_files_only": True,
            "model_id": _sanitize_provenance(model_id, base_dir),
            "model_revision": model_revision,
            "device": device,
            "dtype": dtype,
            "quality_thresholds": {
                "max_edge_continuity_score": 1.0,
                "max_boundary_color_difference_score": 1.0,
                "max_visual_reconstruction_difference_score": 1.0,
            },
        },
        "candidate_count": 3,
        "seed_policy": {"mode": "explicit_list", "seeds": [101, 102, 103]},
        "output_dir": _relative(paths.path(f"inpainting/{layer_id}/candidates"), base_dir),
    }


def _segmentation_provenance(
    result: Mapping[str, Any],
    ranked: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    requests_ref: str,
    run_id: str,
    base_dir: Path,
) -> dict[str, Any]:
    assignment_by_candidate = {
        str(item.get("selected_candidate_id")): item
        for item in assignment.get("assignments", [])
        if isinstance(item, Mapping)
    }
    backend_provenance = result.get("backend_provenance")
    ranked_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in ranked.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    request_by_layer = {
        str(request.get("layer_id")): request
        for request in result.get("requests", [])
        if isinstance(request, Mapping)
    }
    records: list[dict[str, Any]] = []
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        prompt = candidate.get("prompt_provenance")
        prompt_map = prompt if isinstance(prompt, Mapping) else {}
        selected = assignment_by_candidate.get(str(candidate.get("candidate_id")))
        ranked_candidate = ranked_by_id.get(str(candidate.get("candidate_id")), {})
        source_request = request_by_layer.get(str(candidate.get("layer_id")), {})
        selected_rank = ranked_candidate.get("rank")
        records.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "backend": candidate.get("source_backend"),
                "model_id": candidate.get("model_id"),
                "model_revision": candidate.get("model_revision"),
                "checkpoint": (
                    backend_provenance.get("checkpoint")
                    if isinstance(backend_provenance, Mapping)
                    else None
                ),
                "device": (
                    backend_provenance.get("device")
                    if isinstance(backend_provenance, Mapping)
                    else None
                ),
                "dtype": None,
                "seed": None,
                "prompt": prompt_map.get("semantic_prompt"),
                "negative_prompt": None,
                "point_prompts": prompt_map.get("point_prompts", []),
                "box_prompt": prompt_map.get("box_prompt"),
                "source_request": {
                    "artifact": requests_ref,
                    "request": deepcopy(dict(source_request)),
                },
                "run_id": run_id,
                "created_artifacts": [
                    candidate.get("soft_mask_file"),
                    candidate.get("binary_mask_file"),
                    candidate.get("preview_file"),
                ],
                "quality_metrics": ranked_candidate.get("ranking_metrics", {}),
                "selection_reason": (
                    f"assignment_selected_rank_{selected_rank}"
                    if selected is not None and isinstance(selected_rank, int)
                    else "not_selected"
                ),
                "review_status": selected.get("status") if selected is not None else "pending",
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "candidates": _sanitize_provenance(records, base_dir),
    }


def _inpainting_provenance(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    request_ref: str,
    run_id: str,
    base_dir: Path,
) -> list[dict[str, Any]]:
    ranks = {
        str(item.get("candidate_id")): item.get("rank")
        for item in selection.get("ranking", [])
        if isinstance(item, Mapping)
    }
    review = selection.get("review")
    review_status = review.get("status") if isinstance(review, Mapping) else "pending"
    backend_config = result.get("backend_config")
    config = backend_config if isinstance(backend_config, Mapping) else {}
    records: list[dict[str, Any]] = []
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id"))
        rank = ranks.get(candidate_id)
        records.append(
            {
                "candidate_id": candidate_id,
                "layer_id": result.get("layer_id"),
                "backend": candidate.get("backend"),
                "model_id": candidate.get("model_id"),
                "model_revision": candidate.get("model_revision"),
                "checkpoint": config.get("checkpoint"),
                "device": config.get("device", "cpu"),
                "dtype": config.get("dtype", "float32"),
                "seed": candidate.get("seed"),
                "prompt": request.get("prompt"),
                "negative_prompt": request.get("negative_prompt"),
                "point_prompts": [],
                "box_prompt": None,
                "source_request": request_ref,
                "run_id": run_id,
                "created_artifacts": [
                    candidate.get("output_file"),
                    candidate.get("preview_file"),
                ],
                "quality_metrics": candidate.get("quality_metrics", {}),
                "selection_reason": f"rank_{rank}" if rank is not None else "rejected",
                "review_status": review_status,
            }
        )
    sanitized = _sanitize_provenance(records, base_dir)
    assert isinstance(sanitized, list)
    return [dict(item) for item in sanitized if isinstance(item, Mapping)]


def _validate_backend_options(args: argparse.Namespace) -> None:
    if args.segmentation_backend != "disabled":
        registry.get_segmentation(
            args.segmentation_backend,
            {
                "model_id": args.segmentation_model_id,
                "model_revision": args.segmentation_model_revision,
                "checkpoint": args.segmentation_checkpoint,
                "model_config": args.segmentation_model_config,
                "grounding_model": args.grounding_model,
                "grounding_model_revision": args.grounding_model_revision,
                "device": args.device,
            },
        )
    if args.inpainting_backend != "disabled":
        registry.get_inpainting(args.inpainting_backend)
    enabled = {
        value
        for value in (args.segmentation_backend, args.inpainting_backend)
        if value != "disabled"
    }
    if args.auto_approve_mock and enabled - {"mock"}:
        raise ValueError("--auto-approve-mock is valid only when all enabled backends are mock")


def _wait_result(state: dict[str, Any], paths: RunPaths, base_dir: Path) -> dict[str, Any]:
    state["outcome"] = "waiting_for_review"
    _atomic_yaml(paths.state, state)
    return {
        "status": "waiting_for_review",
        "run_id": state["run_id"],
        "run_state": _relative(paths.state, base_dir),
        "queue_candidate": state.get("queue_candidate"),
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = args.base_dir.resolve()
    queue_path = resolve_inside_base(base_dir, str(args.queue), "canonical queue")
    queue_bytes = queue_path.read_bytes()
    queue = load_yaml_mapping(queue_path)
    _queue_assets(queue)
    _validate_queue_artifact_safety(queue)
    queue_ref = _relative(queue_path, base_dir)
    source_path = _source_path(queue, base_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"source image not found: {source_path}")
    source_sha256 = file_sha256(source_path)
    queue_sha256 = _sha256_bytes(queue_bytes)
    run_id = args.run_id or str(uuid.uuid4())
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must contain only letters, digits, dot, underscore, or hyphen")
    output_value = args.output_dir or Path("generated") / "runs" / run_id
    run_root = resolve_inside_base(base_dir, str(output_value), "run output directory")
    paths = RunPaths(run_root)
    limits = ResourceLimits(
        max_cpu_workers=args.max_cpu_workers,
        max_gpu_workers=args.max_gpu_workers,
        gpu_memory_budget_mb=args.gpu_memory_budget_mb,
        model_exclusive_lock=not args.no_model_exclusive_lock,
    )
    scheduler = ResourceScheduler(limits)
    configuration_sha256 = _configuration_sha256(args, limits)

    if paths.state.exists():
        if not args.resume:
            raise FileExistsError("run already exists; pass --resume or choose another run ID")
        state = load_yaml_mapping(paths.state)
        _validate_resume_state(
            state,
            run_id=run_id,
            queue=queue,
            queue_ref=queue_ref,
            queue_sha256=queue_sha256,
            source_sha256=source_sha256,
            configuration_sha256=configuration_sha256,
        )
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires an existing run.yaml")
        if run_root.exists() and any(run_root.iterdir()):
            raise FileExistsError("run output directory is not empty")
        paths.create()
        state = _initial_state(
            queue=queue,
            queue_ref=queue_ref,
            queue_sha256=queue_sha256,
            source_sha256=source_sha256,
            run_id=run_id,
            segmentation_backend=args.segmentation_backend,
            inpainting_backend=args.inpainting_backend,
            limits=limits,
            configuration_sha256=configuration_sha256,
        )
        _atomic_yaml(paths.state, state)
    paths.create()

    current_stage = "segmentation"
    try:
        working_queue: dict[str, Any] = deepcopy(queue)
        segmentation_stage = _stage(state, "segmentation")
        assignment_stage = _stage(state, "assignment")
        assignment_path = paths.path("assignments/assignment.yaml")
        segmentation_result_path = paths.path("segmentation/result.yaml")
        ranked_path = paths.path("segmentation/ranked.yaml")
        segmentation_requests_path = paths.path("segmentation/requests.yaml")

        if args.segmentation_backend != "disabled":
            if segmentation_stage.get("status") != "completed":
                current_stage = "segmentation"
                segmentation_stage["status"] = "running"
                segmentation_stage["request"] = {
                    "run_id": run_id,
                    "canonical_queue": queue_ref,
                }
                _atomic_yaml(paths.state, state)
                segment_args = [
                    queue_ref,
                    "--base-dir",
                    str(base_dir),
                    "--backend",
                    args.segmentation_backend.replace("_", "-"),
                    "--output",
                    _relative(segmentation_result_path, base_dir),
                    "--output-dir",
                    _relative(paths.path("segmentation/candidates"), base_dir),
                    "--run-id",
                    run_id,
                    "--device",
                    args.device,
                    "--execute",
                ]
                if args.resume:
                    segment_args.append("--force")
                for option, value in (
                    ("--model-id", args.segmentation_model_id),
                    ("--model-revision", args.segmentation_model_revision),
                    ("--checkpoint", args.segmentation_checkpoint),
                    ("--model-config", args.segmentation_model_config),
                    ("--grounding-model", args.grounding_model),
                    ("--grounding-model-revision", args.grounding_model_revision),
                ):
                    if value is not None:
                        segment_args.extend((option, str(value)))
                code = scheduler.run_stage(
                    "segmentation",
                    lambda: segmentation_main(segment_args),
                    resource="cpu" if args.segmentation_backend == "mock" else "gpu",
                )
                if code != 0:
                    raise StageFailure("segmentation", "segmentation command failed")
                result = _load_run_artifact(segmentation_result_path, run_id=run_id)
                sanitized = _sanitize_provenance(result, base_dir)
                assert isinstance(sanitized, Mapping)
                _atomic_yaml(segmentation_result_path, sanitized)
                result = dict(sanitized)
                rank_args = [
                    _relative(segmentation_result_path, base_dir),
                    "--base-dir",
                    str(base_dir),
                    "--output",
                    _relative(ranked_path, base_dir),
                    "--execute",
                ]
                if args.resume:
                    rank_args.append("--force")
                if segmentation_ranker_main(rank_args) != 0:
                    raise StageFailure("segmentation", "segmentation ranking failed")
                ranked = _load_run_artifact(ranked_path, run_id=run_id)
                _atomic_yaml(
                    segmentation_requests_path,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "requests": deepcopy(result.get("requests", [])),
                    },
                )
                assignment_args = [
                    queue_ref,
                    _relative(ranked_path, base_dir),
                    "--base-dir",
                    str(base_dir),
                    "--output",
                    _relative(assignment_path, base_dir),
                    "--execute",
                ]
                if args.resume:
                    assignment_args.append("--force")
                if assignment_main(assignment_args) != 0:
                    raise StageFailure("assignment", "assignment planning failed")
                segmentation_stage.update(
                    {
                        "status": "completed",
                        "result": _relative(segmentation_result_path, base_dir),
                        "ranked": _relative(ranked_path, base_dir),
                        "request": _relative(segmentation_requests_path, base_dir),
                        "artifact_sha256": _segmentation_artifact_digests(
                            result,
                            result_path=segmentation_result_path,
                            ranked_path=ranked_path,
                            requests_path=segmentation_requests_path,
                            base_dir=base_dir,
                            paths=paths,
                        ),
                    }
                )
                assignment_stage.update(
                    {
                        "status": "waiting_for_review",
                        "plan": _relative(assignment_path, base_dir),
                    }
                )
                _atomic_yaml(paths.state, state)
            result = _load_run_artifact(segmentation_result_path, run_id=run_id)
            ranked = _load_run_artifact(ranked_path, run_id=run_id)
            _verify_segmentation_artifacts(
                segmentation_stage,
                result,
                result_path=segmentation_result_path,
                ranked_path=ranked_path,
                requests_path=segmentation_requests_path,
                base_dir=base_dir,
                paths=paths,
            )
            assignment = _load_run_artifact(
                assignment_path,
                run_id=run_id,
                run_id_field="segmentation_run_id",
            )
            _verify_assignment_matches_segmentation(
                queue,
                result,
                ranked,
                assignment,
                ranked_ref=_relative(ranked_path, base_dir),
                run_id=run_id,
            )
            if assignment.get("review_status") != "approved" and args.auto_approve_mock:
                assignment = _approve_assignment_for_mock(assignment)
                _atomic_yaml(assignment_path, assignment)
            if assignment.get("review_status") != "approved":
                provenance = _segmentation_provenance(
                    result,
                    ranked,
                    assignment,
                    requests_ref=_relative(segmentation_requests_path, base_dir),
                    run_id=run_id,
                    base_dir=base_dir,
                )
                _atomic_yaml(paths.path("segmentation/provenance.yaml"), provenance)
                return _wait_result(state, paths, base_dir)
            current_stage = "assignment"
            working_queue = apply_assignment_plan(
                queue,
                assignment,
                source_sha256=source_sha256,
            )
            assigned_queue_path = paths.path("queue-candidates/after-assignment.yaml")
            _atomic_yaml(assigned_queue_path, working_queue)
            assignment_stage.update(
                {
                    "status": "completed",
                    "queue_candidate": _relative(assigned_queue_path, base_dir),
                }
            )
            state["queue_candidate"] = _relative(assigned_queue_path, base_dir)
            _atomic_yaml(
                paths.path("segmentation/provenance.yaml"),
                _segmentation_provenance(
                    result,
                    ranked,
                    assignment,
                    requests_ref=_relative(segmentation_requests_path, base_dir),
                    run_id=run_id,
                    base_dir=base_dir,
                ),
            )
            _atomic_yaml(paths.state, state)

        if args.inpainting_backend == "disabled":
            for name in ("extraction", "quality", "refinement"):
                stage = _stage(state, name)
                if stage.get("status") not in {"completed", "skipped"}:
                    stage["status"] = "skipped"
            state["outcome"] = "completed"
            _atomic_yaml(paths.state, state)
            if queue_path.read_bytes() != queue_bytes:
                raise StageFailure("assignment", "canonical queue changed during the run")
            return {
                "status": "completed",
                "run_id": run_id,
                "run_state": _relative(paths.state, base_dir),
                "queue_candidate": state.get("queue_candidate"),
            }

        extraction_stage = _stage(state, "extraction")
        if extraction_stage.get("status") == "completed":
            candidate_ref = extraction_stage.get("queue_candidate")
            if not isinstance(candidate_ref, str):
                raise StageFailure("extraction", "completed extraction has no queue candidate")
            working_queue = load_yaml_mapping(resolve_inside_base(base_dir, candidate_ref, "queue"))
        else:
            current_stage = "extraction"
            extraction_stage["status"] = "running"
            extraction_stage["request"] = {
                "run_id": run_id,
                "queue": state.get("queue_candidate") or queue_ref,
            }
            _atomic_yaml(paths.state, state)
            source = load_rgba(source_path)
            working_queue = _extract_parts(
                working_queue,
                source=source,
                base_dir=base_dir,
                paths=paths,
                scheduler=scheduler,
                resume=args.resume,
            )
            extracted_queue_path = paths.path("queue-candidates/after-extraction.yaml")
            _atomic_yaml(extracted_queue_path, working_queue)
            extraction_context = {
                "schema_version": 1,
                "run_id": run_id,
                "input_queue": state.get("queue_candidate") or queue_ref,
                "output_queue": _relative(extracted_queue_path, base_dir),
            }
            _atomic_yaml(paths.path("extracted-parts/input.yaml"), extraction_context)
            extraction_stage.update(
                {
                    "status": "completed",
                    "result": _relative(paths.path("extracted-parts/input.yaml"), base_dir),
                    "queue_candidate": _relative(extracted_queue_path, base_dir),
                }
            )
            state["queue_candidate"] = _relative(extracted_queue_path, base_dir)
            _atomic_yaml(paths.state, state)

        inpainting_stage = _stage(state, "inpainting")
        targets = [
            asset
            for asset in _queue_assets(working_queue)
            if asset.get("generation_method") == "inpaint"
        ]
        if not targets:
            inpainting_stage["status"] = "skipped"
        elif inpainting_stage.get("status") != "completed":
            current_stage = "inpainting"
            selections: list[dict[str, Any]] = []
            requests: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            selection_refs = inpainting_stage.get("selections")
            if inpainting_stage.get("status") == "waiting_for_review" and isinstance(
                selection_refs, list
            ):
                for ref in selection_refs:
                    if not isinstance(ref, str):
                        raise StageFailure("inpainting", "invalid selection artifact reference")
                    selection = _load_run_artifact(
                        resolve_inside_base(base_dir, ref, "selection"),
                        run_id=run_id,
                    )
                    if (
                        args.auto_approve_mock
                        and selection.get("review", {}).get("status") != "approved"
                    ):
                        selection = _approve_selection_for_mock(selection)
                        _atomic_yaml(resolve_inside_base(base_dir, ref, "selection"), selection)
                    selections.append(selection)
            else:
                inpainting_stage["status"] = "running"
                _atomic_yaml(paths.state, state)
                tasks: list[ScheduledTask] = []
                request_paths: dict[str, Path] = {}
                result_paths: dict[str, Path] = {}
                for asset in targets:
                    layer_id = str(asset["layer_id"])
                    request = _inpainting_request(
                        working_queue,
                        asset,
                        run_id=run_id,
                        backend=args.inpainting_backend,
                        model_id=args.inpainting_model_id,
                        model_revision=args.inpainting_model_revision,
                        device=args.device,
                        dtype=args.inpainting_dtype,
                        base_dir=base_dir,
                        paths=paths,
                    )
                    request_path = paths.path(f"inpainting/{layer_id}/request.yaml")
                    result_path = paths.path(f"inpainting/{layer_id}/result.yaml")
                    _atomic_yaml(request_path, request)
                    request_paths[layer_id] = request_path
                    result_paths[layer_id] = result_path
                    command = [
                        _relative(request_path, base_dir),
                        "--backend",
                        args.inpainting_backend,
                        "--base-dir",
                        str(base_dir),
                        "--output",
                        _relative(result_path, base_dir),
                        "--execute",
                    ]
                    if args.inpainting_model_id is not None:
                        command.extend(("--model-id", args.inpainting_model_id))
                    if args.resume:
                        command.append("--force")
                    tasks.append(
                        ScheduledTask(
                            f"inpaint:{layer_id}",
                            partial(inpainting_main, command),
                            resource="cpu" if args.inpainting_backend == "mock" else "gpu",
                        )
                    )
                task_results = scheduler.run(tasks)
                failed_tasks = [
                    item
                    for item in task_results.values()
                    if item.status != "completed" or item.value != 0
                ]
                if failed_tasks:
                    raise StageFailure(
                        "inpainting",
                        "inpainting generation failed: "
                        + "; ".join(str(item.error or item.value) for item in failed_tasks),
                    )
                provenance_records: list[dict[str, Any]] = []
                for asset in targets:
                    layer_id = str(asset["layer_id"])
                    request_path = request_paths[layer_id]
                    result_path = result_paths[layer_id]
                    request = _load_run_artifact(request_path, run_id=run_id)
                    result = _load_run_artifact(result_path, run_id=run_id)
                    sanitized_result = _sanitize_provenance(result, base_dir)
                    assert isinstance(sanitized_result, Mapping)
                    result = dict(sanitized_result)
                    _atomic_yaml(result_path, result)
                    selection = rank_candidates(
                        result,
                        result_ref=_relative(result_path, base_dir),
                        result_sha256=file_sha256(result_path),
                    )
                    selection_path = paths.path(f"inpainting/{layer_id}/selection.yaml")
                    if args.auto_approve_mock:
                        selection = _approve_selection_for_mock(selection)
                    _atomic_yaml(selection_path, selection)
                    requests.append(
                        {"layer_id": layer_id, "path": _relative(request_path, base_dir)}
                    )
                    results.append({"layer_id": layer_id, "path": _relative(result_path, base_dir)})
                    selections.append(selection)
                    provenance_records.extend(
                        _inpainting_provenance(
                            result,
                            request,
                            selection,
                            request_ref=_relative(request_path, base_dir),
                            run_id=run_id,
                            base_dir=base_dir,
                        )
                    )
                selection_paths = [
                    _relative(
                        paths.path(f"inpainting/{asset['layer_id']}/selection.yaml"), base_dir
                    )
                    for asset in targets
                ]
                inpainting_stage.update(
                    {
                        "status": "waiting_for_review",
                        "requests": requests,
                        "results": results,
                        "selections": selection_paths,
                    }
                )
                _atomic_yaml(
                    paths.path("inpainting/provenance.yaml"),
                    {"schema_version": 1, "run_id": run_id, "candidates": provenance_records},
                )
                _atomic_yaml(paths.state, state)
            if any(
                not isinstance(selection.get("review"), Mapping)
                or selection["review"].get("status") != "approved"
                for selection in selections
            ):
                return _wait_result(state, paths, base_dir)
            reviewed_provenance: list[dict[str, Any]] = []
            for selection in selections:
                reviewed_layer_id = selection.get("layer_id")
                if not isinstance(reviewed_layer_id, str):
                    raise StageFailure("inpainting", "selection layer_id is required")
                request_path = paths.path(f"inpainting/{reviewed_layer_id}/request.yaml")
                result_path = paths.path(f"inpainting/{reviewed_layer_id}/result.yaml")
                request = _load_run_artifact(request_path, run_id=run_id)
                result = _load_run_artifact(result_path, run_id=run_id)
                reviewed_provenance.extend(
                    _inpainting_provenance(
                        result,
                        request,
                        selection,
                        request_ref=_relative(request_path, base_dir),
                        run_id=run_id,
                        base_dir=base_dir,
                    )
                )
            _atomic_yaml(
                paths.path("inpainting/provenance.yaml"),
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "candidates": reviewed_provenance,
                },
            )
            for selection in selections:
                if selection.get("run_id") != run_id:
                    raise StageFailure("inpainting", "selection run ID mismatch")
                selected_layer_id = selection.get("layer_id")
                if not isinstance(selected_layer_id, str):
                    raise StageFailure("inpainting", "selection layer_id is required")
                selected_result_path = paths.path(f"inpainting/{selected_layer_id}/result.yaml")
                selected_result = _load_run_artifact(
                    selected_result_path,
                    run_id=run_id,
                )
                verify_selection_against_result(
                    selection,
                    selected_result,
                    result_sha256=file_sha256(selected_result_path),
                )
                verify_candidate_artifacts(selected_result, base_dir)
                working_queue = apply_selection_to_queue(working_queue, selection)
            inpainted_queue_path = paths.path("queue-candidates/after-inpainting.yaml")
            _atomic_yaml(inpainted_queue_path, working_queue)
            inpainting_stage.update(
                {
                    "status": "completed",
                    "queue_candidate": _relative(inpainted_queue_path, base_dir),
                }
            )
            state["queue_candidate"] = _relative(inpainted_queue_path, base_dir)
            _atomic_yaml(paths.state, state)
        elif isinstance(inpainting_stage.get("queue_candidate"), str):
            working_queue = load_yaml_mapping(
                resolve_inside_base(base_dir, inpainting_stage["queue_candidate"], "queue")
            )

        quality_stage = _stage(state, "quality")
        queue_candidate_ref = state.get("queue_candidate")
        if not isinstance(queue_candidate_ref, str):
            raise StageFailure("quality", "quality stage requires a queue candidate")
        queue_candidate_path = resolve_inside_base(base_dir, queue_candidate_ref, "queue candidate")
        if quality_stage.get("status") != "completed":
            current_stage = "quality"
            quality_stage["status"] = "running"
            mask_manifest_path = paths.path("masks/mask-manifest.yaml")
            manifest = build_mask_manifest(working_queue, queue_ref=queue_candidate_ref)
            _atomic_yaml(mask_manifest_path, manifest)
            reconstructed_path = paths.path("previews/reconstructed.png")
            recomposition_diff_path = paths.path("previews/recomposition-difference.png")
            quality_path = paths.path("quality/result.yaml")
            quality_diff_path = paths.path("quality/difference.png")
            quality_input_path = paths.path("quality/input.yaml")
            _atomic_yaml(
                quality_input_path,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "queue_candidate": queue_candidate_ref,
                    "mask_manifest": _relative(mask_manifest_path, base_dir),
                },
            )
            _atomic_yaml(paths.state, state)
            if (
                recompose_main(
                    [
                        _relative(mask_manifest_path, base_dir),
                        "--base-dir",
                        str(base_dir),
                        "--output",
                        _relative(reconstructed_path, base_dir),
                        "--difference-output",
                        _relative(recomposition_diff_path, base_dir),
                        "--execute",
                        *(["--force"] if args.resume else []),
                    ]
                )
                != 0
            ):
                raise StageFailure("quality", "part recomposition failed")
            if (
                quality_main(
                    [
                        _relative(mask_manifest_path, base_dir),
                        "--base-dir",
                        str(base_dir),
                        "--reconstructed",
                        _relative(reconstructed_path, base_dir),
                        "--difference-output",
                        _relative(quality_diff_path, base_dir),
                        "--output",
                        _relative(quality_path, base_dir),
                        "--execute",
                        *(["--force"] if args.resume else []),
                    ]
                )
                != 0
            ):
                raise StageFailure("quality", "global quality evaluation failed")
            quality_stage.update(
                {
                    "status": "completed",
                    "request": _relative(quality_input_path, base_dir),
                    "result": _relative(quality_path, base_dir),
                }
            )
            _atomic_yaml(paths.state, state)

        refinement_stage = _stage(state, "refinement")
        if refinement_stage.get("status") != "completed":
            current_stage = "refinement"
            refinement_stage["status"] = "running"
            quality_ref = quality_stage.get("result")
            if not isinstance(quality_ref, str):
                raise StageFailure("refinement", "refinement requires a quality result")
            refinement_plan_path = paths.path("refinement/plan.yaml")
            refined_queue_path = paths.path("queue-candidates/refined.yaml")
            refinement_stage["request"] = {
                "run_id": run_id,
                "queue_candidate": queue_candidate_ref,
                "quality": quality_ref,
            }
            _atomic_yaml(paths.state, state)
            if (
                refinement_main(
                    [
                        _relative(queue_candidate_path, base_dir),
                        quality_ref,
                        "--base-dir",
                        str(base_dir),
                        "--output",
                        _relative(refinement_plan_path, base_dir),
                        "--refined-queue-output",
                        _relative(refined_queue_path, base_dir),
                        "--execute",
                        *(["--force"] if args.resume else []),
                    ]
                )
                != 0
            ):
                raise StageFailure("refinement", "refinement planning failed")
            plan = load_yaml_mapping(refinement_plan_path)
            refinement_stage.update(
                {
                    "status": "completed",
                    "result": _relative(refinement_plan_path, base_dir),
                    "failed_parts": plan.get("summary", {}).get("failed_parts"),
                    "queue_candidate": _relative(refined_queue_path, base_dir),
                }
            )
            state["queue_candidate"] = _relative(refined_queue_path, base_dir)
            final_queue = load_yaml_mapping(refined_queue_path)
            _atomic_yaml(
                paths.path("queue-candidates/diff-summary.yaml"),
                _asset_diff(queue, final_queue),
            )
            failed_parts = plan.get("summary", {}).get("failed_parts", 0)
            state["outcome"] = "refinement_required" if failed_parts else "completed"
            _atomic_yaml(paths.state, state)

        if queue_path.read_bytes() != queue_bytes:
            raise StageFailure(current_stage, "canonical queue changed during the run")
        return {
            "status": state["outcome"],
            "run_id": run_id,
            "run_state": _relative(paths.state, base_dir),
            "queue_candidate": state.get("queue_candidate"),
        }
    except Exception as exc:
        failure = exc if isinstance(exc, StageFailure) else StageFailure(current_stage, str(exc))
        failed_stage_state = _stage(state, failure.stage)
        failed_stage_state["status"] = "failed"
        failed_stage_state["error"] = str(_sanitize_provenance(str(failure), base_dir))
        _block_after(state, failure.stage)
        state["outcome"] = "failed"
        _atomic_yaml(paths.state, state)
        if queue_path.read_bytes() != queue_bytes:
            raise RuntimeError("canonical queue changed while handling a failed run") from failure
        if isinstance(exc, StageFailure):
            raise
        raise failure from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate review-gated segmentation and source-preserving inpainting."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--segmentation-backend",
        choices=("disabled", "mock", "sam2", "grounded_sam2"),
        required=True,
    )
    parser.add_argument(
        "--inpainting-backend",
        choices=("disabled", "mock", "diffusers", "flux_fill"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-approve-mock", action="store_true")
    parser.add_argument("--max-cpu-workers", type=int, default=4)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument("--gpu-memory-budget-mb", type=int, default=0)
    parser.add_argument("--no-model-exclusive-lock", action="store_true")
    parser.add_argument("--segmentation-model-id")
    parser.add_argument("--segmentation-model-revision")
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--segmentation-model-config")
    parser.add_argument("--grounding-model", type=Path)
    parser.add_argument("--grounding-model-revision")
    parser.add_argument("--inpainting-model-id")
    parser.add_argument("--inpainting-model-revision")
    parser.add_argument("--inpainting-dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_backend_options(args)
        base_dir = args.base_dir.resolve()
        queue_path = resolve_inside_base(base_dir, str(args.queue), "canonical queue")
        queue = load_yaml_mapping(queue_path)
        _queue_assets(queue)
        _validate_queue_artifact_safety(queue)
        run_id = args.run_id or str(uuid.uuid4())
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run ID must contain only letters, digits, dot, underscore, or hyphen")
        output_value = args.output_dir or Path("generated") / "runs" / run_id
        output = resolve_inside_base(base_dir, str(output_value), "run output directory")
        if not args.execute:
            result = {
                "status": "planned",
                "run_id": run_id,
                "canonical_queue": _relative(queue_path, base_dir),
                "output_dir": _relative(output, base_dir),
                "segmentation_backend": args.segmentation_backend,
                "inpainting_backend": args.inpainting_backend,
                "model_load_attempted": False,
                "file_changes": False,
            }
        else:
            args.run_id = run_id
            result = _execute(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
