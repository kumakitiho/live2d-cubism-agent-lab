from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import ArtifactIssue, load_yaml_mapping, validate_layer_map
from tools.asset_feedback_validator import load_layer_map_context, validate_asset_feedback
from tools.asset_manifest_validator import load_asset_manifest, validate_asset_manifest
from tools.asset_pipeline_common import (
    GENERATION_METHODS,
    QUALITY_STATUSES,
    is_non_negative_int,
    is_positive_int,
    validate_dependency_dag,
)
from tools.asset_queue_builder import derive_asset_manifest, derive_layer_map

REQUIRED_PART_FAMILIES = {"eyes", "mouth", "hair", "body", "hidden_fill"}
JOB_STATUSES = {"planned", "running", "generated", "approved", "rejected", "blocked"}
JOB_VALIDATION_KEYS = {"same_canvas_alignment", "transparent_png", "style_match_reviewed"}
MERGE_VALIDATION_KEYS = {
    "unique_layer_ids",
    "same_canvas_alignment",
    "transparent_pngs",
    "inferred_assets_reviewed",
    "manifest_valid",
    "blocking_feedback_resolved",
}


@dataclass(frozen=True)
class QueueValidationReport:
    issues: tuple[ArtifactIssue, ...]
    warnings: tuple[ArtifactIssue, ...]
    merge_ready: bool
    validation_mode: str

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "merge_ready": self.merge_ready,
            "validation_mode": self.validation_mode,
            "issues": [issue.format() for issue in self.issues],
            "warnings": [warning.format() for warning in self.warnings],
        }


def _is_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_feedback_documents(
    data: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, dict[str, Any]]:
    raw_inputs = data.get("feedback_inputs")
    if not isinstance(raw_inputs, list):
        return {}
    documents: dict[str, dict[str, Any]] = {}
    for value in raw_inputs:
        if not isinstance(value, str) or not value.strip():
            continue
        feedback_path = _resolve_path(base_dir, value)
        feedback = load_yaml_mapping(feedback_path)
        model_refs = feedback.get("model_refs")
        if not isinstance(model_refs, Mapping):
            raise ValueError(f"feedback model_refs must be a mapping: {feedback_path}")
        layer_map_ref = model_refs.get("layer_map")
        if not isinstance(layer_map_ref, str) or not layer_map_ref.strip():
            raise ValueError(f"feedback model_refs.layer_map is required: {feedback_path}")
        layer_map_path = _resolve_path(base_dir, layer_map_ref)
        layer_ids, layer_map_project = load_layer_map_context(layer_map_path)
        feedback_issues = validate_asset_feedback(
            feedback,
            known_layer_ids=layer_ids,
            layer_map_project=layer_map_project,
            layer_map_path=layer_map_path,
            base_dir=base_dir,
        )
        if feedback_issues:
            details = "; ".join(issue.format() for issue in feedback_issues)
            raise ValueError(f"invalid feedback {feedback_path}: {details}")
        documents[value] = feedback
    return documents


def validate_asset_generation_queue(
    data: Mapping[str, Any],
    *,
    feedback_documents: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_document: Mapping[str, Any] | None = None,
    output_manifest_path: Path | None = None,
    layer_map_document: Mapping[str, Any] | None = None,
    output_layer_map_path: Path | None = None,
    queue_ref: str | None = None,
    base_dir: Path | None = None,
) -> QueueValidationReport:
    issues: list[ArtifactIssue] = []
    warnings: list[ArtifactIssue] = []
    raw_validation_mode = data.get("validation_mode")
    validation_mode = raw_validation_mode if raw_validation_mode in {"strict", "dev"} else "strict"

    def record(issue: ArtifactIssue, *, warning_in_dev: bool = False) -> None:
        if validation_mode == "dev" and warning_in_dev:
            warnings.append(issue)
        else:
            issues.append(issue)

    for key in (
        "schema_version",
        "validation_mode",
        "project",
        "source_image",
        "character_spec",
        "canvas",
        "derivatives",
        "import_constraints",
        "feedback_inputs",
        "assets",
        "jobs",
        "merge_gate",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))

    schema_version = data.get("schema_version")
    if schema_version not in {2, 3}:
        issues.append(ArtifactIssue("schema_version", "must equal 2 or 3"))
    if raw_validation_mode not in {"strict", "dev"}:
        issues.append(ArtifactIssue("validation_mode", "must be strict or dev"))
    for key in ("project", "character_spec"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(ArtifactIssue(key, "must be a non-empty string"))

    source_image = data.get("source_image")
    if not isinstance(source_image, Mapping):
        issues.append(ArtifactIssue("source_image", "must be a mapping"))
    else:
        for key in ("path", "rights_status"):
            value = source_image.get(key)
            if not isinstance(value, str) or not value.strip():
                issues.append(ArtifactIssue(f"source_image.{key}", "must be a non-empty string"))

    derivatives_value = data.get("derivatives")
    if not isinstance(derivatives_value, Mapping):
        issues.append(ArtifactIssue("derivatives", "must be a mapping"))
    else:
        derivative_suffixes = {
            "asset_manifest": {".yaml", ".yml"},
            "layer_map": {".yaml", ".yml"},
            "model_import_psd": {".psd"},
        }
        if schema_version == 3:
            derivative_suffixes["mask_manifest"] = {".yaml", ".yml"}
        for key, suffixes in derivative_suffixes.items():
            value = derivatives_value.get(key)
            if not isinstance(value, str) or Path(value).suffix.lower() not in suffixes:
                issues.append(
                    ArtifactIssue(
                        f"derivatives.{key}",
                        f"must use one of these suffixes: {sorted(suffixes)}",
                    )
                )

    asset_ids: set[str] = set()
    asset_readiness: dict[str, object] = {}
    asset_generation_methods: dict[str, object] = {}
    asset_dependencies: dict[str, set[str]] = {}
    draw_orders: set[int] = set()
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        issues.append(ArtifactIssue("assets", "must be a non-empty list"))
    else:
        for index, asset in enumerate(assets):
            base = f"assets[{index}]"
            if not isinstance(asset, Mapping):
                issues.append(ArtifactIssue(base, "must be a mapping"))
                continue
            required_asset_fields = [
                "layer_id",
                "layer_name",
                "layer_path",
                "source_file",
                "generation_method",
            ]
            if schema_version == 3:
                required_asset_fields.extend(
                    [
                        "target_mask",
                        "protect_mask",
                        "inpaint_mask",
                        "dependencies",
                        "draw_order",
                        "overlap_margin_px",
                        "quality_status",
                        "refinement_attempts",
                    ]
                )
            for key in required_asset_fields:
                if key not in asset:
                    issues.append(ArtifactIssue(f"{base}.{key}", "is required"))
            layer_id = asset.get("layer_id")
            if isinstance(layer_id, str) and layer_id.strip():
                if layer_id in asset_ids:
                    issues.append(ArtifactIssue(f"{base}.layer_id", f"duplicate id: {layer_id}"))
                asset_ids.add(layer_id)
                asset_readiness[layer_id] = asset.get("readiness")
                asset_generation_methods[layer_id] = asset.get("generation_method")
            else:
                issues.append(ArtifactIssue(f"{base}.layer_id", "must be a non-empty string"))
                continue
            layer_path = asset.get("layer_path")
            if not isinstance(layer_path, str) or not layer_path.strip():
                issues.append(ArtifactIssue(f"{base}.layer_path", "must be a non-empty string"))
            string_fields = ["layer_name", "source_file"]
            if schema_version == 3:
                string_fields.extend(["target_mask", "protect_mask", "inpaint_mask"])
            for key in string_fields:
                if not isinstance(asset.get(key), str) or not str(asset.get(key, "")).strip():
                    issues.append(ArtifactIssue(f"{base}.{key}", "must be a non-empty string"))
            allowed_generation_methods = set(GENERATION_METHODS)
            if schema_version == 2:
                allowed_generation_methods.add("mask_extract")
            if asset.get("generation_method") not in allowed_generation_methods:
                issues.append(
                    ArtifactIssue(
                        f"{base}.generation_method",
                        f"must be one of {sorted(allowed_generation_methods)}",
                    )
                )
            dependencies = asset.get("dependencies", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(value, str) and value.strip() for value in dependencies
            ):
                issues.append(ArtifactIssue(f"{base}.dependencies", "must be a list of strings"))
                asset_dependencies[layer_id] = set()
            else:
                asset_dependencies[layer_id] = set(dependencies)
            draw_order = asset.get("draw_order", asset.get("order"))
            if not is_positive_int(draw_order):
                issues.append(ArtifactIssue(f"{base}.draw_order", "must be a positive integer"))
            elif draw_order in draw_orders:
                issues.append(ArtifactIssue(f"{base}.draw_order", f"duplicate order: {draw_order}"))
            else:
                assert isinstance(draw_order, int)
                draw_orders.add(draw_order)
            if schema_version == 3 and not is_non_negative_int(asset.get("overlap_margin_px")):
                issues.append(
                    ArtifactIssue(f"{base}.overlap_margin_px", "must be a non-negative integer")
                )
            if schema_version == 3 and asset.get("quality_status") not in QUALITY_STATUSES:
                issues.append(
                    ArtifactIssue(
                        f"{base}.quality_status",
                        f"must be one of {sorted(QUALITY_STATUSES)}",
                    )
                )
            if schema_version == 3 and not is_non_negative_int(asset.get("refinement_attempts")):
                issues.append(
                    ArtifactIssue(f"{base}.refinement_attempts", "must be a non-negative integer")
                )
            if not isinstance(asset.get("include_in_import"), bool):
                issues.append(ArtifactIssue(f"{base}.include_in_import", "must be boolean"))
        issues.extend(validate_dependency_dag(asset_dependencies, path="assets"))
        for layer_id, dependency_ids in asset_dependencies.items():
            if asset_readiness.get(layer_id) != "approved":
                continue
            for dependency_id in dependency_ids:
                if asset_readiness.get(dependency_id) != "approved":
                    record(
                        ArtifactIssue(
                            f"assets.{layer_id}.dependencies",
                            f"approved asset requires approved dependency: {dependency_id}",
                        ),
                        warning_in_dev=True,
                    )

    try:
        derived_manifest = derive_asset_manifest(data)
        derived_layer_map = derive_layer_map(data)
    except ValueError as exc:
        issues.append(ArtifactIssue("derived_artifacts", str(exc)))
    else:
        manifest_report = validate_asset_manifest(derived_manifest)
        for manifest_error in manifest_report.errors:
            issues.append(
                ArtifactIssue(
                    f"derived_manifest.{manifest_error.path}",
                    manifest_error.message,
                )
            )
        for layer_error in validate_layer_map(derived_layer_map):
            issues.append(
                ArtifactIssue(
                    f"derived_layer_map.{layer_error.path}",
                    layer_error.message,
                )
            )

    feedback_inputs = data.get("feedback_inputs")
    feedback_input_refs: set[str] = set()
    if not isinstance(feedback_inputs, list) or not all(
        isinstance(item, str) and item.strip() for item in feedback_inputs
    ):
        issues.append(ArtifactIssue("feedback_inputs", "must be a list of non-empty strings"))
    else:
        feedback_input_refs = {str(item) for item in feedback_inputs}
        if len(feedback_input_refs) != len(feedback_inputs):
            issues.append(ArtifactIssue("feedback_inputs", "must not contain duplicates"))

    jobs = data.get("jobs")
    job_ids: set[str] = set()
    family_to_job: dict[str, str] = {}
    target_to_job: dict[str, str] = {}
    feedback_ref_to_job: dict[str, str] = {}
    approved_jobs: set[str] = set()
    job_dependencies: list[tuple[str, str]] = []
    if not isinstance(jobs, list) or not jobs:
        issues.append(ArtifactIssue("jobs", "must be a non-empty list"))
    else:
        for index, job in enumerate(jobs):
            base = f"jobs[{index}]"
            if not isinstance(job, Mapping):
                issues.append(ArtifactIssue(base, "must be a mapping"))
                continue
            for key in (
                "id",
                "part_family",
                "targets",
                "operations",
                "can_run_in_parallel",
                "depends_on",
                "feedback_refs",
                "status",
                "validation",
            ):
                if key not in job:
                    issues.append(ArtifactIssue(f"{base}.{key}", "is required"))

            job_id = job.get("id")
            if not isinstance(job_id, str) or not job_id.strip():
                issues.append(ArtifactIssue(f"{base}.id", "must be a non-empty string"))
                continue
            if job_id in job_ids:
                issues.append(ArtifactIssue(f"{base}.id", f"duplicate id: {job_id}"))
            job_ids.add(job_id)

            family = job.get("part_family")
            if not isinstance(family, str) or not family.strip():
                issues.append(ArtifactIssue(f"{base}.part_family", "must be a non-empty string"))
            elif family in family_to_job:
                issues.append(
                    ArtifactIssue(f"{base}.part_family", f"duplicate part family: {family}")
                )
            else:
                family_to_job[family] = job_id

            for key in ("targets", "operations"):
                if not _is_string_list(job.get(key)):
                    issues.append(
                        ArtifactIssue(f"{base}.{key}", "must be a non-empty list of strings")
                    )
            if schema_version == 3 and isinstance(job.get("targets"), list) and isinstance(
                job.get("operations"), list
            ):
                operations = set(job["operations"])
                for target in job["targets"]:
                    method = asset_generation_methods.get(str(target))
                    if method in GENERATION_METHODS and method not in operations:
                        issues.append(
                            ArtifactIssue(
                                f"{base}.operations",
                                f"must include target generation method {method}: {target}",
                            )
                        )
            feedback_refs = job.get("feedback_refs")
            if not isinstance(feedback_refs, list) or not all(
                isinstance(item, str) and item.strip() for item in feedback_refs
            ):
                issues.append(
                    ArtifactIssue(f"{base}.feedback_refs", "must be a list of non-empty strings")
                )
            else:
                for feedback_ref in feedback_refs:
                    owner = feedback_ref_to_job.get(feedback_ref)
                    if owner is not None:
                        issues.append(
                            ArtifactIssue(
                                f"{base}.feedback_refs",
                                f"feedback is already assigned to job {owner}: {feedback_ref}",
                            )
                        )
                    else:
                        feedback_ref_to_job[feedback_ref] = job_id
            targets = job.get("targets")
            if isinstance(targets, list):
                for target in targets:
                    if not isinstance(target, str):
                        continue
                    owner = target_to_job.get(target)
                    if owner is not None:
                        issues.append(
                            ArtifactIssue(
                                f"{base}.targets",
                                f"target layer is already owned by job {owner}: {target}",
                            )
                        )
                    else:
                        target_to_job[target] = job_id
            if job.get("can_run_in_parallel") is not True:
                issues.append(ArtifactIssue(f"{base}.can_run_in_parallel", "must be true"))

            depends_on = job.get("depends_on")
            if not isinstance(depends_on, list) or not all(
                isinstance(item, str) and item.strip() for item in depends_on
            ):
                issues.append(ArtifactIssue(f"{base}.depends_on", "must be a list of strings"))
            else:
                job_dependencies.extend((job_id, item) for item in depends_on)

            status = job.get("status")
            if status not in JOB_STATUSES:
                issues.append(
                    ArtifactIssue(f"{base}.status", f"must be one of {sorted(JOB_STATUSES)}")
                )

            validation = job.get("validation")
            job_validation_ready = False
            if not isinstance(validation, Mapping):
                issues.append(ArtifactIssue(f"{base}.validation", "must be a mapping"))
            else:
                required_keys = set(JOB_VALIDATION_KEYS)
                if family == "hidden_fill":
                    required_keys.add("inferred_marked")
                values: list[bool] = []
                for key in required_keys:
                    value = validation.get(key)
                    if not isinstance(value, bool):
                        issues.append(ArtifactIssue(f"{base}.validation.{key}", "must be boolean"))
                        values.append(False)
                    else:
                        values.append(value)
                job_validation_ready = all(values)
            if status == "approved":
                targets_ready = isinstance(targets, list) and all(
                    isinstance(target, str) and asset_readiness.get(target) == "approved"
                    for target in targets
                )
                if not job_validation_ready or not targets_ready:
                    record(
                        ArtifactIssue(
                            f"{base}.validation",
                            "approved jobs require true validations and approved target assets",
                        ),
                        warning_in_dev=True,
                    )
                else:
                    approved_jobs.add(job_id)

        missing_families = REQUIRED_PART_FAMILIES - set(family_to_job)
        for family in sorted(missing_families):
            issues.append(ArtifactIssue("jobs", f"missing required part family: {family}"))
        for job_id, dependency in job_dependencies:
            if dependency not in job_ids:
                issues.append(
                    ArtifactIssue(
                        f"jobs.{job_id}.depends_on",
                        f"unknown dependency: {dependency}",
                    )
                )
            if dependency == job_id:
                issues.append(
                    ArtifactIssue(f"jobs.{job_id}.depends_on", "job cannot depend on itself")
                )
        dependency_graph: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
        for job_id, dependency in job_dependencies:
            if dependency in job_ids and dependency != job_id:
                dependency_graph[job_id].add(dependency)
        for job_id, dependency in job_dependencies:
            if job_id in approved_jobs and dependency not in approved_jobs:
                record(
                    ArtifactIssue(
                        f"jobs.{job_id}.depends_on",
                        f"approved job requires approved dependency: {dependency}",
                    ),
                    warning_in_dev=True,
                )
        remaining = {job_id: set(values) for job_id, values in dependency_graph.items()}
        while True:
            ready = {
                job_id for job_id, dependency_values in remaining.items() if not dependency_values
            }
            if not ready:
                break
            for job_id in ready:
                remaining.pop(job_id)
            for dependency_values in remaining.values():
                dependency_values.difference_update(ready)
        if remaining:
            issues.append(
                ArtifactIssue(
                    "jobs.depends_on",
                    f"dependency cycle detected: {sorted(remaining)}",
                )
            )
        for target in sorted(set(target_to_job) - asset_ids):
            issues.append(ArtifactIssue("jobs.targets", f"unknown canonical asset: {target}"))
        for layer_id in sorted(asset_ids - set(target_to_job)):
            issues.append(ArtifactIssue("assets", f"asset is not assigned to a job: {layer_id}"))

    feedback_gate_verified = not feedback_input_refs
    if feedback_input_refs and feedback_documents is not None:
        feedback_gate_verified = True
        missing_documents = feedback_input_refs - set(feedback_documents)
        for feedback_ref in sorted(missing_documents):
            feedback_gate_verified = False
            issues.append(
                ArtifactIssue(
                    "feedback_inputs", f"feedback document was not loaded: {feedback_ref}"
                )
            )

        feedback_ids: set[str] = set()
        for feedback_ref in sorted(feedback_input_refs & set(feedback_documents)):
            feedback_document = feedback_documents[feedback_ref]
            if feedback_document.get("project") != data.get("project"):
                feedback_gate_verified = False
                issues.append(
                    ArtifactIssue(
                        feedback_ref,
                        "feedback project must match queue project",
                    )
                )
            feedback = feedback_document.get("feedback")
            if not isinstance(feedback, list):
                feedback_gate_verified = False
                issues.append(ArtifactIssue(feedback_ref, "feedback must be a list"))
                continue
            for index, entry in enumerate(feedback):
                base = f"{feedback_ref}.feedback[{index}]"
                if not isinstance(entry, Mapping):
                    feedback_gate_verified = False
                    issues.append(ArtifactIssue(base, "must be a mapping"))
                    continue
                feedback_id = entry.get("id")
                target_layer_id = entry.get("target_layer_id")
                if not isinstance(feedback_id, str) or not isinstance(target_layer_id, str):
                    feedback_gate_verified = False
                    issues.append(ArtifactIssue(base, "id and target_layer_id must be strings"))
                    continue
                if feedback_id in feedback_ids:
                    feedback_gate_verified = False
                    issues.append(
                        ArtifactIssue("feedback_inputs", f"duplicate feedback id: {feedback_id}")
                    )
                feedback_ids.add(feedback_id)

                expected_job = target_to_job.get(target_layer_id)
                if expected_job is None:
                    feedback_gate_verified = False
                    issues.append(
                        ArtifactIssue(
                            f"{base}.target_layer_id",
                            f"target layer is not owned by a queue job: {target_layer_id}",
                        )
                    )
                elif feedback_ref_to_job.get(feedback_id) != expected_job:
                    feedback_gate_verified = False
                    issues.append(
                        ArtifactIssue(
                            f"{base}.id",
                            f"must be assigned to owning job {expected_job}: {feedback_id}",
                        )
                    )

                if entry.get("status") not in {
                    "resolved",
                    "rejected",
                }:
                    feedback_gate_verified = False
                    if validation_mode == "dev":
                        warnings.append(
                            ArtifactIssue(
                                f"{base}.status",
                                "unresolved feedback prevents merge readiness",
                            )
                        )

        unknown_refs = set(feedback_ref_to_job) - feedback_ids
        for feedback_id in sorted(unknown_refs):
            feedback_gate_verified = False
            issues.append(
                ArtifactIssue(
                    "jobs.feedback_refs",
                    f"feedback ref does not exist in feedback inputs: {feedback_id}",
                )
            )

    merge_gate = data.get("merge_gate")
    merge_validations_ready = False
    required_jobs: set[str] = set()
    if not isinstance(merge_gate, Mapping):
        issues.append(ArtifactIssue("merge_gate", "must be a mapping"))
    else:
        if merge_gate.get("strategy") != "all_required_jobs_approved":
            issues.append(
                ArtifactIssue(
                    "merge_gate.strategy",
                    "must equal all_required_jobs_approved",
                )
            )
        raw_required_jobs = merge_gate.get("required_jobs")
        if not _is_string_list(raw_required_jobs):
            issues.append(
                ArtifactIssue("merge_gate.required_jobs", "must be a non-empty list of strings")
            )
        else:
            assert isinstance(raw_required_jobs, list)
            required_jobs = {str(item) for item in raw_required_jobs}
            unknown = required_jobs - job_ids
            for job_id in sorted(unknown):
                issues.append(
                    ArtifactIssue("merge_gate.required_jobs", f"unknown job id: {job_id}")
                )
            expected_required = {
                family_to_job[family]
                for family in REQUIRED_PART_FAMILIES
                if family in family_to_job
            }
            missing = expected_required - required_jobs
            for job_id in sorted(missing):
                issues.append(
                    ArtifactIssue("merge_gate.required_jobs", f"missing required job: {job_id}")
                )

        validation = merge_gate.get("validation")
        blocking_feedback_declared = False
        if not isinstance(validation, Mapping):
            issues.append(ArtifactIssue("merge_gate.validation", "must be a mapping"))
        else:
            merge_values: list[bool] = []
            for key in MERGE_VALIDATION_KEYS:
                value = validation.get(key)
                if not isinstance(value, bool):
                    issues.append(ArtifactIssue(f"merge_gate.validation.{key}", "must be boolean"))
                    merge_values.append(False)
                else:
                    merge_values.append(value)
            merge_validations_ready = all(merge_values)
            blocking_feedback_declared = validation.get("blocking_feedback_resolved") is True
            if blocking_feedback_declared and not feedback_gate_verified:
                record(
                    ArtifactIssue(
                        "merge_gate.validation.blocking_feedback_resolved",
                        "cannot be true while feedback is unresolved or unverified",
                    ),
                    warning_in_dev=True,
                )

    derivatives = data.get("derivatives")
    declared_manifest = (
        derivatives.get("asset_manifest") if isinstance(derivatives, Mapping) else None
    )
    if not isinstance(declared_manifest, str) or not declared_manifest.strip():
        issues.append(ArtifactIssue("derivatives.asset_manifest", "must be a non-empty string"))
    elif output_manifest_path is not None:
        root = (base_dir or Path.cwd()).resolve()
        declared_manifest_path = _resolve_path(root, declared_manifest).resolve()
        if declared_manifest_path != output_manifest_path.resolve():
            issues.append(
                ArtifactIssue(
                    "derivatives.asset_manifest",
                    "must match the manifest used for handoff validation",
                )
            )

    if manifest_document is not None:
        if output_manifest_path is None:
            issues.append(
                ArtifactIssue(
                    "derivatives.asset_manifest",
                    "manifest path is required when manifest data is supplied",
                )
            )
        try:
            derived_from = manifest_document.get("derived_from")
            manifest_queue_ref = (
                derived_from.get("asset_generation_queue")
                if isinstance(derived_from, Mapping)
                else None
            )
            expected_manifest = derive_asset_manifest(
                data,
                queue_ref=(
                    queue_ref
                    if queue_ref is not None
                    else manifest_queue_ref
                    if isinstance(manifest_queue_ref, str)
                    else None
                ),
            )
        except ValueError as exc:
            issues.append(ArtifactIssue("derivatives.asset_manifest", str(exc)))
        else:
            if dict(manifest_document) != expected_manifest:
                issues.append(
                    ArtifactIssue(
                        "derivatives.asset_manifest",
                        "manifest must be regenerated from the canonical queue",
                    )
                )

    declared_layer_map = derivatives.get("layer_map") if isinstance(derivatives, Mapping) else None
    if not isinstance(declared_layer_map, str) or not declared_layer_map.strip():
        issues.append(ArtifactIssue("derivatives.layer_map", "must be a non-empty string"))
    elif output_layer_map_path is not None:
        root = (base_dir or Path.cwd()).resolve()
        declared_layer_map_path = _resolve_path(root, declared_layer_map).resolve()
        if declared_layer_map_path != output_layer_map_path.resolve():
            issues.append(
                ArtifactIssue(
                    "derivatives.layer_map",
                    "must match the layer map used for handoff validation",
                )
            )

    if layer_map_document is not None:
        if output_layer_map_path is None:
            issues.append(
                ArtifactIssue(
                    "derivatives.layer_map",
                    "layer map path is required when layer map data is supplied",
                )
            )
        try:
            derived_from = layer_map_document.get("derived_from")
            layer_map_queue_ref = (
                derived_from.get("asset_generation_queue")
                if isinstance(derived_from, Mapping)
                else None
            )
            expected_layer_map = derive_layer_map(
                data,
                queue_ref=(
                    queue_ref
                    if queue_ref is not None
                    else layer_map_queue_ref
                    if isinstance(layer_map_queue_ref, str)
                    else None
                ),
            )
        except ValueError as exc:
            issues.append(ArtifactIssue("derivatives.layer_map", str(exc)))
        else:
            if dict(layer_map_document) != expected_layer_map:
                issues.append(
                    ArtifactIssue(
                        "derivatives.layer_map",
                        "layer map must be regenerated from the canonical queue",
                    )
                )

    merge_ready = (
        not issues
        and bool(required_jobs)
        and required_jobs.issubset(approved_jobs)
        and merge_validations_ready
        and feedback_gate_verified
    )
    return QueueValidationReport(
        issues=tuple(issues),
        warnings=tuple(warnings),
        merge_ready=merge_ready,
        validation_mode=validation_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a parallel Live2D asset queue.")
    parser.add_argument("queue", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-merge-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base_dir = args.base_dir.resolve()
        data = load_yaml_mapping(args.queue)
        feedback_documents = load_feedback_documents(data, base_dir=base_dir)
        manifest_path = (
            _resolve_path(base_dir, str(args.manifest)) if args.manifest is not None else None
        )
        manifest = load_asset_manifest(manifest_path) if manifest_path is not None else None
        layer_map_path: Path | None = None
        layer_map: dict[str, Any] | None = None
        if manifest is not None:
            derivatives = data.get("derivatives")
            layer_map_ref = (
                derivatives.get("layer_map") if isinstance(derivatives, Mapping) else None
            )
            if not isinstance(layer_map_ref, str) or not layer_map_ref.strip():
                raise ValueError("queue derivatives.layer_map is required for handoff validation")
            layer_map_path = _resolve_path(base_dir, layer_map_ref)
            layer_map = load_yaml_mapping(layer_map_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = validate_asset_generation_queue(
        data,
        feedback_documents=feedback_documents,
        manifest_document=manifest,
        output_manifest_path=manifest_path,
        layer_map_document=layer_map,
        output_layer_map_path=layer_map_path,
        queue_ref=args.queue.as_posix(),
        base_dir=base_dir,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"valid: {str(report.valid).lower()}")
        print(f"merge_ready: {str(report.merge_ready).lower()}")
        print(f"validation_mode: {report.validation_mode}")
        for issue in report.issues:
            print(f"ERROR: {issue.format()}")
        for warning in report.warnings:
            print(f"WARN: {warning.format()}")
    if not report.valid:
        return 1
    if args.require_merge_ready and not report.merge_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
