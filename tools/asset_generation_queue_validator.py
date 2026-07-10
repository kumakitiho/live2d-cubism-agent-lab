from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import ArtifactIssue, load_yaml_mapping
from tools.asset_feedback_validator import load_layer_map_context, validate_asset_feedback
from tools.asset_manifest_validator import load_asset_manifest

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
    merge_ready: bool

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "merge_ready": self.merge_ready,
            "issues": [issue.format() for issue in self.issues],
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
    base_dir: Path | None = None,
) -> QueueValidationReport:
    issues: list[ArtifactIssue] = []
    for key in (
        "schema_version",
        "project",
        "source_image",
        "character_spec",
        "feedback_inputs",
        "jobs",
        "merge_gate",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))

    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))
    for key in ("project", "source_image", "character_spec"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(ArtifactIssue(key, "must be a non-empty string"))
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
    dependencies: list[tuple[str, str]] = []
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
                "outputs",
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

            for key in ("targets", "operations", "outputs"):
                if not _is_string_list(job.get(key)):
                    issues.append(
                        ArtifactIssue(f"{base}.{key}", "must be a non-empty list of strings")
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
                dependencies.extend((job_id, item) for item in depends_on)
                if family in REQUIRED_PART_FAMILIES and depends_on:
                    issues.append(
                        ArtifactIssue(
                            f"{base}.depends_on",
                            "required part-family jobs must start without dependencies",
                        )
                    )

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
                if not job_validation_ready:
                    issues.append(
                        ArtifactIssue(
                            f"{base}.validation",
                            "all required validations must be true for an approved job",
                        )
                    )
                else:
                    approved_jobs.add(job_id)

        missing_families = REQUIRED_PART_FAMILIES - set(family_to_job)
        for family in sorted(missing_families):
            issues.append(ArtifactIssue("jobs", f"missing required part family: {family}"))
        for job_id, dependency in dependencies:
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
                issues.append(
                    ArtifactIssue(
                        "merge_gate.validation.blocking_feedback_resolved",
                        "cannot be true while feedback is unresolved or unverified",
                    )
                )

        output_manifest = merge_gate.get("output_manifest")
        if not isinstance(output_manifest, str) or not output_manifest.strip():
            issues.append(ArtifactIssue("merge_gate.output_manifest", "must be a non-empty string"))
        elif output_manifest_path is not None:
            root = (base_dir or Path.cwd()).resolve()
            declared_manifest_path = _resolve_path(root, output_manifest).resolve()
            if declared_manifest_path != output_manifest_path.resolve():
                issues.append(
                    ArtifactIssue(
                        "merge_gate.output_manifest",
                        "must match the manifest used for handoff validation",
                    )
                )

        if manifest_document is not None:
            if output_manifest_path is None:
                issues.append(
                    ArtifactIssue(
                        "merge_gate.output_manifest",
                        "manifest path is required when manifest data is supplied",
                    )
                )
            if manifest_document.get("project") != data.get("project"):
                issues.append(
                    ArtifactIssue(
                        "merge_gate.output_manifest.project",
                        "manifest project must match queue project",
                    )
                )
            manifest_source = manifest_document.get("source_image")
            manifest_source_path = (
                manifest_source.get("path") if isinstance(manifest_source, Mapping) else None
            )
            queue_source_path = data.get("source_image")
            if not isinstance(manifest_source_path, str) or not isinstance(queue_source_path, str):
                issues.append(
                    ArtifactIssue(
                        "merge_gate.output_manifest.source_image",
                        "queue and manifest source image paths are required",
                    )
                )
            else:
                root = (base_dir or Path.cwd()).resolve()
                if (
                    _resolve_path(root, manifest_source_path).resolve()
                    != _resolve_path(root, queue_source_path).resolve()
                ):
                    issues.append(
                        ArtifactIssue(
                            "merge_gate.output_manifest.source_image",
                            "manifest source image must match queue source image",
                        )
                    )

    merge_ready = (
        not issues
        and bool(required_jobs)
        and required_jobs.issubset(approved_jobs)
        and merge_validations_ready
        and feedback_gate_verified
    )
    return QueueValidationReport(issues=tuple(issues), merge_ready=merge_ready)


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
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = validate_asset_generation_queue(
        data,
        feedback_documents=feedback_documents,
        manifest_document=manifest,
        output_manifest_path=manifest_path,
        base_dir=base_dir,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"valid: {str(report.valid).lower()}")
        print(f"merge_ready: {str(report.merge_ready).lower()}")
        for issue in report.issues:
            print(f"ERROR: {issue.format()}")
    if not report.valid:
        return 1
    if args.require_merge_ready and not report.merge_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
