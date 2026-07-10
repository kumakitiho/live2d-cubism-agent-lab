from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import ArtifactIssue, load_yaml_mapping

ISSUE_TYPES = {
    "alignment",
    "missing_hidden_fill",
    "edge_artifact",
    "deformation_break",
    "layer_separation",
    "style_mismatch",
    "transparency",
    "mask_error",
    "other",
}
SEVERITIES = {"low", "medium", "high", "blocking"}
EVIDENCE_KINDS = {"screenshot", "operation_report", "parameter_probe", "manual_observation"}
REQUESTED_ACTIONS = {
    "regenerate",
    "resegment",
    "inpaint",
    "redraw",
    "adjust_mask",
    "split_layer",
    "merge_layer",
    "review",
}
FEEDBACK_STATUSES = {"open", "accepted", "in_progress", "resolved", "rejected"}


def load_layer_map_context(path: Path) -> tuple[set[str], str]:
    layer_map = load_yaml_mapping(path)
    layers = layer_map.get("layers")
    if not isinstance(layers, list):
        raise ValueError("layer map layers must be a list")
    ids: set[str] = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"layer map layers[{index}] must be a mapping")
        layer_id = layer.get("layer_id", layer.get("name"))
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"layer map layers[{index}] requires layer_id or name")
        ids.add(layer_id)
    project = layer_map.get("project")
    if not isinstance(project, str) or not project.strip():
        raise ValueError("layer map project must be a non-empty string")
    return ids, project


def load_layer_ids(path: Path) -> set[str]:
    ids, _project = load_layer_map_context(path)
    return ids


def validate_asset_feedback(
    data: Mapping[str, Any],
    *,
    known_layer_ids: set[str] | None = None,
    layer_map_project: str | None = None,
    layer_map_path: Path | None = None,
    base_dir: Path | None = None,
) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    for key in (
        "schema_version",
        "project",
        "source_stage",
        "target_stage",
        "model_refs",
        "feedback",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))

    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))
    if not isinstance(data.get("project"), str) or not str(data.get("project", "")).strip():
        issues.append(ArtifactIssue("project", "must be a non-empty string"))
    elif layer_map_project is not None and data.get("project") != layer_map_project:
        issues.append(ArtifactIssue("project", "must match layer map project"))
    if data.get("source_stage") != "live2d-cubism-workflow":
        issues.append(ArtifactIssue("source_stage", "must equal live2d-cubism-workflow"))
    if data.get("target_stage") != "image-to-live2d-assets":
        issues.append(ArtifactIssue("target_stage", "must equal image-to-live2d-assets"))

    model_refs = data.get("model_refs")
    if not isinstance(model_refs, Mapping):
        issues.append(ArtifactIssue("model_refs", "must be a mapping"))
    else:
        for key in ("model_import_psd", "layer_map", "action_plan_report"):
            value = model_refs.get(key)
            if not isinstance(value, str) or not value.strip():
                issues.append(ArtifactIssue(f"model_refs.{key}", "must be a non-empty string"))
        layer_map_ref = model_refs.get("layer_map")
        if layer_map_path is not None and isinstance(layer_map_ref, str):
            root = (base_dir or Path.cwd()).resolve()
            referenced_path = Path(layer_map_ref)
            if not referenced_path.is_absolute():
                referenced_path = root / referenced_path
            if referenced_path.resolve() != layer_map_path.resolve():
                issues.append(
                    ArtifactIssue(
                        "model_refs.layer_map",
                        "must reference the layer map used for validation",
                    )
                )

    feedback = data.get("feedback")
    if not isinstance(feedback, list) or not feedback:
        issues.append(ArtifactIssue("feedback", "must be a non-empty list"))
        return issues

    feedback_ids: list[str] = []
    for index, entry in enumerate(feedback):
        base = f"feedback[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(ArtifactIssue(base, "must be a mapping"))
            continue
        for key in (
            "id",
            "target_layer_id",
            "issue_type",
            "severity",
            "evidence",
            "requested_action",
            "status",
        ):
            if key not in entry:
                issues.append(ArtifactIssue(f"{base}.{key}", "is required"))

        feedback_id = entry.get("id")
        if isinstance(feedback_id, str) and feedback_id.strip():
            feedback_ids.append(feedback_id)
        else:
            issues.append(ArtifactIssue(f"{base}.id", "must be a non-empty string"))

        target_layer_id = entry.get("target_layer_id")
        if not isinstance(target_layer_id, str) or not target_layer_id.strip():
            issues.append(ArtifactIssue(f"{base}.target_layer_id", "must be a non-empty string"))
        elif known_layer_ids is not None and target_layer_id not in known_layer_ids:
            issues.append(
                ArtifactIssue(
                    f"{base}.target_layer_id",
                    f"does not exist in layer map: {target_layer_id}",
                )
            )

        if entry.get("issue_type") not in ISSUE_TYPES:
            issues.append(
                ArtifactIssue(f"{base}.issue_type", f"must be one of {sorted(ISSUE_TYPES)}")
            )
        if entry.get("severity") not in SEVERITIES:
            issues.append(ArtifactIssue(f"{base}.severity", f"must be one of {sorted(SEVERITIES)}"))
        if entry.get("status") not in FEEDBACK_STATUSES:
            issues.append(
                ArtifactIssue(f"{base}.status", f"must be one of {sorted(FEEDBACK_STATUSES)}")
            )

        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            issues.append(ArtifactIssue(f"{base}.evidence", "must be a mapping"))
        else:
            if evidence.get("kind") not in EVIDENCE_KINDS:
                issues.append(
                    ArtifactIssue(
                        f"{base}.evidence.kind",
                        f"must be one of {sorted(EVIDENCE_KINDS)}",
                    )
                )
            for key in ("path", "description"):
                value = evidence.get(key)
                if not isinstance(value, str) or not value.strip():
                    issues.append(
                        ArtifactIssue(f"{base}.evidence.{key}", "must be a non-empty string")
                    )

        requested_action = entry.get("requested_action")
        if not isinstance(requested_action, Mapping):
            issues.append(ArtifactIssue(f"{base}.requested_action", "must be a mapping"))
        else:
            if requested_action.get("action") not in REQUESTED_ACTIONS:
                issues.append(
                    ArtifactIssue(
                        f"{base}.requested_action.action",
                        f"must be one of {sorted(REQUESTED_ACTIONS)}",
                    )
                )
            details = requested_action.get("details")
            if not isinstance(details, str) or not details.strip():
                issues.append(
                    ArtifactIssue(
                        f"{base}.requested_action.details",
                        "must be a non-empty string",
                    )
                )

    for feedback_id, count in Counter(feedback_ids).items():
        if count > 1:
            issues.append(ArtifactIssue("feedback", f"duplicate feedback id: {feedback_id}"))
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Cubism-to-asset feedback YAML.")
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--layer-map", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = load_yaml_mapping(args.feedback)
        known_layer_ids, layer_map_project = load_layer_map_context(args.layer_map)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    issues = validate_asset_feedback(
        data,
        known_layer_ids=known_layer_ids,
        layer_map_project=layer_map_project,
        layer_map_path=args.layer_map,
        base_dir=args.base_dir,
    )
    if issues:
        for issue in issues:
            print(f"ERROR: {issue.format()}")
        return 1
    print("OK: asset feedback is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
