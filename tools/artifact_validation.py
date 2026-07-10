from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TEMPORARY_NAME_TOKENS = ("copy", "temp", "tmp", "old", "rough", "test", "コピー")
HUMAN_CONFIRMED_SPEC_FIELDS = {
    "source_image.rights_status",
    "model_scope",
    "motion_level",
    "target_runtime",
    "purpose",
    "expressions",
    "physics_targets",
}


@dataclass(frozen=True)
class ArtifactIssue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def validate_character_spec(data: Mapping[str, Any]) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    required = (
        "schema_version",
        "project",
        "source_image",
        "model_scope",
        "motion_level",
        "target_runtime",
        "purpose",
        "appearance",
        "motions",
        "expressions",
        "physics_targets",
        "spec_provenance",
        "constraints",
        "deliverables",
    )
    for key in required:
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))

    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))

    if data.get("model_scope") not in {"bust_up", "half_body", "full_body"}:
        issues.append(ArtifactIssue("model_scope", "must be bust_up, half_body, or full_body"))
    if data.get("motion_level") not in {"minimal", "standard", "expressive"}:
        issues.append(ArtifactIssue("motion_level", "must be minimal, standard, or expressive"))
    for key in ("project", "target_runtime", "purpose"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(ArtifactIssue(key, "must be a non-empty string"))

    source_image = data.get("source_image")
    if not isinstance(source_image, Mapping):
        issues.append(ArtifactIssue("source_image", "must be a mapping"))
    else:
        source_path = source_image.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            issues.append(ArtifactIssue("source_image.path", "must be a non-empty string"))
        if source_image.get("rights_status") != "confirmed":
            issues.append(
                ArtifactIssue(
                    "source_image.rights_status",
                    "must be confirmed before asset handoff",
                )
            )

    for key in ("appearance", "motions", "spec_provenance", "constraints"):
        if key in data and not isinstance(data[key], Mapping):
            issues.append(ArtifactIssue(key, "must be a mapping"))

    for key in ("expressions", "physics_targets", "deliverables"):
        value = data.get(key)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            issues.append(ArtifactIssue(key, "must be a non-empty list of strings"))

    appearance = data.get("appearance")
    if isinstance(appearance, Mapping):
        if not isinstance(appearance.get("observed"), Mapping):
            issues.append(ArtifactIssue("appearance.observed", "must be a mapping"))
        for key in ("requested_changes", "uncertain"):
            value = appearance.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                issues.append(ArtifactIssue(f"appearance.{key}", "must be a list of strings"))

    motions = data.get("motions")
    if isinstance(motions, Mapping):
        for key in ("face", "body"):
            value = motions.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                issues.append(ArtifactIssue(f"motions.{key}", "must be a list of strings"))

    provenance = data.get("spec_provenance")
    if isinstance(provenance, Mapping):
        provenance_sets: dict[str, set[str]] = {}
        for key in (
            "image_inferred_fields",
            "user_confirmed_fields",
            "assumptions",
            "open_questions",
        ):
            value = provenance.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                issues.append(ArtifactIssue(f"spec_provenance.{key}", "must be a list of strings"))
            else:
                values = [str(item) for item in value]
                if len(values) != len(set(values)):
                    issues.append(
                        ArtifactIssue(f"spec_provenance.{key}", "must not contain duplicates")
                    )
                provenance_sets[key] = set(values)

        inferred = provenance_sets.get("image_inferred_fields", set())
        confirmed = provenance_sets.get("user_confirmed_fields", set())
        for field in sorted(HUMAN_CONFIRMED_SPEC_FIELDS & inferred):
            issues.append(
                ArtifactIssue(
                    "spec_provenance.image_inferred_fields",
                    f"human-only field cannot be image-inferred: {field}",
                )
            )
        for field in sorted(HUMAN_CONFIRMED_SPEC_FIELDS - confirmed):
            issues.append(
                ArtifactIssue(
                    "spec_provenance.user_confirmed_fields",
                    f"must include human-confirmed field: {field}",
                )
            )
        for field in sorted(inferred & confirmed):
            issues.append(
                ArtifactIssue(
                    "spec_provenance",
                    f"field cannot be both image-inferred and user-confirmed: {field}",
                )
            )
        open_questions = provenance.get("open_questions")
        if isinstance(open_questions, list) and open_questions:
            issues.append(
                ArtifactIssue(
                    "spec_provenance.open_questions",
                    "must be empty before asset handoff",
                )
            )

    constraints = data.get("constraints")
    if isinstance(constraints, Mapping):
        for guardrail in (
            "no_full_auto_rigging_claim",
            "human_visual_review_required",
            "separate_material_and_import_psd",
        ):
            if constraints.get(guardrail) is not True:
                issues.append(ArtifactIssue(f"constraints.{guardrail}", "must be true"))

    return issues


def validate_layer_map(data: Mapping[str, Any]) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))

    canvas = data.get("canvas")
    if not isinstance(canvas, Mapping):
        issues.append(ArtifactIssue("canvas", "must be a mapping"))
    else:
        for dimension in ("width", "height"):
            value = canvas.get(dimension)
            if not isinstance(value, int) or value <= 0:
                issues.append(ArtifactIssue(f"canvas.{dimension}", "must be a positive integer"))

    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        issues.append(ArtifactIssue("layers", "must be a non-empty list"))
        return issues

    names: list[str] = []
    sided_roles: dict[str, set[str]] = {}
    for index, layer in enumerate(layers):
        base = f"layers[{index}]"
        if not isinstance(layer, Mapping):
            issues.append(ArtifactIssue(base, "must be a mapping"))
            continue
        for key in ("path", "name", "role", "side", "source", "readiness", "required"):
            if key not in layer:
                issues.append(ArtifactIssue(f"{base}.{key}", "is required"))

        name = layer.get("name")
        if isinstance(name, str):
            names.append(name)
            lowered = name.lower()
            if any(token in lowered for token in TEMPORARY_NAME_TOKENS):
                issues.append(ArtifactIssue(f"{base}.name", "looks temporary"))
        elif name is not None:
            issues.append(ArtifactIssue(f"{base}.name", "must be a string"))

        side = layer.get("side")
        if side not in {"L", "R", "C", "none"}:
            issues.append(ArtifactIssue(f"{base}.side", "must be L, R, C, or none"))

        role = layer.get("role")
        if isinstance(role, str) and side in {"L", "R"}:
            sided_roles.setdefault(role, set()).add(str(side))

    for name, count in Counter(names).items():
        if count > 1:
            issues.append(ArtifactIssue("layers", f"duplicate layer name: {name}"))

    for role, sides in sided_roles.items():
        if sides != {"L", "R"}:
            issues.append(ArtifactIssue("layers", f"missing L/R pair for role: {role}"))

    return issues
