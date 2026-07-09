from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TEMPORARY_NAME_TOKENS = ("copy", "temp", "tmp", "old", "rough", "test", "コピー")


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
        "model",
        "appearance",
        "motions",
        "expressions",
        "constraints",
        "deliverables",
    )
    for key in required:
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))

    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))

    for key in ("model", "appearance", "motions", "constraints"):
        if key in data and not isinstance(data[key], Mapping):
            issues.append(ArtifactIssue(key, "must be a mapping"))

    for key in ("expressions", "deliverables"):
        if key in data and not isinstance(data[key], list):
            issues.append(ArtifactIssue(key, "must be a list"))

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
