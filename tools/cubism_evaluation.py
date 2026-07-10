from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import ArtifactIssue, load_yaml_mapping
from tools.asset_feedback_validator import (
    EVIDENCE_KINDS,
    ISSUE_TYPES,
    REQUESTED_ACTIONS,
    SEVERITIES,
    load_layer_map_context,
    validate_asset_feedback,
)

CHECK_CATEGORIES = {"eye", "mouth", "mesh", "texture"}
CHECK_STATUSES = {"pass", "warn", "fail", "not_run"}


@dataclass(frozen=True)
class CubismEvaluationReport:
    issues: tuple[ArtifactIssue, ...]
    warnings: tuple[ArtifactIssue, ...]
    validation_mode: str
    result: str

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def feedback_convertible(self) -> bool:
        outcome_messages = {
            "required check is warn",
            "required check is not_run",
            "failed check blocks evaluation",
        }
        return self.result in {"warn", "fail"} and not any(
            issue.message not in outcome_messages for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "validation_mode": self.validation_mode,
            "result": self.result,
            "feedback_convertible": self.feedback_convertible,
            "issues": [issue.format() for issue in self.issues],
            "warnings": [warning.format() for warning in self.warnings],
        }


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _expected_summary_result(checks: list[Mapping[str, Any]]) -> str:
    statuses = [check.get("status") for check in checks]
    if "fail" in statuses:
        return "fail"
    if any(check.get("required") is True and check.get("status") == "not_run" for check in checks):
        return "incomplete"
    if "warn" in statuses:
        return "warn"
    return "pass"


def validate_cubism_evaluation(
    data: Mapping[str, Any],
    *,
    known_layer_ids: set[str] | None = None,
    layer_map_project: str | None = None,
    layer_map_path: Path | None = None,
    base_dir: Path | None = None,
) -> CubismEvaluationReport:
    issues: list[ArtifactIssue] = []
    warnings: list[ArtifactIssue] = []
    raw_mode = data.get("validation_mode")
    validation_mode = raw_mode if raw_mode in {"strict", "dev"} else "strict"

    for key in (
        "schema_version",
        "validation_mode",
        "project",
        "source_stage",
        "model_refs",
        "checks",
        "summary",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))
    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))
    if raw_mode not in {"strict", "dev"}:
        issues.append(ArtifactIssue("validation_mode", "must be strict or dev"))
    if not _is_non_empty_string(data.get("project")):
        issues.append(ArtifactIssue("project", "must be a non-empty string"))
    elif layer_map_project is not None and data.get("project") != layer_map_project:
        issues.append(ArtifactIssue("project", "must match layer map project"))
    if data.get("source_stage") != "live2d-cubism-workflow":
        issues.append(ArtifactIssue("source_stage", "must equal live2d-cubism-workflow"))

    model_refs = data.get("model_refs")
    if not isinstance(model_refs, Mapping):
        issues.append(ArtifactIssue("model_refs", "must be a mapping"))
    else:
        for key in ("model_import_psd", "layer_map", "action_plan_report"):
            if not _is_non_empty_string(model_refs.get(key)):
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

    checks_value = data.get("checks")
    checks: list[Mapping[str, Any]] = []
    if not isinstance(checks_value, list) or not checks_value:
        issues.append(ArtifactIssue("checks", "must be a non-empty list"))
    else:
        check_ids: set[str] = set()
        categories: set[str] = set()
        required_categories: set[str] = set()
        for index, check in enumerate(checks_value):
            base = f"checks[{index}]"
            if not isinstance(check, Mapping):
                issues.append(ArtifactIssue(base, "must be a mapping"))
                continue
            checks.append(check)
            for key in (
                "id",
                "category",
                "target_layer_ids",
                "required",
                "status",
                "evidence",
                "issue_type",
                "severity",
                "requested_action",
            ):
                if key not in check:
                    issues.append(ArtifactIssue(f"{base}.{key}", "is required"))

            check_id = check.get("id")
            if not _is_non_empty_string(check_id):
                issues.append(ArtifactIssue(f"{base}.id", "must be a non-empty string"))
            elif str(check_id) in check_ids:
                issues.append(ArtifactIssue(f"{base}.id", f"duplicate id: {check_id}"))
            else:
                check_ids.add(str(check_id))

            category = check.get("category")
            if category not in CHECK_CATEGORIES:
                issues.append(
                    ArtifactIssue(f"{base}.category", f"must be one of {sorted(CHECK_CATEGORIES)}")
                )
            else:
                categories.add(str(category))

            target_layer_ids = check.get("target_layer_ids")
            if (
                not isinstance(target_layer_ids, list)
                or not target_layer_ids
                or not all(_is_non_empty_string(layer_id) for layer_id in target_layer_ids)
            ):
                issues.append(
                    ArtifactIssue(
                        f"{base}.target_layer_ids",
                        "must be a non-empty list of layer IDs",
                    )
                )
            elif known_layer_ids is not None:
                for layer_id in target_layer_ids:
                    if layer_id not in known_layer_ids:
                        issues.append(
                            ArtifactIssue(
                                f"{base}.target_layer_ids",
                                f"does not exist in layer map: {layer_id}",
                            )
                        )

            if not isinstance(check.get("required"), bool):
                issues.append(ArtifactIssue(f"{base}.required", "must be boolean"))
            elif check.get("required") is True and category in CHECK_CATEGORIES:
                required_categories.add(str(category))
            status = check.get("status")
            if status not in CHECK_STATUSES:
                issues.append(
                    ArtifactIssue(f"{base}.status", f"must be one of {sorted(CHECK_STATUSES)}")
                )

            evidence = check.get("evidence")
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
                    if not _is_non_empty_string(evidence.get(key)):
                        issues.append(
                            ArtifactIssue(f"{base}.evidence.{key}", "must be a non-empty string")
                        )

            if status in {"warn", "fail"}:
                if check.get("issue_type") not in ISSUE_TYPES:
                    issues.append(
                        ArtifactIssue(
                            f"{base}.issue_type",
                            f"must be one of {sorted(ISSUE_TYPES)} for warn/fail",
                        )
                    )
                if check.get("severity") not in SEVERITIES:
                    issues.append(
                        ArtifactIssue(
                            f"{base}.severity",
                            f"must be one of {sorted(SEVERITIES)} for warn/fail",
                        )
                    )
                requested_action = check.get("requested_action")
                if not isinstance(requested_action, Mapping):
                    issues.append(
                        ArtifactIssue(f"{base}.requested_action", "must be a mapping for warn/fail")
                    )
                else:
                    if requested_action.get("action") not in REQUESTED_ACTIONS:
                        issues.append(
                            ArtifactIssue(
                                f"{base}.requested_action.action",
                                f"must be one of {sorted(REQUESTED_ACTIONS)}",
                            )
                        )
                    if not _is_non_empty_string(requested_action.get("details")):
                        issues.append(
                            ArtifactIssue(
                                f"{base}.requested_action.details",
                                "must be a non-empty string",
                            )
                        )

            if check.get("required") is True and status in {"warn", "not_run"}:
                finding = ArtifactIssue(f"{base}.status", f"required check is {status}")
                if validation_mode == "dev":
                    warnings.append(finding)
                else:
                    issues.append(finding)
            elif status == "fail":
                issues.append(ArtifactIssue(f"{base}.status", "failed check blocks evaluation"))
            elif status == "warn":
                warnings.append(ArtifactIssue(f"{base}.status", "optional check requires review"))

        for category in sorted(CHECK_CATEGORIES - categories):
            issues.append(ArtifactIssue("checks", f"missing required category: {category}"))
        for category in sorted(CHECK_CATEGORIES - required_categories):
            issues.append(
                ArtifactIssue(
                    "checks",
                    f"category must contain a required check: {category}",
                )
            )

    expected_result = _expected_summary_result(checks) if checks else "incomplete"
    summary = data.get("summary")
    if not isinstance(summary, Mapping):
        issues.append(ArtifactIssue("summary", "must be a mapping"))
    elif summary.get("result") != expected_result:
        issues.append(
            ArtifactIssue(
                "summary.result",
                f"must equal derived result {expected_result}",
            )
        )

    return CubismEvaluationReport(
        issues=tuple(issues),
        warnings=tuple(warnings),
        validation_mode=validation_mode,
        result=expected_result,
    )


def evaluation_to_asset_feedback(data: Mapping[str, Any]) -> dict[str, Any]:
    model_refs = data.get("model_refs")
    checks = data.get("checks")
    if not isinstance(model_refs, Mapping) or not isinstance(checks, list):
        raise ValueError("evaluation must contain model_refs and checks")

    feedback: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") not in {"warn", "fail"}:
            continue
        target_layer_ids = check.get("target_layer_ids")
        if not isinstance(target_layer_ids, list):
            continue
        for target_layer_id in target_layer_ids:
            feedback.append(
                {
                    "id": f"evaluation-{check.get('id')}-{target_layer_id}",
                    "target_layer_id": target_layer_id,
                    "issue_type": check.get("issue_type"),
                    "severity": check.get("severity"),
                    "evidence": deepcopy(check.get("evidence")),
                    "requested_action": deepcopy(check.get("requested_action")),
                    "status": "open",
                }
            )
    if not feedback:
        raise ValueError("evaluation has no warn/fail checks to convert")

    return {
        "schema_version": 1,
        "project": data.get("project"),
        "source_stage": "live2d-cubism-workflow",
        "target_stage": "image-to-live2d-assets",
        "model_refs": deepcopy(dict(model_refs)),
        "feedback": feedback,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--layer-map", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Cubism evaluation and create feedback.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    _add_common_arguments(validate_parser)
    validate_parser.add_argument("--json", action="store_true")
    feedback_parser = subparsers.add_parser("to-feedback")
    _add_common_arguments(feedback_parser)
    feedback_parser.add_argument("--output", type=Path, required=True)
    feedback_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        data = load_yaml_mapping(args.evaluation)
        layer_ids, layer_map_project = load_layer_map_context(args.layer_map)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = validate_cubism_evaluation(
        data,
        known_layer_ids=layer_ids,
        layer_map_project=layer_map_project,
        layer_map_path=args.layer_map,
        base_dir=base_dir,
    )
    if args.command == "validate":
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"valid: {str(report.valid).lower()}")
            print(f"result: {report.result}")
            print(f"validation_mode: {report.validation_mode}")
            for issue in report.issues:
                print(f"ERROR: {issue.format()}")
            for warning in report.warnings:
                print(f"WARN: {warning.format()}")
        return 0 if report.valid else 1

    if not report.feedback_convertible:
        for issue in report.issues:
            print(f"ERROR: {issue.format()}")
        return 1
    try:
        feedback = evaluation_to_asset_feedback(data)
        feedback_issues = validate_asset_feedback(
            feedback,
            known_layer_ids=layer_ids,
            layer_map_project=layer_map_project,
            layer_map_path=args.layer_map,
            base_dir=base_dir,
        )
        if feedback_issues:
            details = "; ".join(issue.format() for issue in feedback_issues)
            raise ValueError(f"generated feedback is invalid: {details}")
        output = args.output if args.output.is_absolute() else base_dir / args.output
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite without --force: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(feedback, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"OK: wrote asset feedback: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
