from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_MODES = {"file", "ui_macro", "api", "manual_checkpoint"}

ALLOWED_COMMANDS: dict[str, set[str]] = {
    "file": {
        "file.validate_layer_map",
        "file.verify_imported_document",
    },
    "ui_macro": {
        "cubism_ui.focus",
        "cubism_ui.screenshot",
        "cubism_ui.import_psd",
        "cubism_ui.apply_auto_mesh",
        "cubism_ui.save",
        "cubism_ui.undo",
    },
    "api": {
        "cubism_api.register",
        "cubism_api.get_approval",
        "cubism_api.get_documents",
        "cubism_api.get_document_snapshot",
        "cubism_api.get_current_model_uid",
        "cubism_api.get_current_edit_mode",
        "cubism_api.get_parameters",
        "cubism_api.get_parameter_values",
        "cubism_api.set_parameter_values",
        "cubism_api.clear_parameter_values",
        "cubism_api.send_log",
    },
    "manual_checkpoint": set(),
}

COMMAND_ARGUMENTS: dict[str, tuple[set[str], set[str]]] = {
    "file.validate_layer_map": ({"path"}, set()),
    "file.verify_imported_document": (
        {"before_step", "import_step", "after_step"},
        set(),
    ),
    "cubism_ui.focus": (set(), set()),
    "cubism_ui.screenshot": ({"path"}, set()),
    "cubism_ui.import_psd": (
        {"psd_path"},
        {
            "open_mode",
            "screenshot",
            "failure_screenshot",
            "dialog_timeout",
            "import_timeout",
        },
    ),
    "cubism_ui.apply_auto_mesh": (
        set(),
        {"preset", "alpha", "screenshot", "failure_screenshot", "dialog_timeout"},
    ),
    "cubism_ui.save": (set(), set()),
    "cubism_ui.undo": (set(), set()),
    "cubism_api.register": (set(), set()),
    "cubism_api.get_approval": (set(), set()),
    "cubism_api.get_documents": (set(), set()),
    "cubism_api.get_document_snapshot": (set(), set()),
    "cubism_api.get_current_model_uid": (set(), set()),
    "cubism_api.get_current_edit_mode": (set(), set()),
    "cubism_api.get_parameters": (set(), {"model_uid", "use_current_model"}),
    "cubism_api.get_parameter_values": (
        set(),
        {"model_uid", "use_current_model", "ids"},
    ),
    "cubism_api.set_parameter_values": (
        {"parameters"},
        {"model_uid", "use_current_model"},
    ),
    "cubism_api.clear_parameter_values": (set(), {"model_uid", "use_current_model"}),
    "cubism_api.send_log": ({"message"}, {"type", "display"}),
}

OUTPUT_PATH_ARGUMENTS: dict[str, set[str]] = {
    "cubism_ui.screenshot": {"path"},
    "cubism_ui.import_psd": {"screenshot", "failure_screenshot"},
    "cubism_ui.apply_auto_mesh": {"screenshot", "failure_screenshot"},
}

WORKSPACE_INPUT_PATH_ARGUMENTS: dict[str, set[str]] = {
    "file.validate_layer_map": {"path"},
}

FORBIDDEN_UI_ARGUMENT_KEYS = {
    "x",
    "y",
    "screen_x",
    "screen_y",
    "coordinates",
    "start_coordinates",
    "end_coordinates",
}
FORBIDDEN_COMMAND_TOKENS = {"click", "drag", "mousemove", "move_pointer"}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


class ActionPlanError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = list(issues)
        super().__init__("\n".join(issue.format() for issue in self.issues))


def load_action_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"action plan not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ActionPlanError([ValidationIssue("$", "root must be a mapping")])
    return raw


def _walk_mapping_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield path, key_text
            yield from _walk_mapping_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping_keys(child, f"{prefix}[{index}]")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_plan_path(value: Any, workspace_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return (workspace_root / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_command_arguments(
    command: str,
    args: Mapping[str, Any],
    base: str,
    workspace_root: Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required, optional = COMMAND_ARGUMENTS.get(command, (set(), set()))
    provided = {str(key) for key in args}
    for key in args:
        if not isinstance(key, str):
            issues.append(ValidationIssue(f"{base}.args", "all argument keys must be strings"))
            return issues
    for key in sorted(required - provided):
        issues.append(ValidationIssue(f"{base}.args.{key}", "is required"))
    for key in sorted(provided - required - optional):
        issues.append(ValidationIssue(f"{base}.args.{key}", "is not allowed"))

    for key in (required | optional) & provided:
        value = args[key]
        path = f"{base}.args.{key}"
        if key in {
            "path",
            "psd_path",
            "screenshot",
            "failure_screenshot",
            "model_uid",
            "message",
            "type",
            "preset",
            "open_mode",
            "before_step",
            "import_step",
            "after_step",
        } and (not isinstance(value, str) or not value):
            issues.append(ValidationIssue(path, "must be a non-empty string"))
        elif key in {"dialog_timeout", "import_timeout"} and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        ):
            issues.append(ValidationIssue(path, "must be a positive number"))
        elif key == "alpha" and (
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
        ):
            issues.append(ValidationIssue(path, "must be an integer from 0 to 255"))
        elif key in {"use_current_model", "display"} and not isinstance(value, bool):
            issues.append(ValidationIssue(path, "must be a boolean"))
        elif key in {"ids", "parameters"} and not isinstance(value, list):
            issues.append(ValidationIssue(path, "must be a list"))

    for key in OUTPUT_PATH_ARGUMENTS.get(command, set()):
        if key not in args:
            continue
        resolved = _resolve_plan_path(args[key], workspace_root)
        output_root = (workspace_root / "outputs").resolve()
        if resolved is None:
            continue
        if resolved.suffix.lower() != ".png":
            issues.append(ValidationIssue(f"{base}.args.{key}", "must end in .png"))
        if not _is_within(resolved, output_root):
            issues.append(
                ValidationIssue(f"{base}.args.{key}", "must stay within outputs/")
            )

    for key in WORKSPACE_INPUT_PATH_ARGUMENTS.get(command, set()):
        if key not in args:
            continue
        resolved = _resolve_plan_path(args[key], workspace_root)
        if resolved is not None and not _is_within(resolved, workspace_root):
            issues.append(
                ValidationIssue(f"{base}.args.{key}", "must stay within the workspace")
            )

    if (
        command == "cubism_ui.import_psd"
        and "open_mode" in args
        and args["open_mode"] not in {"create_new_model", "create_new_model_legacy_blend"}
    ):
        issues.append(
            ValidationIssue(f"{base}.args.open_mode", "is not a supported open mode")
        )
    if (
        command == "cubism_ui.apply_auto_mesh"
        and "preset" in args
        and args["preset"] not in {"Standard", "DeformationSmall", "DeformationLarge"}
    ):
        issues.append(ValidationIssue(f"{base}.args.preset", "is not a supported preset"))
    if (
        command == "cubism_api.send_log"
        and "type" in args
        and args["type"] not in {"info", "warning"}
    ):
        issues.append(ValidationIssue(f"{base}.args.type", "must be info or warning"))
    if (
        command == "cubism_api.send_log"
        and isinstance(args.get("message"), str)
        and len(args["message"]) > 5000
    ):
        issues.append(
            ValidationIssue(f"{base}.args.message", "must not exceed 5000 characters")
        )
    if command == "cubism_api.get_parameter_values" and isinstance(args.get("ids"), list):
        for index, parameter_id in enumerate(args["ids"]):
            if not isinstance(parameter_id, str) or not parameter_id:
                issues.append(
                    ValidationIssue(
                        f"{base}.args.ids[{index}]",
                        "must be a non-empty string",
                    )
                )
    if command == "cubism_api.set_parameter_values" and isinstance(
        args.get("parameters"), list
    ):
        for index, parameter in enumerate(args["parameters"]):
            item_path = f"{base}.args.parameters[{index}]"
            if not isinstance(parameter, Mapping):
                issues.append(ValidationIssue(item_path, "must be a mapping"))
                continue
            if set(parameter) != {"Id", "Value"}:
                issues.append(ValidationIssue(item_path, "must contain only Id and Value"))
                continue
            if not isinstance(parameter["Id"], str) or not parameter["Id"]:
                issues.append(ValidationIssue(f"{item_path}.Id", "must be a non-empty string"))
            value = parameter["Value"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                issues.append(ValidationIssue(f"{item_path}.Value", "must be numeric"))

    return issues


def _validate_import_verification_order(steps: Sequence[Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    indexed = [(index, step) for index, step in enumerate(steps) if isinstance(step, Mapping)]
    id_to_step = {
        str(step.get("id")): (index, step)
        for index, step in indexed
        if isinstance(step.get("id"), str)
    }

    for auto_index, auto_step in indexed:
        if auto_step.get("command") != "cubism_ui.apply_auto_mesh":
            continue
        prior_imports = [
            (index, step)
            for index, step in indexed
            if index < auto_index and step.get("command") == "cubism_ui.import_psd"
        ]
        if not prior_imports:
            continue
        import_index, import_step = prior_imports[-1]
        import_id = str(import_step.get("id"))

        verified = False
        for verify_index, verify_step in indexed:
            if not import_index < verify_index < auto_index:
                continue
            if verify_step.get("command") != "file.verify_imported_document":
                continue
            args = verify_step.get("args")
            if not isinstance(args, Mapping) or args.get("import_step") != import_id:
                continue
            before_entry = id_to_step.get(str(args.get("before_step")))
            after_entry = id_to_step.get(str(args.get("after_step")))
            if before_entry is None or after_entry is None:
                continue
            before_index, before_step = before_entry
            after_index, after_step = after_entry
            if (
                before_index < import_index
                and before_step.get("command") == "cubism_api.get_documents"
                and import_index < after_index < verify_index
                and after_step.get("command") == "cubism_api.get_document_snapshot"
            ):
                verified = True
                break

        if not verified:
            issues.append(
                ValidationIssue(
                    f"steps[{auto_index}].command",
                    "auto-mesh after import requires before GetDocuments, after document "
                    "snapshot, and verify_imported_document in that order",
                )
            )
    return issues


def validate_action_plan(
    plan: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    root = (workspace_root or Path.cwd()).resolve()

    if plan.get("schema_version") != 1:
        issues.append(ValidationIssue("schema_version", "must equal 1"))

    project = plan.get("project")
    if not isinstance(project, str) or not project.strip():
        issues.append(ValidationIssue("project", "must be a non-empty string"))

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        issues.append(ValidationIssue("steps", "must be a non-empty list"))
        return issues

    seen_ids: set[str] = set()
    for index, raw_step in enumerate(steps):
        base = f"steps[{index}]"
        if not isinstance(raw_step, Mapping):
            issues.append(ValidationIssue(base, "must be a mapping"))
            continue

        step_id = raw_step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            issues.append(ValidationIssue(f"{base}.id", "must be a non-empty string"))
        elif step_id in seen_ids:
            issues.append(ValidationIssue(f"{base}.id", f"duplicate id: {step_id}"))
        else:
            seen_ids.add(step_id)

        mode = raw_step.get("mode")
        if mode not in ALLOWED_MODES:
            issues.append(
                ValidationIssue(
                    f"{base}.mode",
                    f"must be one of {sorted(ALLOWED_MODES)}",
                )
            )
            continue

        args = raw_step.get("args", {})
        if not isinstance(args, Mapping):
            issues.append(ValidationIssue(f"{base}.args", "must be a mapping"))
            args = {}

        command = raw_step.get("command")
        if mode == "manual_checkpoint":
            instruction = raw_step.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                issues.append(
                    ValidationIssue(f"{base}.instruction", "is required for manual_checkpoint")
                )
            if command is not None:
                issues.append(
                    ValidationIssue(f"{base}.command", "must be omitted for manual_checkpoint")
                )
            continue

        if not isinstance(command, str) or not command:
            issues.append(ValidationIssue(f"{base}.command", "must be a non-empty string"))
            continue

        if any(token in command.lower() for token in FORBIDDEN_COMMAND_TOKENS):
            issues.append(ValidationIssue(f"{base}.command", "contains a forbidden UI primitive"))

        if command not in ALLOWED_COMMANDS[mode]:
            issues.append(
                ValidationIssue(
                    f"{base}.command",
                    f"command is not allowed for mode {mode}: {command}",
                )
            )

        if command in COMMAND_ARGUMENTS:
            issues.extend(_validate_command_arguments(command, args, base, root))

        if mode == "ui_macro":
            for arg_path, key in _walk_mapping_keys(args, f"{base}.args"):
                if key.lower() in FORBIDDEN_UI_ARGUMENT_KEYS:
                    issues.append(
                        ValidationIssue(arg_path, "arbitrary screen coordinates are forbidden")
                    )

        recovery = raw_step.get("recovery")
        if recovery is not None:
            if not isinstance(recovery, Mapping):
                issues.append(ValidationIssue(f"{base}.recovery", "must be a mapping"))
            elif recovery.get("command") != "cubism_ui.undo":
                issues.append(
                    ValidationIssue(
                        f"{base}.recovery.command",
                        "only cubism_ui.undo is supported",
                    )
                )
            elif command != "cubism_ui.apply_auto_mesh":
                issues.append(
                    ValidationIssue(
                        f"{base}.recovery",
                        "undo recovery is allowed only for cubism_ui.apply_auto_mesh",
                    )
                )

    issues.extend(_validate_import_verification_order(steps))
    return issues


def require_valid_action_plan(
    plan: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
) -> None:
    issues = validate_action_plan(plan, workspace_root=workspace_root)
    if issues:
        raise ActionPlanError(issues)
