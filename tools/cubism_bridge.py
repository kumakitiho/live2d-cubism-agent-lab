from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.action_plan import load_action_plan, require_valid_action_plan
from tools.artifact_validation import (
    load_yaml_mapping,
    validate_character_spec,
    validate_layer_map,
)
from tools.cubism_api import (
    ConnectionOptions,
    CubismAPISession,
    build_named_operation,
    plan_operation,
)
from tools.cubism_ui import (
    DEFAULT_PROCESS_NAME,
    DEFAULT_WINDOW_TITLE,
    MacroExecutionError,
    build_named_macro,
    execute_actions,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_artifact(command: str, args: Mapping[str, Any]) -> dict[str, Any]:
    path_value = args.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{command} requires args.path")
    path = Path(path_value)
    data = load_yaml_mapping(path)
    if command == "file.validate_character_spec":
        issues = validate_character_spec(data)
    elif command == "file.validate_layer_map":
        issues = validate_layer_map(data)
    else:
        raise ValueError(f"unknown file command: {command}")
    return {
        "status": "completed" if not issues else "error",
        "path": str(path.resolve()),
        "issues": [issue.format() for issue in issues],
    }


def _find_step_result(steps: Sequence[Mapping[str, Any]], step_id: str) -> Mapping[str, Any]:
    for step in steps:
        if step.get("id") == step_id:
            return step
    raise ValueError(f"referenced step result not found: {step_id}")


def _verify_imported_document(
    args: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    execute: bool,
) -> dict[str, Any]:
    if not execute:
        return {
            "status": "planned",
            "verification": "compare ModelingDocuments count and current ModelUID after import",
        }

    before = _find_step_result(steps, str(args["before_step"]))
    imported = _find_step_result(steps, str(args["import_step"]))
    after = _find_step_result(steps, str(args["after_step"]))

    import_result = imported.get("result")
    if not isinstance(import_result, Mapping) or "import_psd" not in import_result.get(
        "applied_mutations", []
    ):
        return {"status": "error", "issues": ["import macro did not report import_psd"]}

    before_result = before.get("result")
    after_result = after.get("result")
    if not isinstance(before_result, Mapping) or not isinstance(after_result, Mapping):
        return {"status": "error", "issues": ["API step result is missing"]}

    before_response = before_result.get("response")
    after_response = after_result.get("response")
    if not isinstance(before_response, Mapping) or not isinstance(after_response, Mapping):
        return {"status": "error", "issues": ["API response is missing"]}

    before_data = before_response.get("Data")
    after_documents = after_response.get("Documents")
    current_model = after_response.get("CurrentModel")
    if not isinstance(before_data, Mapping) or not isinstance(after_documents, Mapping):
        return {"status": "error", "issues": ["document snapshot has an invalid shape"]}

    before_models = before_data.get("ModelingDocuments", [])
    after_models = after_documents.get("ModelingDocuments", [])
    if not isinstance(before_models, list) or not isinstance(after_models, list):
        return {"status": "error", "issues": ["ModelingDocuments must be arrays"]}
    if len(after_models) != len(before_models) + 1:
        return {
            "status": "error",
            "issues": [
                "expected exactly one new ModelingDocument "
                f"(before={len(before_models)}, after={len(after_models)})"
            ],
        }

    before_uids = {
        document.get("DocumentUID")
        for document in before_models
        if isinstance(document, Mapping) and document.get("DocumentUID")
    }
    if len(before_uids) != len(before_models):
        return {
            "status": "error",
            "issues": ["before snapshot has missing or duplicate DocumentUID values"],
        }
    new_documents = [
        document
        for document in after_models
        if isinstance(document, Mapping) and document.get("DocumentUID") not in before_uids
    ]
    if len(new_documents) != 1:
        return {
            "status": "error",
            "issues": [f"expected one new DocumentUID, found {len(new_documents)}"],
        }

    current_uid = current_model.get("ModelUID") if isinstance(current_model, Mapping) else None
    if not current_uid:
        return {"status": "error", "issues": ["current ModelUID is missing after import"]}

    new_document = new_documents[0]
    views = new_document.get("Views", [])
    current_is_modeled = isinstance(views, list) and any(
        isinstance(view, Mapping) and view.get("ModelUID") == current_uid for view in views
    )
    if not current_is_modeled:
        return {
            "status": "error",
            "issues": ["current ModelUID is not present in the new ModelingDocument"],
        }

    return {
        "status": "completed",
        "before_modeling_documents": len(before_models),
        "after_modeling_documents": len(after_models),
        "new_document_uid": new_document.get("DocumentUID"),
        "current_model_uid_verified": True,
    }


def _run_file_command(
    command: str,
    args: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    execute: bool,
) -> dict[str, Any]:
    if command == "file.verify_imported_document":
        return _verify_imported_document(args, steps, execute=execute)
    return _validate_artifact(command, args)


def _render_report(run: Mapping[str, Any]) -> str:
    lines = [
        "# Cubism Action Plan Report",
        "",
        f"- Project: `{run.get('project', 'unknown')}`",
        f"- Mode: `{run.get('mode')}`",
        f"- Status: `{run.get('status')}`",
        f"- Started: `{run.get('started_at')}`",
        f"- Updated: `{run.get('updated_at')}`",
        "",
    ]
    for result in run.get("steps", []):
        lines.extend(
            [
                f"## {result.get('id', 'unknown')}",
                "",
                f"- Mode: `{result.get('mode')}`",
                f"- Command: `{result.get('command', '-')}`",
                f"- Status: `{result.get('status')}`",
                "",
                "```json",
                json.dumps(result.get("result", {}), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _write_report(path: Path, run: dict[str, Any]) -> None:
    run["updated_at"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(run), encoding="utf-8")


def _can_run_undo_recovery(command: Any, result: Mapping[str, Any]) -> bool:
    applied = result.get("applied_mutations", [])
    return command == "cubism_ui.undo" and isinstance(applied, list) and "auto_mesh" in applied


def run_action_plan_data(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    report_path: Path = Path("outputs/action_plan_report.md"),
    manual_policy: str = "stop",
    window_title: str = DEFAULT_WINDOW_TITLE,
    process_name: str = DEFAULT_PROCESS_NAME,
    api_options: ConnectionOptions | None = None,
) -> dict[str, Any]:
    require_valid_action_plan(plan)
    if manual_policy not in {"stop", "skip"}:
        raise ValueError("manual_policy must be stop or skip")
    return asyncio.run(
        _run_action_plan_data_async(
            plan,
            execute=execute,
            report_path=report_path,
            manual_policy=manual_policy,
            window_title=window_title,
            process_name=process_name,
            api_options=api_options or ConnectionOptions(),
        )
    )


async def _run_action_plan_data_async(
    plan: Mapping[str, Any],
    *,
    execute: bool,
    report_path: Path,
    manual_policy: str,
    window_title: str,
    process_name: str,
    api_options: ConnectionOptions,
) -> dict[str, Any]:

    run: dict[str, Any] = {
        "project": plan["project"],
        "mode": "execute" if execute else "dry-run",
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "steps": [],
    }
    _write_report(report_path, run)
    api_session: CubismAPISession | None = None
    api_session_open = False

    try:
        for step in plan["steps"]:
            step_id = str(step["id"])
            mode = str(step["mode"])
            command = step.get("command")
            args = dict(step.get("args", {}))
            entry: dict[str, Any] = {
                "id": step_id,
                "mode": mode,
                "command": command,
                "status": "running",
                "result": {},
            }
            run["steps"].append(entry)

            if mode == "manual_checkpoint":
                entry["result"] = {"instruction": step["instruction"]}
                if not execute:
                    entry["status"] = "planned"
                    _write_report(report_path, run)
                    continue
                if manual_policy == "skip":
                    entry["status"] = "skipped"
                    _write_report(report_path, run)
                    continue
                entry["status"] = "pending_manual_review"
                run["status"] = "stopped_for_manual_checkpoint"
                _write_report(report_path, run)
                return run

            try:
                if mode == "file":
                    result = _run_file_command(
                        str(command),
                        args,
                        run["steps"],
                        execute=execute,
                    )
                    if result["status"] == "error":
                        entry["status"] = "error"
                        entry["result"] = result
                        run["status"] = "error"
                        _write_report(report_path, run)
                        return run
                elif mode == "ui_macro":
                    actions = build_named_macro(str(command), args)
                    failure = args.get("failure_screenshot")
                    failure_path = Path(str(failure)) if failure else None
                    if execute:
                        result = await asyncio.to_thread(
                            execute_actions,
                            actions,
                            execute=True,
                            window_title=window_title,
                            process_name=process_name,
                            failure_screenshot=failure_path,
                        )
                    else:
                        result = execute_actions(
                            actions,
                            execute=False,
                            window_title=window_title,
                            process_name=process_name,
                            failure_screenshot=failure_path,
                        )
                elif mode == "api":
                    operation = build_named_operation(str(command), args)
                    if execute:
                        if api_session is None:
                            api_session = CubismAPISession(api_options)
                            await api_session.__aenter__()
                            api_session_open = True
                        result = await api_session.run(operation)
                    else:
                        result = plan_operation(operation, api_options)
                else:
                    raise ValueError(f"unsupported mode: {mode}")
                entry["result"] = result
                entry["status"] = str(result.get("status", "completed"))
            except MacroExecutionError as exc:
                entry["status"] = "error"
                entry["result"] = exc.result
                recovery = step.get("recovery")
                if execute and isinstance(recovery, Mapping):
                    recovery_command = recovery.get("command")
                    if _can_run_undo_recovery(recovery_command, exc.result):
                        try:
                            entry["recovery"] = await asyncio.to_thread(
                                execute_actions,
                                build_named_macro("cubism_ui.undo", {}),
                                execute=True,
                                window_title=window_title,
                                process_name=process_name,
                            )
                        except Exception as recovery_exc:
                            entry["recovery"] = {
                                "status": "error",
                                "error": f"{type(recovery_exc).__name__}: {recovery_exc}",
                            }
                    elif recovery_command == "cubism_ui.undo":
                        entry["recovery"] = {
                            "status": "skipped",
                            "reason": "auto_mesh mutation was not confirmed as applied",
                        }
                run["status"] = "error"
                _write_report(report_path, run)
                return run
            except Exception as exc:
                entry["status"] = "error"
                entry["result"] = {"error": f"{type(exc).__name__}: {exc}"}
                run["status"] = "error"
                _write_report(report_path, run)
                return run

            _write_report(report_path, run)

        run["status"] = "completed" if execute else "planned"
        _write_report(report_path, run)
        return run
    finally:
        if api_session is not None and api_session_open:
            await api_session.__aexit__(None, None, None)


def _high_level_import_plan(psd_path: Path, open_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": psd_path.stem,
        "steps": [
            {
                "id": "before_documents",
                "mode": "api",
                "command": "cubism_api.get_documents",
            },
            {
                "id": "before_screenshot",
                "mode": "ui_macro",
                "command": "cubism_ui.screenshot",
                "args": {"path": "outputs/import_before.png"},
            },
            {
                "id": "import_psd",
                "mode": "ui_macro",
                "command": "cubism_ui.import_psd",
                "args": {
                    "psd_path": str(psd_path),
                    "open_mode": open_mode,
                    "screenshot": "outputs/import_after.png",
                    "failure_screenshot": "outputs/import_failed.png",
                },
            },
            {
                "id": "after_snapshot",
                "mode": "api",
                "command": "cubism_api.get_document_snapshot",
            },
            {
                "id": "verify_imported_document",
                "mode": "file",
                "command": "file.verify_imported_document",
                "args": {
                    "before_step": "before_documents",
                    "import_step": "import_psd",
                    "after_step": "after_snapshot",
                },
            },
        ],
    }


def _high_level_auto_mesh_plan(preset: str, alpha: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": "current-cubism-model",
        "steps": [
            {
                "id": "apply_auto_mesh",
                "mode": "ui_macro",
                "command": "cubism_ui.apply_auto_mesh",
                "args": {
                    "preset": preset,
                    "alpha": alpha,
                    "screenshot": "outputs/auto_mesh_after.png",
                    "failure_screenshot": "outputs/auto_mesh_failed.png",
                },
                "recovery": {"command": "cubism_ui.undo"},
            },
            {
                "id": "visual_review",
                "mode": "manual_checkpoint",
                "instruction": "メッシュ欠け、テクスチャ消失、半透明ゴミを確認する。",
            },
        ],
    }


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/action_plan_report.md"))
    parser.add_argument("--manual-policy", choices=("stop", "skip"), default="stop")
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22033)
    parser.add_argument("--token-file", type=Path, default=Path(".live2d-agent/cubism-token.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-level Cubism UI/API bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    import_psd = sub.add_parser("import-psd-and-verify")
    import_psd.add_argument("psd_path", type=Path)
    import_psd.add_argument(
        "--open-mode",
        choices=("create_new_model", "create_new_model_legacy_blend"),
        default="create_new_model",
    )
    _add_run_options(import_psd)

    auto_mesh = sub.add_parser("apply-auto-mesh-and-capture")
    auto_mesh.add_argument("--preset", default="Standard")
    auto_mesh.add_argument("--alpha", type=int, default=0)
    _add_run_options(auto_mesh)

    run_plan = sub.add_parser("run-action-plan")
    run_plan.add_argument("plan", type=Path)
    _add_run_options(run_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "import-psd-and-verify":
            plan = _high_level_import_plan(args.psd_path, args.open_mode)
        elif args.command == "apply-auto-mesh-and-capture":
            plan = _high_level_auto_mesh_plan(args.preset, args.alpha)
        elif args.command == "run-action-plan":
            plan = load_action_plan(args.plan)
        else:
            parser.error(f"unsupported command: {args.command}")
            return 2

        options = ConnectionOptions(
            host=args.host,
            port=args.port,
            token_file=args.token_file,
        )
        result = run_action_plan_data(
            plan,
            execute=args.execute,
            report_path=args.report,
            manual_policy=args.manual_policy,
            window_title=args.window_title,
            process_name=args.process_name,
            api_options=options,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
