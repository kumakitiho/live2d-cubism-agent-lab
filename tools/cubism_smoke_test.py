from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from tools.asset_pipeline_common import resolve_inside_base
from tools.cubism_api import ConnectionOptions, CubismAPISession, build_named_operation
from tools.cubism_control_tree_snapshot import (
    ControlTreeBackend,
    DiagnosticFailure,
    WindowsControlTreeBackend,
    observed_labels,
    serialize_control_tree,
)
from tools.cubism_environment_probe import require_loopback_host
from tools.cubism_ui import (
    DEFAULT_PROCESS_NAME,
    DEFAULT_WINDOW_TITLE,
    MacroExecutionError,
    UIBackend,
    WindowsCubismBackend,
    build_auto_mesh_actions,
    build_import_psd_actions,
    build_screenshot_actions,
    build_undo_actions,
    execute_actions,
)
from tools.cubism_ui_profiles import CubismUIProfile, select_profile

STAGE_NAMES = (
    "preflight",
    "api_connection",
    "initial_snapshot",
    "psd_import",
    "import_verification",
    "auto_mesh",
    "visual_capture",
    "undo",
    "final_snapshot",
)
STAGE_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "blocked",
    "waiting_for_user",
    "skipped",
}


class SmokeAPI(Protocol):
    def approval(self) -> bool: ...

    def snapshot(self) -> dict[str, Any]: ...


class LiveSmokeAPI:
    def __init__(self, options: ConnectionOptions) -> None:
        self.options = options

    async def _approval(self) -> bool:
        async with CubismAPISession(self.options) as session:
            result = await session.run(build_named_operation("cubism_api.get_approval"))
            return bool(result.get("approved"))

    async def _snapshot(self) -> dict[str, Any]:
        async with CubismAPISession(self.options) as session:
            documents = await session.run(build_named_operation("cubism_api.get_documents"))
            current_model = await session.run(
                build_named_operation("cubism_api.get_current_model_uid")
            )
            edit_mode = await session.run(build_named_operation("cubism_api.get_current_edit_mode"))
        document_response = documents.get("response", {})
        document_data = (
            document_response.get("Data", {}) if isinstance(document_response, Mapping) else {}
        )
        model_response = current_model.get("response", {})
        model_data = model_response.get("Data", {}) if isinstance(model_response, Mapping) else {}
        edit_response = edit_mode.get("response", {})
        edit_data = edit_response.get("Data", {}) if isinstance(edit_response, Mapping) else {}
        return {
            "documents": (
                document_data.get("ModelingDocuments", [])
                if isinstance(document_data, Mapping)
                else []
            ),
            "current_model_uid": (
                model_data.get("ModelUID") if isinstance(model_data, Mapping) else None
            ),
            "current_edit_mode": (
                edit_data.get("EditMode") or edit_data.get("CurrentEditMode")
                if isinstance(edit_data, Mapping)
                else None
            ),
        }

    def approval(self) -> bool:
        return asyncio.run(self._approval())

    def snapshot(self) -> dict[str, Any]:
        return asyncio.run(self._snapshot())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _relative(path: Path, base_dir: Path) -> str:
    return path.resolve().relative_to(base_dir.resolve()).as_posix()


def _document_uids(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    documents = snapshot.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("documents must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        if not isinstance(document, Mapping) or not document.get("DocumentUID"):
            raise ValueError("each document must have a DocumentUID")
        uid = str(document["DocumentUID"])
        if uid in result:
            raise ValueError("DocumentUID values must be unique")
        result[uid] = document
    return result


def verify_import_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    for phase, snapshot in (("before", before), ("after", after)):
        if snapshot.get("current_edit_mode") != "Modeling":
            raise DiagnosticFailure(
                "wrong_edit_mode",
                f"{phase} import edit mode must be Modeling, got "
                f"{snapshot.get('current_edit_mode')!r}",
            )
    before_documents = _document_uids(before)
    after_documents = _document_uids(after)
    new_uids = sorted(set(after_documents) - set(before_documents))
    if len(new_uids) != 1:
        raise DiagnosticFailure(
            "import_verification_failed",
            f"expected exactly one new DocumentUID, found {len(new_uids)}",
        )
    model_uid = after.get("current_model_uid")
    if not model_uid:
        raise DiagnosticFailure(
            "import_verification_failed", "current ModelUID is unavailable after import"
        )
    new_document = after_documents[new_uids[0]]
    views = new_document.get("Views", [])
    if not isinstance(views, list) or not any(
        isinstance(view, Mapping) and view.get("ModelUID") == model_uid for view in views
    ):
        raise DiagnosticFailure(
            "import_verification_failed",
            "current ModelUID is not associated with the new document",
        )
    return {
        "before_document_uids": sorted(before_documents),
        "after_document_uids": sorted(after_documents),
        "new_document_uid": new_uids[0],
        "current_model_uid": str(model_uid),
        "current_edit_mode": after.get("current_edit_mode"),
    }


def _save_control_available(snapshot: Mapping[str, Any], profile: CubismUIProfile) -> bool:
    import re

    controls = snapshot.get("controls", [])
    if not isinstance(controls, list):
        return False
    for control in controls:
        if not isinstance(control, Mapping) or not control.get("enabled"):
            continue
        name = str(control.get("name", ""))
        if any(
            re.search(pattern, name, re.IGNORECASE)
            for pattern in profile.save_control_patterns
        ):
            return True
    return False


def _classify_exception(exc: BaseException, fallback: str) -> str:
    if isinstance(exc, DiagnosticFailure):
        return exc.code
    text = str(exc).casefold()
    if isinstance(exc, MacroExecutionError):
        failed_action = exc.result.get("failed_action", {})
        if isinstance(failed_action, Mapping) and failed_action.get("name") in {
            "hotkey",
            "press_key",
        }:
            return "shortcut_failed"
    if "ambiguous" in text:
        return "ambiguous_control"
    if "not exposed" in text or "named control not found" in text:
        return "control_not_exposed_by_uia"
    if "dialog" in text and ("not found" in text or "did not close" in text):
        return "dialog_not_found"
    if "save" in text and "dialog" in text:
        return "save_as_dialog_opened"
    if "timeout" in text:
        return "timeout"
    return fallback


def _snapshot_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    documents = _document_uids(snapshot)
    return {
        "document_uids": sorted(documents),
        "current_model_uid": snapshot.get("current_model_uid"),
        "current_edit_mode": snapshot.get("current_edit_mode"),
    }


def _macro_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": result.get("status"),
        "completed_actions": result.get("completed_actions", 0),
        "applied_mutations": list(result.get("applied_mutations", [])),
    }
    if result.get("verification_required"):
        summary["verification_required"] = result["verification_required"]
    failed_action = result.get("failed_action")
    if isinstance(failed_action, Mapping):
        summary["failed_action"] = str(failed_action.get("name", "unknown"))
    if result.get("error"):
        summary["error"] = str(result["error"])
    return summary


def _planned_report(psd_path: Path, preset: str, alpha: int, output: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "dry-run",
        "status": "planned",
        "input_psd": str(psd_path),
        "preset": preset,
        "alpha": alpha,
        "output": str(output),
        "stages": {name: {"status": "planned"} for name in STAGE_NAMES},
        "external_connections": False,
        "files_written": False,
        "visual_review_required": True,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(report), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _block_following(report: dict[str, Any], failed_stage: str) -> None:
    failed_index = STAGE_NAMES.index(failed_stage)
    for name in STAGE_NAMES[failed_index + 1 :]:
        if report["stages"][name]["status"] == "planned":
            report["stages"][name] = {
                "status": "blocked",
                "blocked_by": failed_stage,
            }


def run_smoke_test(
    psd_path: Path,
    *,
    preset: str = "Standard",
    alpha: int = 10,
    output: Path = Path("outputs/cubism-smoke-test.yaml"),
    base_dir: Path | None = None,
    execute: bool = False,
    control_backend: ControlTreeBackend | None = None,
    api: SmokeAPI | None = None,
    ui_backend: UIBackend | None = None,
    api_options: ConnectionOptions | None = None,
    timestamp: str | None = None,
    process_name: str = DEFAULT_PROCESS_NAME,
    window_title: str = DEFAULT_WINDOW_TITLE,
) -> dict[str, Any]:
    if not execute:
        return _planned_report(psd_path, preset, alpha, output)
    if not 0 <= alpha <= 255:
        raise ValueError("alpha must be between 0 and 255")
    resolved_psd = psd_path.expanduser().resolve()
    if resolved_psd.suffix.casefold() != ".psd" or not resolved_psd.is_file():
        raise ValueError(f"valid PSD input is required: {resolved_psd}")

    root = (base_dir or Path.cwd()).resolve()
    options = api_options or ConnectionOptions()
    require_loopback_host(options.host)
    report_path = resolve_inside_base(root, str(output), "smoke-test report output")
    evidence_dir = report_path.parent
    import_screenshot = resolve_inside_base(
        root, str(evidence_dir / "cubism-smoke-import.png"), "import screenshot"
    )
    mesh_screenshot = resolve_inside_base(
        root, str(evidence_dir / "cubism-smoke-auto-mesh.png"), "auto-mesh screenshot"
    )
    undo_screenshot = resolve_inside_base(
        root, str(evidence_dir / "cubism-smoke-undo.png"), "undo screenshot"
    )
    failure_screenshot = resolve_inside_base(
        root, str(evidence_dir / "cubism-smoke-failure.png"), "failure screenshot"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp or _utc_now(),
        "mode": "execute",
        "status": "running",
        "input_psd": resolved_psd.name,
        "preset": preset,
        "alpha": alpha,
        "process_id": None,
        "profile_id": None,
        "stages": {name: {"status": "planned"} for name in STAGE_NAMES},
        "evidence": {
            "import_screenshot": _relative(import_screenshot, root),
            "auto_mesh_screenshot": _relative(mesh_screenshot, root),
            "undo_screenshot": _relative(undo_screenshot, root),
            "failure_screenshot": None,
        },
        "visual_review_required": True,
    }
    _write_report(report_path, report)

    active_control = control_backend
    active_ui = ui_backend
    identity: Any = None
    profile: CubismUIProfile | None = None
    before_snapshot: dict[str, Any] | None = None
    auto_mesh_applied = False
    active_api = api or LiveSmokeAPI(options)

    def complete(name: str, details: Mapping[str, Any] | None = None) -> None:
        report["stages"][name] = {"status": "completed", "details": dict(details or {})}
        _write_report(report_path, report)

    def stop(name: str, exc: BaseException, fallback: str, *, waiting: bool = False) -> None:
        code = _classify_exception(exc, fallback)
        report["stages"][name] = {
            "status": "waiting_for_user" if waiting else "failed",
            "failure": code,
            "error": str(exc),
        }
        _block_following(report, name)
        report["status"] = "waiting_for_user" if waiting else "failed"
        if active_control is not None:
            try:
                active_control.capture_screenshot(failure_screenshot)
                report["evidence"]["failure_screenshot"] = _relative(failure_screenshot, root)
            except Exception as screenshot_exc:
                report["stages"][name]["screenshot_error"] = str(screenshot_exc)
        _write_report(report_path, report)

    try:
        report["stages"]["preflight"] = {"status": "running"}
        active_control = active_control or WindowsControlTreeBackend(
            process_name=process_name, window_title=window_title
        )
        identity = active_control.discover()
        tree = serialize_control_tree(active_control.collect_roots(identity), identity)
        selection = select_profile(
            identity.title,
            observed_labels(tree),
            version_evidence=f"{identity.title} {identity.executable_name}",
        )
        if selection.profile is None:
            raise DiagnosticFailure(
                selection.failure_code or "unsupported_language",
                "profile selection failed; "
                f"candidates={selection.candidates}; reasons={selection.reasons}",
            )
        profile = selection.profile
        report["process_id"] = identity.process_id
        report["profile_id"] = profile.profile_id
        complete(
            "preflight",
            {
                "executable_name": identity.executable_name,
                "window_title": identity.title,
                "profile": profile.profile_id,
            },
        )
    except Exception as exc:
        stop("preflight", exc, "cubism_not_found")
        return report

    try:
        report["stages"]["api_connection"] = {"status": "running"}
        if not active_api.approval():
            raise DiagnosticFailure(
                "external_api_not_approved",
                "Enable Allow in Cubism External Application Integration",
            )
        complete("api_connection", {"reachable": True, "approved": True})
    except Exception as exc:
        stop(
            "api_connection",
            exc,
            "external_api_unreachable",
            waiting=isinstance(exc, DiagnosticFailure)
            and exc.code == "external_api_not_approved",
        )
        return report

    try:
        report["stages"]["initial_snapshot"] = {"status": "running"}
        before_snapshot = active_api.snapshot()
        if before_snapshot.get("current_edit_mode") != "Modeling":
            raise DiagnosticFailure(
                "wrong_edit_mode",
                "initial edit mode must be Modeling before import",
            )
        complete("initial_snapshot", _snapshot_summary(before_snapshot))
    except Exception as exc:
        stop("initial_snapshot", exc, "external_api_unreachable")
        return report

    try:
        report["stages"]["psd_import"] = {"status": "running"}
        if active_ui is None:
            active_ui = WindowsCubismBackend(
                window_title=window_title,
                process_name=process_name,
                process_id=identity.process_id,
                profile_id=profile.profile_id,
            )
        import_result = execute_actions(
            build_import_psd_actions(resolved_psd, screenshot=import_screenshot),
            execute=True,
            backend=active_ui,
            failure_screenshot=failure_screenshot,
        )
        if "import_psd" not in import_result.get("applied_mutations", []):
            raise DiagnosticFailure(
                "import_verification_failed", "import macro did not record an import mutation"
            )
        complete("psd_import", _macro_summary(import_result))
    except Exception as exc:
        details = exc.result if isinstance(exc, MacroExecutionError) else None
        stop("psd_import", exc, "dialog_not_found")
        if details:
            report["stages"]["psd_import"]["macro_result"] = _macro_summary(details)
            _write_report(report_path, report)
        return report

    try:
        report["stages"]["import_verification"] = {"status": "running"}
        if before_snapshot is None:
            raise DiagnosticFailure(
                "import_verification_failed",
                "verification_unavailable: initial snapshot is missing",
            )
        after_import = active_api.snapshot()
        verification = verify_import_snapshots(before_snapshot, after_import)
        complete("import_verification", verification)
    except Exception as exc:
        stop("import_verification", exc, "import_verification_failed", waiting=True)
        return report

    try:
        report["stages"]["auto_mesh"] = {"status": "running"}
        mesh_result = execute_actions(
            build_auto_mesh_actions(preset=preset, alpha=alpha, screenshot=mesh_screenshot),
            execute=True,
            backend=active_ui,
            failure_screenshot=failure_screenshot,
        )
        auto_mesh_applied = "auto_mesh" in mesh_result.get("applied_mutations", [])
        if not auto_mesh_applied:
            raise DiagnosticFailure(
                "auto_mesh_verification_failed", "auto-mesh mutation was not confirmed"
            )
        complete(
            "auto_mesh",
            {
                **_macro_summary(mesh_result),
                "dialog_closed": True,
                "visual_review_required": True,
            },
        )
    except Exception as exc:
        recovery: dict[str, Any] | None = None
        if (
            isinstance(exc, MacroExecutionError)
            and "auto_mesh" in exc.result.get("applied_mutations", [])
            and active_ui is not None
        ):
            try:
                recovery_result = execute_actions(
                    [*build_undo_actions(), *build_screenshot_actions(undo_screenshot)],
                    execute=True,
                    backend=active_ui,
                    failure_screenshot=failure_screenshot,
                )
                recovery = _macro_summary(recovery_result)
                recovery["status"] = "completed"
            except Exception as recovery_exc:
                recovery = {
                    "status": "failed",
                    "failure": _classify_exception(recovery_exc, "undo_failed"),
                    "error": str(recovery_exc),
                }
        stop("auto_mesh", exc, "auto_mesh_verification_failed")
        if recovery is not None:
            report["stages"]["auto_mesh"]["recovery"] = recovery
            _write_report(report_path, report)
        return report

    try:
        report["stages"]["visual_capture"] = {"status": "running"}
        if not mesh_screenshot.is_file():
            raise DiagnosticFailure(
                "auto_mesh_verification_failed", "auto-mesh screenshot was not saved"
            )
        complete(
            "visual_capture",
            {"screenshot": _relative(mesh_screenshot, root), "visual_review_required": True},
        )
    except Exception as exc:
        stop("visual_capture", exc, "auto_mesh_verification_failed")
        return report

    try:
        report["stages"]["undo"] = {"status": "running"}
        if not auto_mesh_applied:
            raise DiagnosticFailure("undo_failed", "Undo is forbidden before a confirmed mutation")
        undo_result = execute_actions(
            [*build_undo_actions(), *build_screenshot_actions(undo_screenshot)],
            execute=True,
            backend=active_ui,
            failure_screenshot=failure_screenshot,
        )
        if "undo" not in undo_result.get("applied_mutations", []):
            raise DiagnosticFailure("undo_failed", "Undo mutation was not recorded")
        if not undo_screenshot.is_file():
            raise DiagnosticFailure("undo_failed", "Undo screenshot was not saved")
        complete(
            "undo",
            {
                **_macro_summary(undo_result),
                "screenshot": _relative(undo_screenshot, root),
            },
        )
    except Exception as exc:
        stop("undo", exc, "undo_failed")
        return report

    try:
        report["stages"]["final_snapshot"] = {"status": "running"}
        final_api = active_api.snapshot()
        final_tree = serialize_control_tree(active_control.collect_roots(identity), identity)
        if not _save_control_available(final_tree, profile):
            raise DiagnosticFailure(
                "control_not_exposed_by_uia",
                "verification_unavailable: an enabled Save control was not exposed by UIA",
            )
        complete(
            "final_snapshot",
            {
                **_snapshot_summary(final_api),
                "save_control_enabled": True,
                "undo_attempted": True,
            },
        )
    except Exception as exc:
        stop("final_snapshot", exc, "control_not_exposed_by_uia", waiting=True)
        return report

    report["status"] = "completed"
    _write_report(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a named, staged Cubism smoke test")
    parser.add_argument("psd_path", type=Path)
    parser.add_argument("--preset", default="Standard")
    parser.add_argument("--alpha", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/cubism-smoke-test.yaml"))
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22033)
    parser.add_argument("--token-file", type=Path, default=Path(".live2d-agent/cubism-token.json"))
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke_test(
            args.psd_path,
            preset=args.preset,
            alpha=args.alpha,
            output=args.output,
            base_dir=args.base_dir,
            execute=args.execute,
            api_options=ConnectionOptions(
                host=args.host, port=args.port, token_file=args.token_file
            ),
            process_name=args.process_name,
            window_title=args.window_title,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
