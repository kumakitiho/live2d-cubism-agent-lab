from __future__ import annotations

import argparse
import asyncio
import ctypes
import ipaddress
import json
import platform
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tools.asset_pipeline_common import resolve_inside_base
from tools.cubism_api import (
    ConnectionOptions,
    CubismAPIError,
    CubismAPISession,
    build_named_operation,
)
from tools.cubism_control_tree_snapshot import (
    ControlTreeBackend,
    DiagnosticFailure,
    WindowsControlTreeBackend,
    observed_labels,
    serialize_control_tree,
)
from tools.cubism_ui import DEFAULT_PROCESS_NAME, DEFAULT_WINDOW_TITLE
from tools.cubism_ui_profiles import select_profile

FAILURE_CODES = {
    "cubism_not_found",
    "wrong_process",
    "window_not_found",
    "unsupported_version",
    "unsupported_language",
    "external_api_unreachable",
    "external_api_not_approved",
    "wrong_edit_mode",
    "dialog_not_found",
    "control_not_exposed_by_uia",
    "ambiguous_control",
    "shortcut_failed",
    "import_verification_failed",
    "auto_mesh_verification_failed",
    "undo_failed",
    "save_as_dialog_opened",
    "timeout",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def require_loopback_host(host: str) -> None:
    value = host.strip().casefold()
    if value == "localhost":
        return
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(
            "Cubism External API host must be a numeric loopback or localhost"
        ) from exc
    if not address.is_loopback:
        raise ValueError("Cubism External API host must be loopback")


def _platform_report() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "windows_version": platform.platform() if platform.system() == "Windows" else None,
    }


def _screen_report() -> dict[str, Any]:
    width: int | None = None
    height: int | None = None
    dpi_scale: float | None = None
    monitors: list[dict[str, Any]] = []
    if platform.system() == "Windows":
        try:
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            dpi = int(user32.GetDpiForSystem()) if hasattr(user32, "GetDpiForSystem") else 96
            dpi_scale = round(dpi / 96, 3)
            monitors.append({"index": 0, "width": width, "height": height, "primary": True})
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return {
        "width": width,
        "height": height,
        "dpi_scale": dpi_scale,
        "monitors": monitors,
    }


def _relative_output(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(base_dir.resolve()).as_posix()


def _current_document_uid(documents: object, model_uid: object) -> str | None:
    if not isinstance(documents, list) or not model_uid:
        return None
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        views = document.get("Views", [])
        if isinstance(views, list) and any(
            isinstance(view, Mapping) and view.get("ModelUID") == model_uid for view in views
        ):
            uid = document.get("DocumentUID")
            return str(uid) if uid else None
    return None


async def _probe_external_api_async(options: ConnectionOptions) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reachable": False,
        "endpoint": options.url,
        "approved": None,
        "current_edit_mode": None,
        "current_document_uid": None,
        "current_model_uid": None,
        "error": None,
    }
    try:
        async with CubismAPISession(options) as session:
            result["reachable"] = True
            approval = await session.run(build_named_operation("cubism_api.get_approval"))
            result["approved"] = bool(approval.get("approved"))
            if not result["approved"]:
                result["error"] = "external_api_not_approved"
                return result
            snapshot = await session.run(build_named_operation("cubism_api.get_document_snapshot"))
            edit_mode = await session.run(build_named_operation("cubism_api.get_current_edit_mode"))
            response = snapshot.get("response", {})
            documents_data = response.get("Documents", {}) if isinstance(response, Mapping) else {}
            current_model_data = (
                response.get("CurrentModel", {}) if isinstance(response, Mapping) else {}
            )
            documents = (
                documents_data.get("ModelingDocuments", [])
                if isinstance(documents_data, Mapping)
                else []
            )
            model_uid = (
                current_model_data.get("ModelUID")
                if isinstance(current_model_data, Mapping)
                else None
            )
            edit_response = edit_mode.get("response", {})
            edit_data = edit_response.get("Data", {}) if isinstance(edit_response, Mapping) else {}
            result["current_model_uid"] = str(model_uid) if model_uid else None
            result["current_document_uid"] = _current_document_uid(documents, model_uid)
            if isinstance(edit_data, Mapping):
                mode = edit_data.get("EditMode") or edit_data.get("CurrentEditMode")
                result["current_edit_mode"] = str(mode) if mode else None
    except (CubismAPIError, OSError, TimeoutError, ValueError) as exc:
        result["error"] = f"external_api_unreachable: {type(exc).__name__}: {exc}"
    return result


def probe_external_api(options: ConnectionOptions) -> dict[str, Any]:
    return asyncio.run(_probe_external_api_async(options))


def _empty_api(endpoint: str) -> dict[str, Any]:
    return {
        "reachable": False,
        "endpoint": endpoint,
        "approved": None,
        "current_edit_mode": None,
        "current_document_uid": None,
        "current_model_uid": None,
        "error": "not attempted because Cubism Editor identity was not verified",
    }


def _empty_cubism() -> dict[str, Any]:
    return {
        "detected": False,
        "executable": None,
        "executable_name": None,
        "version": None,
        "process_id": None,
        "window_title": None,
        "language_guess": None,
        "workspace_guess": None,
        "foreground": None,
        "window_rect": None,
    }


def _empty_uia(snapshot_file: str | None) -> dict[str, Any]:
    return {
        "backend": "uia",
        "top_window_detected": False,
        "dialog_labels": [],
        "control_type_counts": {},
        "known_controls": [],
        "missing_controls": [],
        "profile_match": None,
        "profile_candidates": [],
        "profile_reasons": [],
        "snapshot_file": snapshot_file,
    }


def _dry_run_plan(
    output: Path, control_tree_output: Path | None, screenshot: Path | None
) -> dict[str, Any]:
    return {
        "status": "planned",
        "mode": "dry-run",
        "operations": [
            "verify CubismEditor executable name and window title",
            "probe External API approval and current document/model/edit mode",
            "capture sanitized process-scoped UIA control tree",
            "select an English or Japanese Cubism UI profile",
        ],
        "planned_outputs": {
            "report": str(output),
            "control_tree": str(control_tree_output) if control_tree_output else None,
            "screenshot": str(screenshot) if screenshot else None,
        },
        "dependencies": ["Windows", "psutil", "pywinauto", "PyAutoGUI", "websockets"],
        "external_connections": False,
        "files_written": False,
    }


def run_environment_probe(
    *,
    output: Path = Path("outputs/cubism-environment-report.yaml"),
    control_tree_output: Path | None = None,
    screenshot: Path | None = None,
    base_dir: Path | None = None,
    execute: bool = False,
    backend: ControlTreeBackend | None = None,
    api_probe: Callable[[ConnectionOptions], dict[str, Any]] = probe_external_api,
    api_options: ConnectionOptions | None = None,
    timestamp: str | None = None,
    platform_report: dict[str, Any] | None = None,
    screen_report: dict[str, Any] | None = None,
    process_name: str = DEFAULT_PROCESS_NAME,
    window_title: str = DEFAULT_WINDOW_TITLE,
) -> dict[str, Any]:
    if not execute:
        return _dry_run_plan(output, control_tree_output, screenshot)

    root = (base_dir or Path.cwd()).resolve()
    report_path = resolve_inside_base(root, str(output), "environment report output")
    tree_path = (
        resolve_inside_base(root, str(control_tree_output), "control-tree output")
        if control_tree_output is not None
        else None
    )
    screenshot_path = (
        resolve_inside_base(root, str(screenshot), "diagnostic screenshot output")
        if screenshot is not None
        else None
    )
    options = api_options or ConnectionOptions()
    require_loopback_host(options.host)
    platform_data = platform_report or _platform_report()
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp or _utc_now(),
        "platform": platform_data["platform"],
        "windows_version": platform_data.get("windows_version"),
        "screen": screen_report or _screen_report(),
        "cubism": _empty_cubism(),
        "external_api": _empty_api(options.url),
        "uia": _empty_uia(_relative_output(tree_path, root)),
        "shortcuts": {
            "available": ["Ctrl+O", "Ctrl+A", "Ctrl+Shift+A", "Ctrl+Z"],
            "tested": False,
            "results": [],
        },
        "diagnosis": {
            "status": "blocked",
            "blockers": [],
            "warnings": [],
            "recommended_next_action": "Start Cubism Editor and rerun with --execute.",
        },
    }
    active_backend = backend
    snapshot: dict[str, Any] | None = None
    try:
        active_backend = active_backend or WindowsControlTreeBackend(
            process_name=process_name, window_title=window_title
        )
        identity = active_backend.discover()
        report["cubism"].update(
            {
                "detected": True,
                "executable": identity.executable,
                "executable_name": identity.executable_name,
                "process_id": identity.process_id,
                "window_title": identity.title,
                "foreground": identity.foreground,
                "window_rect": list(identity.rect) if identity.rect else None,
            }
        )
        roots = active_backend.collect_roots(identity)
        snapshot = serialize_control_tree(roots, identity)
        labels = observed_labels(snapshot)
        selection = select_profile(
            identity.title,
            labels,
            version_evidence=f"{identity.title} {identity.executable_name}",
        )
        counts = Counter(
            str(item.get("control_type", ""))
            for item in snapshot["controls"]
            if isinstance(item, Mapping)
        )
        dialog_labels = [
            str(item.get("name"))
            for item in snapshot["controls"]
            if isinstance(item, Mapping) and item.get("control_type") in {"Dialog", "Window"}
        ]
        report["uia"].update(
            {
                "top_window_detected": True,
                "dialog_labels": dialog_labels,
                "control_type_counts": dict(sorted(counts.items())),
                "known_controls": sorted(set(labels))[:100],
                "missing_controls": [] if selection.profile else ["language/profile evidence"],
                "profile_match": selection.profile.profile_id if selection.profile else None,
                "profile_candidates": list(selection.candidates),
                "profile_reasons": list(selection.reasons),
            }
        )
        if selection.profile:
            report["cubism"]["language_guess"] = selection.profile.language
            report["cubism"]["version"] = selection.profile.profile_id.split("-")[1]
        else:
            report["diagnosis"]["blockers"].append(
                selection.failure_code or "unsupported_language"
            )
            report["diagnosis"]["warnings"].extend(selection.reasons)

        if tree_path is not None:
            tree_path.parent.mkdir(parents=True, exist_ok=True)
            tree_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        report["external_api"] = api_probe(options)
        if not report["external_api"].get("reachable"):
            report["diagnosis"]["blockers"].append("external_api_unreachable")
        elif not report["external_api"].get("approved"):
            report["diagnosis"]["blockers"].append("external_api_not_approved")
        if report["external_api"].get("current_edit_mode"):
            report["cubism"]["workspace_guess"] = report["external_api"]["current_edit_mode"]
    except DiagnosticFailure as exc:
        report["diagnosis"]["blockers"].append(exc.code)
        report["diagnosis"]["warnings"].append(str(exc))
    except Exception as exc:
        report["diagnosis"]["blockers"].append("control_not_exposed_by_uia")
        report["diagnosis"]["warnings"].append(f"{type(exc).__name__}: {exc}")

    if report["diagnosis"]["blockers"]:
        report["diagnosis"]["status"] = "blocked"
        report["diagnosis"]["recommended_next_action"] = (
            "Review blockers, the sanitized control tree, and the diagnostic screenshot."
        )
        if screenshot_path is not None and active_backend is not None:
            try:
                active_backend.capture_screenshot(screenshot_path)
            except Exception as exc:
                report["diagnosis"]["warnings"].append(f"screenshot failed: {exc}")
    else:
        report["diagnosis"]["status"] = "ready"
        report["diagnosis"]["recommended_next_action"] = (
            "Review the report, then run the named smoke test with --execute if approved."
        )
        if screenshot_path is not None and active_backend is not None:
            active_backend.capture_screenshot(screenshot_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose the local Cubism Editor environment")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cubism-environment-report.yaml"),
    )
    parser.add_argument("--control-tree-output", type=Path)
    parser.add_argument("--screenshot", type=Path)
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
        result = run_environment_probe(
            output=args.output,
            control_tree_output=args.control_tree_output,
            screenshot=args.screenshot,
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.execute and result.get("diagnosis", {}).get("status") == "blocked":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
