from __future__ import annotations

import argparse
import json
import platform
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tools.asset_pipeline_common import resolve_inside_base
from tools.cubism_ui import DEFAULT_PROCESS_NAME, DEFAULT_WINDOW_TITLE


class DiagnosticFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class WindowCandidate:
    process_id: int
    executable_name: str
    executable: str | None
    title: str
    class_name: str = ""
    enabled: bool = True
    visible: bool = True
    foreground: bool = False
    rect: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class CubismIdentity:
    process_id: int
    executable_name: str
    executable: str | None
    title: str
    foreground: bool
    rect: tuple[int, int, int, int] | None


@dataclass
class SnapshotNode:
    name: str
    automation_id: str
    control_type: str
    class_name: str
    enabled: bool
    visible: bool
    process_id: int
    supported_patterns: list[str] = field(default_factory=list)
    children: list[SnapshotNode] = field(default_factory=list)


class ControlTreeBackend(Protocol):
    def discover(self) -> CubismIdentity: ...

    def collect_roots(self, identity: CubismIdentity) -> list[SnapshotNode]: ...

    def capture_screenshot(self, path: Path) -> None: ...


_WINDOWS_PATH = re.compile(r"(?i)(?:[A-Z]:\\|\\\\)[^\r\n\t\"<>|]+")
_POSIX_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+(?:/[^\s]*)?")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(token|password|passwd|credential|secret|api[-_ ]?key)\b\s*[:=]\s*[^\s,;]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\b(?:authorization\s*:\s*)?bearer\s+[^\s,;]+")


def sanitize_control_text(value: str, *, control_type: str = "") -> str:
    if control_type.casefold() == "edit":
        return "<redacted-input>"
    sanitized = _WINDOWS_PATH.sub("<redacted-path>", value)
    sanitized = _POSIX_USER_PATH.sub("<redacted-path>", sanitized)
    sanitized = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)
    sanitized = _BEARER_TOKEN.sub("Bearer <redacted>", sanitized)
    return sanitized[:500]


def identify_cubism_window(
    candidates: Sequence[WindowCandidate],
    *,
    process_name: str = DEFAULT_PROCESS_NAME,
    window_title: str = DEFAULT_WINDOW_TITLE,
) -> CubismIdentity:
    process_pattern = re.compile(process_name, re.IGNORECASE)
    fixed_allowlist = re.compile(DEFAULT_PROCESS_NAME, re.IGNORECASE)
    title_pattern = re.compile(window_title, re.IGNORECASE)
    titled = [candidate for candidate in candidates if title_pattern.search(candidate.title)]
    matches = [
        candidate
        for candidate in titled
        if fixed_allowlist.fullmatch(candidate.executable_name)
        and process_pattern.fullmatch(candidate.executable_name)
        and candidate.visible
    ]
    if not matches:
        if titled:
            raise DiagnosticFailure(
                "wrong_process",
                "Cubism-like window title was found, but its executable name was not allowlisted",
            )
        if any(
            fixed_allowlist.fullmatch(candidate.executable_name)
            and process_pattern.fullmatch(candidate.executable_name)
            for candidate in candidates
        ):
            raise DiagnosticFailure(
                "window_not_found", "Cubism Editor process has no matching window"
            )
        raise DiagnosticFailure("cubism_not_found", "Cubism Editor process was not found")
    if len(matches) != 1:
        pids = sorted({candidate.process_id for candidate in matches})
        raise DiagnosticFailure("wrong_process", f"multiple Cubism Editor windows matched: {pids}")
    match = matches[0]
    return CubismIdentity(
        process_id=match.process_id,
        executable_name=match.executable_name,
        executable=match.executable,
        title=match.title,
        foreground=match.foreground,
        rect=match.rect,
    )


def _node_records(node: SnapshotNode, parent_path: str) -> list[dict[str, Any]]:
    name = sanitize_control_text(node.name, control_type=node.control_type)
    segment = name or node.automation_id or node.control_type or "unknown"
    current_path = f"{parent_path}/{segment}" if parent_path else segment
    record = {
        "name": name,
        "automation_id": sanitize_control_text(node.automation_id),
        "control_type": node.control_type,
        "class_name": node.class_name,
        "enabled": node.enabled,
        "visible": node.visible,
        "process_id": node.process_id,
        "parent_path": parent_path,
        "supported_patterns": sorted(set(node.supported_patterns)),
    }
    records = [record]
    for child in node.children:
        if child.process_id == node.process_id:
            records.extend(_node_records(child, current_path))
    return records


def serialize_control_tree(
    roots: Sequence[SnapshotNode], identity: CubismIdentity
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for root in roots:
        if root.process_id == identity.process_id:
            records.extend(_node_records(root, ""))
    return {
        "schema_version": 1,
        "process_id": identity.process_id,
        "executable_name": identity.executable_name,
        "window_title": sanitize_control_text(identity.title),
        "controls": records,
    }


def observed_labels(snapshot: dict[str, Any]) -> list[str]:
    controls = snapshot.get("controls", [])
    if not isinstance(controls, list):
        return []
    return [
        str(control.get("name"))
        for control in controls
        if isinstance(control, dict) and control.get("name") not in {None, "", "<redacted-input>"}
    ]


class WindowsControlTreeBackend:
    def __init__(
        self,
        *,
        process_name: str = DEFAULT_PROCESS_NAME,
        window_title: str = DEFAULT_WINDOW_TITLE,
    ) -> None:
        if platform.system() != "Windows":
            raise DiagnosticFailure("unsupported_version", "UIA diagnostics require Windows")
        try:
            import psutil  # type: ignore[import-untyped]
            from pywinauto import Desktop  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DiagnosticFailure(
                "control_not_exposed_by_uia",
                'install Windows UI dependencies with: pip install -e ".[windows]"',
            ) from exc
        self._psutil = psutil
        self._desktop = Desktop(backend="uia")
        self._process_name = process_name
        self._window_title = window_title
        self._identity: CubismIdentity | None = None

    @staticmethod
    def _text(control: Any) -> str:
        try:
            value = control.window_text()
        except Exception:
            value = ""
        if value:
            return str(value)
        try:
            return str(control.element_info.name or "")
        except Exception:
            return ""

    def _candidates(self) -> list[WindowCandidate]:
        candidates: list[WindowCandidate] = []
        for window in self._desktop.windows():
            try:
                process_id = int(window.element_info.process_id)
                process = self._psutil.Process(process_id)
                rectangle = window.rectangle()
                rect = (rectangle.left, rectangle.top, rectangle.right, rectangle.bottom)
                candidates.append(
                    WindowCandidate(
                        process_id=process_id,
                        executable_name=str(process.name()),
                        executable=str(process.exe()),
                        title=self._text(window),
                        class_name=str(window.element_info.class_name or ""),
                        enabled=bool(window.is_enabled()),
                        visible=bool(window.is_visible()),
                        foreground=bool(window.has_focus()),
                        rect=rect,
                    )
                )
            except (AttributeError, OSError, TypeError, ValueError, self._psutil.Error):
                continue
        return candidates

    def discover(self) -> CubismIdentity:
        self._identity = identify_cubism_window(
            self._candidates(),
            process_name=self._process_name,
            window_title=self._window_title,
        )
        return self._identity

    def _convert(self, control: Any, process_id: int) -> SnapshotNode:
        try:
            info = control.element_info
            node_pid = int(info.process_id)
        except (AttributeError, TypeError, ValueError):
            node_pid = process_id
        patterns = [
            label
            for label, attribute in (
                ("Invoke", "invoke"),
                ("SelectionItem", "select"),
                ("Value", "set_edit_text"),
                ("Toggle", "toggle"),
                ("ExpandCollapse", "expand"),
            )
            if hasattr(control, attribute)
        ]
        try:
            children = [
                self._convert(child, process_id)
                for child in control.children()
                if int(child.element_info.process_id) == process_id
            ]
        except Exception:
            children = []
        try:
            enabled = bool(control.is_enabled())
        except Exception:
            enabled = False
        try:
            visible = bool(control.is_visible())
        except Exception:
            visible = False
        return SnapshotNode(
            name=self._text(control),
            automation_id=str(getattr(control.element_info, "automation_id", "") or ""),
            control_type=str(getattr(control.element_info, "control_type", "") or ""),
            class_name=str(getattr(control.element_info, "class_name", "") or ""),
            enabled=enabled,
            visible=visible,
            process_id=node_pid,
            supported_patterns=patterns,
            children=children,
        )

    def collect_roots(self, identity: CubismIdentity) -> list[SnapshotNode]:
        roots: list[SnapshotNode] = []
        for window in self._desktop.windows():
            try:
                if int(window.element_info.process_id) != identity.process_id:
                    continue
                roots.append(self._convert(window, identity.process_id))
            except Exception:
                continue
        if not roots:
            raise DiagnosticFailure(
                "control_not_exposed_by_uia",
                "Cubism top window/control tree was not exposed by UIA",
            )
        return roots

    def capture_screenshot(self, path: Path) -> None:
        if self._identity is None:
            raise DiagnosticFailure(
                "window_not_found", "Cubism identity must be verified before a screenshot"
            )
        title_pattern = re.compile(self._window_title, re.IGNORECASE)
        candidates: list[tuple[int, Any]] = []
        for window in self._desktop.windows():
            try:
                if int(window.element_info.process_id) != self._identity.process_id:
                    continue
                if not title_pattern.search(self._text(window)):
                    continue
                process = self._psutil.Process(self._identity.process_id)
                if not re.fullmatch(DEFAULT_PROCESS_NAME, str(process.name()), re.IGNORECASE):
                    continue
                rectangle = window.rectangle()
                area = max(0, rectangle.right - rectangle.left) * max(
                    0, rectangle.bottom - rectangle.top
                )
                candidates.append((area, window))
            except Exception:
                continue
        if candidates:
            path.parent.mkdir(parents=True, exist_ok=True)
            max(candidates, key=lambda item: item[0])[1].capture_as_image().save(path)
            return
        raise DiagnosticFailure("window_not_found", "verified Cubism window is unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a sanitized Cubism UIA control tree")
    parser.add_argument("--output", type=Path, default=Path("outputs/cubism-control-tree.json"))
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "mode": "dry-run",
                    "operation": "capture sanitized Cubism UIA control tree",
                    "output": str(args.output),
                    "dependencies": ["Windows", "psutil", "pywinauto"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        output = resolve_inside_base(args.base_dir, str(args.output), "control-tree output")
        backend = WindowsControlTreeBackend(
            process_name=args.process_name, window_title=args.window_title
        )
        identity = backend.discover()
        payload = serialize_control_tree(backend.collect_roots(identity), identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except (ValueError, DiagnosticFailure, OSError) as exc:
        code = exc.code if isinstance(exc, DiagnosticFailure) else "control_not_exposed_by_uia"
        print(json.dumps({"status": "failed", "failure": code, "error": str(exc)}))
        return 1
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
