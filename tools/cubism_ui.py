from __future__ import annotations

import argparse
import json
import platform
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tools.cubism_ui_profiles import CubismUIProfile, get_profile

DEFAULT_WINDOW_TITLE = r".*(?:Live2D Cubism|Cubism Editor|Cubism).*"
DEFAULT_PROCESS_NAME = r"^CubismEditor[A-Za-z0-9_-]*\.exe$"
OPEN_MODES = {"create_new_model", "create_new_model_legacy_blend"}

DIALOG_PATTERNS: dict[str, tuple[str, ...]] = {
    "open_file": (r"^Open$", r"^開く$"),
    "model_settings": (r".*Model Settings.*", r".*モデル設定.*"),
    "auto_mesh": (r".*Automatic Mesh generator.*", r".*メッシュの自動生成.*"),
    "save_file": (r"^Save As$", r"^名前を付けて保存$"),
}

MODEL_MODE_PATTERNS: dict[str, tuple[str, ...]] = {
    "create_new_model": (
        r"^Create new model from (?:the )?(?:imported )?PSD(?: file)?\.?$",
        r"^PSDファイルから新規モデルを作成.*$",
    ),
    "create_new_model_legacy_blend": (
        r"^Create new model.*5\.2.*$",
        r"^.*5\.2.*(?:ブレンド|blend).*$",
    ),
}

PRESET_LABELS: dict[str, tuple[str, ...]] = {
    "Standard": ("Standard", "標準"),
    "DeformationSmall": ("Deformation (small)", "変形度合い（小）", "変形度合い(小)"),
    "DeformationLarge": ("Deformation (large)", "変形度合い（大）", "変形度合い(大)"),
}

AUTO_MESH_CONFIRM_PATTERNS = (r"^OK$", r"^ＯＫ$")


class UIAutomationError(RuntimeError):
    """Raised when a named Cubism control cannot be verified safely."""


class MacroExecutionError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__(message)


@dataclass(frozen=True)
class UIAction:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args}


class UIBackend(Protocol):
    def perform(self, action: UIAction) -> None: ...


class RecordingBackend:
    """Test backend that records actions without touching the desktop."""

    def __init__(self) -> None:
        self.actions: list[UIAction] = []

    def perform(self, action: UIAction) -> None:
        self.actions.append(action)


def validate_psd_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".psd":
        raise ValueError(f"not a PSD file: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"PSD not found: {resolved}")
    return resolved


def _screenshot_action(path: Path) -> UIAction:
    return UIAction("capture_screenshot", {"path": str(path.resolve())})


def build_focus_actions() -> list[UIAction]:
    return [UIAction("focus_cubism")]


def build_screenshot_actions(path: Path) -> list[UIAction]:
    return [_screenshot_action(path)]


def build_import_psd_actions(
    psd_path: Path,
    *,
    open_mode: str = "create_new_model",
    screenshot: Path = Path("outputs/import_after.png"),
    dialog_timeout: float = 15.0,
    import_timeout: float = 90.0,
) -> list[UIAction]:
    resolved = validate_psd_path(psd_path)
    if open_mode not in OPEN_MODES:
        raise ValueError(f"unsupported open mode: {open_mode}")
    return [
        UIAction("focus_cubism"),
        UIAction("hotkey", {"keys": ["ctrl", "o"]}),
        UIAction("wait_for_dialog", {"kind": "open_file", "timeout": dialog_timeout}),
        UIAction("paste_file_path", {"path": str(resolved)}),
        UIAction("press_key", {"key": "enter"}),
        UIAction("wait_for_dialog", {"kind": "model_settings", "timeout": dialog_timeout}),
        UIAction("choose_model_open_mode", {"mode": open_mode}),
        UIAction(
            "wait_for_dialog_closed",
            {"kind": "model_settings", "timeout": import_timeout},
        ),
        UIAction("wait_after_operation", {"seconds": 1.0}),
        _screenshot_action(screenshot),
    ]


def build_auto_mesh_actions(
    *,
    preset: str = "Standard",
    alpha: int = 0,
    screenshot: Path = Path("outputs/auto_mesh_after.png"),
    dialog_timeout: float = 15.0,
) -> list[UIAction]:
    if preset not in PRESET_LABELS:
        raise ValueError(f"unsupported preset: {preset}")
    if not 0 <= alpha <= 255:
        raise ValueError("alpha must be between 0 and 255")
    return [
        UIAction("focus_cubism"),
        UIAction("hotkey", {"keys": ["ctrl", "a"]}),
        UIAction("hotkey", {"keys": ["ctrl", "shift", "a"]}),
        UIAction("wait_for_dialog", {"kind": "auto_mesh", "timeout": dialog_timeout}),
        UIAction("configure_auto_mesh", {"preset": preset, "alpha": alpha}),
        UIAction("wait_for_dialog_closed", {"kind": "auto_mesh", "timeout": dialog_timeout}),
        UIAction("wait_after_operation", {"seconds": 1.0}),
        _screenshot_action(screenshot),
    ]


def build_save_actions() -> list[UIAction]:
    return [
        UIAction("focus_cubism"),
        UIAction("hotkey", {"keys": ["ctrl", "s"]}),
        UIAction("wait_after_operation", {"seconds": 0.5}),
        UIAction("assert_dialog_absent", {"kind": "save_file", "seconds": 1.0}),
    ]


def build_undo_actions() -> list[UIAction]:
    return [
        UIAction("focus_cubism"),
        UIAction("hotkey", {"keys": ["ctrl", "z"]}),
        UIAction("wait_after_operation", {"seconds": 0.5}),
    ]


def build_named_macro(command: str, args: dict[str, Any]) -> list[UIAction]:
    if command == "cubism_ui.focus":
        return build_focus_actions()
    if command == "cubism_ui.screenshot":
        return build_screenshot_actions(Path(str(args["path"])))
    if command == "cubism_ui.import_psd":
        return build_import_psd_actions(
            Path(str(args["psd_path"])),
            open_mode=str(args.get("open_mode", "create_new_model")),
            screenshot=Path(str(args.get("screenshot", "outputs/import_after.png"))),
            dialog_timeout=float(args.get("dialog_timeout", 15.0)),
            import_timeout=float(args.get("import_timeout", 90.0)),
        )
    if command == "cubism_ui.apply_auto_mesh":
        return build_auto_mesh_actions(
            preset=str(args.get("preset", "Standard")),
            alpha=int(args.get("alpha", 0)),
            screenshot=Path(str(args.get("screenshot", "outputs/auto_mesh_after.png"))),
            dialog_timeout=float(args.get("dialog_timeout", 15.0)),
        )
    if command == "cubism_ui.save":
        return build_save_actions()
    if command == "cubism_ui.undo":
        return build_undo_actions()
    raise ValueError(f"unknown named UI macro: {command}")


def execute_actions(
    actions: Sequence[UIAction],
    *,
    execute: bool = False,
    backend: UIBackend | None = None,
    window_title: str = DEFAULT_WINDOW_TITLE,
    process_name: str = DEFAULT_PROCESS_NAME,
    process_id: int | None = None,
    profile_id: str | None = None,
    failure_screenshot: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "planned" if not execute else "running",
        "mode": "execute" if execute else "dry-run",
        "actions": [action.to_dict() for action in actions],
        "completed_actions": 0,
        "applied_mutations": [],
    }
    if not execute:
        return result

    active_backend = backend or WindowsCubismBackend(
        window_title=window_title,
        process_name=process_name,
        process_id=process_id,
        profile_id=profile_id,
    )
    try:
        for action in actions:
            active_backend.perform(action)
            result["completed_actions"] = int(result["completed_actions"]) + 1
            mutation = _mutation_for_action(action)
            if mutation:
                result["applied_mutations"].append(mutation)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failed_action"] = action.to_dict()
        if failure_screenshot is not None:
            try:
                active_backend.perform(_screenshot_action(failure_screenshot))
                result["failure_screenshot"] = str(failure_screenshot.resolve())
            except Exception as screenshot_exc:
                result["failure_screenshot_error"] = str(screenshot_exc)
        raise MacroExecutionError(str(exc), result) from exc

    result["status"] = "completed"
    mutations = set(result["applied_mutations"])
    if "auto_mesh" in mutations:
        result["verification_required"] = "visual_review"
    elif "import_psd" in mutations:
        result["verification_required"] = "api_document_verification"
    return result


def _mutation_for_action(action: UIAction) -> str | None:
    if action.name == "configure_auto_mesh":
        return "auto_mesh"
    if action.name == "choose_model_open_mode":
        return "import_psd"
    if action.name == "hotkey":
        keys = tuple(str(key).lower() for key in action.args.get("keys", []))
        if keys == ("ctrl", "s"):
            return "save"
        if keys == ("ctrl", "z"):
            return "undo"
    return None


class WindowsCubismBackend:
    """Windows-only backend using semantic UIA controls and keyboard shortcuts."""

    def __init__(
        self,
        *,
        window_title: str = DEFAULT_WINDOW_TITLE,
        process_name: str = DEFAULT_PROCESS_NAME,
        process_id: int | None = None,
        profile_id: str | None = None,
    ) -> None:
        if platform.system() != "Windows":
            raise UIAutomationError("real Cubism UI execution is supported only on Windows")
        try:
            import psutil  # type: ignore[import-untyped]
            import pyautogui  # type: ignore[import-untyped]
            from pywinauto import Desktop  # type: ignore[import-untyped]
        except ImportError as exc:
            raise UIAutomationError(
                'install Windows UI dependencies with: pip install -e ".[windows]"'
            ) from exc

        self._pyautogui = pyautogui
        self._psutil = psutil
        self._desktop = Desktop(backend="uia")
        self._window_title = re.compile(window_title, re.IGNORECASE)
        self._process_name = re.compile(process_name, re.IGNORECASE)
        self._fixed_process_allowlist = re.compile(DEFAULT_PROCESS_NAME, re.IGNORECASE)
        self._cubism_pid = process_id
        self._profile: CubismUIProfile | None = get_profile(profile_id) if profile_id else None
        self._active_dialogs: dict[str, Any] = {}
        self._pyautogui.PAUSE = 0.15

    @staticmethod
    def _control_text(control: Any) -> str:
        try:
            text = control.window_text()
        except Exception:
            text = ""
        if text:
            return str(text)
        try:
            return str(control.element_info.name or "")
        except Exception:
            return ""

    def _find_top_window(
        self,
        pattern: re.Pattern[str],
        timeout: float,
        *,
        process_id: int | None = None,
        require_process_match: bool = False,
    ) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for window in self._desktop.windows():
                try:
                    window_pid = int(window.element_info.process_id)
                except (AttributeError, TypeError, ValueError):
                    continue
                if process_id is not None and window_pid != process_id:
                    continue
                if require_process_match:
                    try:
                        executable = self._psutil.Process(window_pid).name()
                    except self._psutil.Error:
                        continue
                    if not self._fixed_process_allowlist.fullmatch(executable):
                        continue
                    if not self._process_name.fullmatch(executable):
                        continue
                if pattern.search(self._control_text(window)):
                    return window
            time.sleep(0.2)
        raise UIAutomationError(f"window not found: {pattern.pattern}")

    def _dialog_pattern(self, kind: str) -> re.Pattern[str]:
        patterns = (
            self._profile.dialog_patterns.get(kind)
            if self._profile is not None
            else DIALOG_PATTERNS.get(kind)
        )
        if patterns is None:
            raise UIAutomationError(f"unknown dialog kind: {kind}")
        return re.compile("|".join(f"(?:{item})" for item in patterns), re.IGNORECASE)

    def _find_dialog(self, kind: str, timeout: float = 1.0) -> Any:
        if self._cubism_pid is None:
            raise UIAutomationError("Cubism process has not been identified")
        dialog = self._find_top_window(
            self._dialog_pattern(kind),
            timeout,
            process_id=self._cubism_pid,
        )
        self._active_dialogs[kind] = dialog
        return dialog

    def _find_matching_control(
        self,
        dialog: Any,
        *,
        control_types: tuple[str, ...],
        patterns: tuple[str, ...],
    ) -> Any:
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        matches: list[Any] = []
        seen: set[int] = set()
        for control_type in control_types:
            for control in dialog.descendants(control_type=control_type):
                text = self._control_text(control)
                try:
                    enabled = bool(control.is_enabled())
                    visible = bool(control.is_visible())
                except Exception:
                    enabled = False
                    visible = False
                if (
                    enabled
                    and visible
                    and any(pattern.search(text) for pattern in compiled)
                    and id(control) not in seen
                ):
                    matches.append(control)
                    seen.add(id(control))
        if not matches:
            raise UIAutomationError(
                f"named control not found; expected patterns: {', '.join(patterns)}"
            )
        if len(matches) != 1:
            raise UIAutomationError(
                "ambiguous named control; expected exactly one match for patterns: "
                f"{', '.join(patterns)}"
            )
        return matches[0]

    def _ensure_cubism_foreground(self) -> None:
        if self._cubism_pid is None:
            raise UIAutomationError("Cubism process has not been identified")
        foreground_pids: list[int] = []
        for window in self._desktop.windows():
            try:
                if window.has_focus():
                    foreground_pids.append(int(window.element_info.process_id))
            except Exception:
                continue
        if foreground_pids != [self._cubism_pid]:
            raise UIAutomationError(
                "Cubism is not the unique foreground process; global shortcut was not sent"
            )
        try:
            executable = self._psutil.Process(self._cubism_pid).name()
        except self._psutil.Error as exc:
            raise UIAutomationError("Cubism foreground process is unavailable") from exc
        if not self._fixed_process_allowlist.fullmatch(executable):
            raise UIAutomationError(
                "foreground process executable is not allowlisted Cubism Editor"
            )

    @staticmethod
    def _invoke_control(control: Any) -> None:
        if hasattr(control, "select"):
            try:
                control.select()
                return
            except Exception:
                pass
        if hasattr(control, "invoke"):
            control.invoke()
            return
        raise UIAutomationError("control does not support select/invoke")

    def _choose_model_open_mode(self, mode: str) -> None:
        dialog = self._active_dialogs.get("model_settings") or self._find_dialog("model_settings")
        patterns = (
            self._profile.model_open_modes.get(mode)
            if self._profile is not None
            else MODEL_MODE_PATTERNS.get(mode)
        )
        if patterns is None:
            raise UIAutomationError(f"unsupported model open mode: {mode}")
        option = self._find_matching_control(
            dialog,
            control_types=("RadioButton", "Button"),
            patterns=patterns,
        )
        self._invoke_control(option)
        ok_button = self._find_matching_control(
            dialog,
            control_types=("Button",),
            patterns=(r"^OK$", r"^ＯＫ$"),
        )
        self._invoke_control(ok_button)

    def _configure_auto_mesh(self, preset: str, alpha: int) -> None:
        dialog = self._active_dialogs.get("auto_mesh") or self._find_dialog("auto_mesh")
        combo_boxes = dialog.descendants(control_type="ComboBox")
        if not combo_boxes:
            raise UIAutomationError(
                "auto-mesh preset combobox is not exposed through UIA"
            )
        if len(combo_boxes) != 1:
            raise UIAutomationError(
                "auto-mesh dialog exposes an ambiguous number of preset combobox controls"
            )

        labels = (
            self._profile.preset_labels.get(preset)
            if self._profile is not None
            else PRESET_LABELS.get(preset)
        )
        if labels is None:
            raise UIAutomationError(f"auto-mesh preset is not in the active profile: {preset}")
        combo = combo_boxes[0]
        selected = False
        for label in labels:
            try:
                combo.select(label)
                selected = True
                break
            except Exception:
                continue
        if not selected:
            raise UIAutomationError(f"auto-mesh preset not found: {preset}")

        alpha_control = self._find_matching_control(
            dialog,
            control_types=("Edit",),
            patterns=(
                self._profile.alpha_edit_patterns
                if self._profile is not None
                else (
                    r".*Alpha value to be considered transparent.*",
                    r".*透明とみなすアルファ値.*",
                )
            ),
        )
        if not hasattr(alpha_control, "set_edit_text"):
            raise UIAutomationError("alpha control does not support text input")
        alpha_control.set_edit_text(str(alpha))
        confirm = self._find_matching_control(
            dialog,
            control_types=("Button",),
            patterns=(
                self._profile.confirm_button_patterns
                if self._profile is not None
                else AUTO_MESH_CONFIRM_PATTERNS
            ),
        )
        self._invoke_control(confirm)

    def perform(self, action: UIAction) -> None:
        name = action.name
        args = action.args
        if name == "focus_cubism":
            window = self._find_top_window(
                self._window_title,
                10.0,
                process_id=self._cubism_pid,
                require_process_match=True,
            )
            self._cubism_pid = int(window.element_info.process_id)
            window.set_focus()
            return
        if name == "hotkey":
            keys = args.get("keys")
            allowed = {
                ("ctrl", "o"),
                ("ctrl", "a"),
                ("ctrl", "shift", "a"),
                ("ctrl", "s"),
                ("ctrl", "z"),
            }
            key_tuple = tuple(str(key).lower() for key in keys or [])
            if key_tuple not in allowed:
                raise UIAutomationError(f"hotkey is not allowlisted: {key_tuple}")
            self._ensure_cubism_foreground()
            self._pyautogui.hotkey(*key_tuple)
            return
        if name == "wait_for_dialog":
            dialog = self._find_dialog(str(args["kind"]), float(args["timeout"]))
            dialog.set_focus()
            return
        if name == "paste_file_path":
            path = Path(str(args["path"]))
            if not path.is_absolute() or not path.is_file():
                raise UIAutomationError(f"invalid absolute file path: {path}")
            dialog = self._active_dialogs.get("open_file") or self._find_dialog("open_file")
            file_name = self._find_matching_control(
                dialog,
                control_types=("Edit",),
                patterns=(r".*File name.*", r".*ファイル名.*"),
            )
            if hasattr(file_name, "set_edit_text"):
                file_name.set_edit_text(str(path))
            else:
                raise UIAutomationError("file-name control does not support text input")
            return
        if name == "press_key":
            key = str(args["key"]).lower()
            if key not in {"enter", "escape"}:
                raise UIAutomationError(f"key is not allowlisted: {key}")
            self._ensure_cubism_foreground()
            self._pyautogui.press(key)
            return
        if name == "choose_model_open_mode":
            self._choose_model_open_mode(str(args["mode"]))
            return
        if name == "wait_for_dialog_closed":
            kind = str(args["kind"])
            timeout = float(args["timeout"])
            pattern = self._dialog_pattern(kind)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                visible = any(
                    int(window.element_info.process_id) == self._cubism_pid
                    and pattern.search(self._control_text(window))
                    for window in self._desktop.windows()
                )
                if not visible:
                    return
                time.sleep(0.25)
            raise UIAutomationError(f"dialog did not close within {timeout}s: {kind}")
        if name == "assert_dialog_present":
            self._find_dialog(str(args["kind"]), float(args["timeout"]))
            return
        if name == "assert_dialog_absent":
            kind = str(args["kind"])
            seconds = float(args["seconds"])
            pattern = self._dialog_pattern(kind)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                visible = any(
                    int(window.element_info.process_id) == self._cubism_pid
                    and pattern.search(self._control_text(window))
                    for window in self._desktop.windows()
                )
                if visible:
                    raise UIAutomationError(f"unexpected dialog is open: {kind}")
                time.sleep(0.2)
            return
        if name == "configure_auto_mesh":
            self._configure_auto_mesh(str(args["preset"]), int(args["alpha"]))
            return
        if name == "wait_after_operation":
            time.sleep(float(args["seconds"]))
            return
        if name == "capture_screenshot":
            path = Path(str(args["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._cubism_pid is None:
                raise UIAutomationError("Cubism process has not been identified")
            window = self._find_top_window(
                self._window_title,
                2.0,
                process_id=self._cubism_pid,
                require_process_match=True,
            )
            window.capture_as_image().save(path)
            return
        raise UIAutomationError(f"unknown backend action: {name}")


def _add_execute_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform real UI operations; omitted means dry-run",
    )
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--profile")
    parser.add_argument(
        "--failure-screenshot",
        type=Path,
        default=Path("outputs/cubism_ui_failed.png"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-level Live2D Cubism UI macros")
    sub = parser.add_subparsers(dest="command", required=True)

    focus = sub.add_parser("focus")
    _add_execute_options(focus)

    screenshot = sub.add_parser("screenshot")
    screenshot.add_argument("path", type=Path)
    _add_execute_options(screenshot)

    import_psd = sub.add_parser("import-psd")
    import_psd.add_argument("psd_path", type=Path)
    import_psd.add_argument("--open-mode", choices=sorted(OPEN_MODES), default="create_new_model")
    import_psd.add_argument("--screenshot", type=Path, default=Path("outputs/import_after.png"))
    import_psd.add_argument("--dialog-timeout", type=float, default=15.0)
    import_psd.add_argument("--import-timeout", type=float, default=90.0)
    _add_execute_options(import_psd)

    auto_mesh = sub.add_parser("apply-auto-mesh")
    auto_mesh.add_argument("--preset", choices=sorted(PRESET_LABELS), default="Standard")
    auto_mesh.add_argument("--alpha", type=int, default=0)
    auto_mesh.add_argument(
        "--screenshot",
        type=Path,
        default=Path("outputs/auto_mesh_after.png"),
    )
    auto_mesh.add_argument("--dialog-timeout", type=float, default=15.0)
    _add_execute_options(auto_mesh)

    save = sub.add_parser("save")
    _add_execute_options(save)

    undo = sub.add_parser("undo")
    _add_execute_options(undo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "focus":
            actions = build_focus_actions()
        elif args.command == "screenshot":
            actions = build_screenshot_actions(args.path)
        elif args.command == "import-psd":
            actions = build_import_psd_actions(
                args.psd_path,
                open_mode=args.open_mode,
                screenshot=args.screenshot,
                dialog_timeout=args.dialog_timeout,
                import_timeout=args.import_timeout,
            )
        elif args.command == "apply-auto-mesh":
            actions = build_auto_mesh_actions(
                preset=args.preset,
                alpha=args.alpha,
                screenshot=args.screenshot,
                dialog_timeout=args.dialog_timeout,
            )
        elif args.command == "save":
            actions = build_save_actions()
        elif args.command == "undo":
            actions = build_undo_actions()
        else:
            parser.error(f"unsupported command: {args.command}")
            return 2

        result = execute_actions(
            actions,
            execute=args.execute,
            window_title=args.window_title,
            process_name=args.process_name,
            process_id=args.process_id,
            profile_id=args.profile,
            failure_screenshot=args.failure_screenshot,
        )
    except (FileNotFoundError, ValueError, UIAutomationError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    except MacroExecutionError as exc:
        print(json.dumps(exc.result, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
