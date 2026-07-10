from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from tools.cubism_api import ConnectionOptions
from tools.cubism_control_tree_snapshot import CubismIdentity, SnapshotNode
from tools.cubism_smoke_test import run_smoke_test, verify_import_snapshots
from tools.cubism_ui import (
    MacroExecutionError,
    RecordingBackend,
    UIAction,
    UIAutomationError,
    WindowsCubismBackend,
    execute_actions,
)


def _tree(*labels: str) -> list[SnapshotNode]:
    return [
        SnapshotNode(
            name="Live2D Cubism Editor 5",
            automation_id="main",
            control_type="Window",
            class_name="QtWindow",
            enabled=True,
            visible=True,
            process_id=42,
            children=[
                SnapshotNode(
                    name=label,
                    automation_id=f"control-{index}",
                    control_type="MenuItem",
                    class_name="",
                    enabled=True,
                    visible=True,
                    process_id=42,
                    supported_patterns=["Invoke"],
                )
                for index, label in enumerate(labels)
            ],
        )
    ]


class FakeControlBackend:
    def __init__(
        self,
        labels: tuple[str, ...] = ("File", "Edit", "Save"),
        *,
        title: str = "Live2D Cubism Editor 5",
        executable_name: str = "CubismEditor5.exe",
    ) -> None:
        self.labels = labels
        self.title = title
        self.executable_name = executable_name
        self.screenshots: list[Path] = []

    def discover(self) -> CubismIdentity:
        return CubismIdentity(
            42,
            self.executable_name,
            r"C:\Program Files\Live2D\CubismEditor5.exe",
            self.title,
            True,
            (0, 0, 100, 100),
        )

    def collect_roots(self, identity: CubismIdentity) -> list[SnapshotNode]:
        return _tree(*self.labels)

    def capture_screenshot(self, path: Path) -> None:
        self.screenshots.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")


class EvidenceRecordingBackend(RecordingBackend):
    def perform(self, action: UIAction) -> None:
        super().perform(action)
        if action.name == "capture_screenshot":
            path = Path(str(action.args["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")


def _before() -> dict[str, Any]:
    return {
        "documents": [
            {
                "DocumentUID": "doc-1",
                "Views": [{"ModelUID": "model-1"}],
                "FilePath": r"C:\Users\alice\private.cmo3",
            }
        ],
        "current_model_uid": "model-1",
        "current_edit_mode": "Modeling",
    }


def _after() -> dict[str, Any]:
    return {
        "documents": [
            {
                "DocumentUID": "doc-1",
                "Views": [{"ModelUID": "model-1"}],
                "FilePath": r"C:\Users\alice\private.cmo3",
            },
            {
                "DocumentUID": "doc-2",
                "Views": [{"ModelUID": "model-2"}],
                "FilePath": r"C:\Users\alice\imported.cmo3",
            },
        ],
        "current_model_uid": "model-2",
        "current_edit_mode": "Modeling",
    }


class FakeAPI:
    def __init__(self, snapshots: list[dict[str, Any]], *, approved: bool = True) -> None:
        self.snapshots = snapshots
        self.approved = approved
        self.calls: list[str] = []

    def approval(self) -> bool:
        self.calls.append("approval")
        return self.approved

    def snapshot(self) -> dict[str, Any]:
        self.calls.append("snapshot")
        if not self.snapshots:
            raise RuntimeError("snapshot unavailable")
        return self.snapshots.pop(0)


def _psd(tmp_path: Path) -> Path:
    path = tmp_path / "model.psd"
    path.write_bytes(b"8BPS")
    return path


def test_smoke_dry_run_has_no_connections_or_writes(tmp_path: Path) -> None:
    api = FakeAPI([])
    output = tmp_path / "outputs" / "smoke.yaml"
    result = run_smoke_test(
        tmp_path / "missing.psd", output=output, execute=False, api=api
    )
    assert result["status"] == "planned"
    assert all(stage["status"] == "planned" for stage in result["stages"].values())
    assert api.calls == []
    assert not output.exists()


def test_import_snapshot_diff_detects_new_document_and_model() -> None:
    result = verify_import_snapshots(_before(), _after())
    assert result["new_document_uid"] == "doc-2"
    assert result["current_model_uid"] == "model-2"


def test_import_snapshot_rejects_wrong_edit_mode() -> None:
    after = _after()
    after["current_edit_mode"] = "FormAnimation"
    with pytest.raises(Exception) as error:
        verify_import_snapshots(_before(), after)
    assert getattr(error.value, "code", None) == "wrong_edit_mode"


def test_import_verification_unavailable_is_not_success(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "smoke.yaml"
    api = FakeAPI([_before()])
    result = run_smoke_test(
        _psd(tmp_path),
        output=output,
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(),
        api=api,
        ui_backend=EvidenceRecordingBackend(),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert result["status"] == "waiting_for_user"
    assert result["stages"]["import_verification"]["status"] == "waiting_for_user"
    assert result["stages"]["auto_mesh"]["status"] == "blocked"


def test_complete_smoke_sequence_uses_named_actions_and_undo(tmp_path: Path) -> None:
    backend = EvidenceRecordingBackend()
    output = tmp_path / "outputs" / "smoke.yaml"
    result = run_smoke_test(
        _psd(tmp_path),
        output=output,
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(),
        api=FakeAPI([_before(), _after(), _after()]),
        ui_backend=backend,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert result["status"] == "completed"
    names = [action.name for action in backend.actions]
    assert names.index("choose_model_open_mode") < names.index("configure_auto_mesh")
    assert names.index("configure_auto_mesh") < max(
        index
        for index, action in enumerate(backend.actions)
        if action.name == "hotkey" and action.args.get("keys") == ["ctrl", "z"]
    )
    assert result["stages"]["undo"]["details"]["applied_mutations"] == ["undo"]
    assert result["visual_review_required"] is True
    schema = yaml.safe_load(
        Path("schemas/cubism_smoke_test_report.schema.yaml").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(yaml.safe_load(output.read_text(encoding="utf-8")))
    report_text = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in report_text
    assert "FilePath" not in report_text


def test_profile_mismatch_stops_before_api_or_ui(tmp_path: Path) -> None:
    api = FakeAPI([_before()])
    ui = EvidenceRecordingBackend()
    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(("Archivo", "Editar")),
        api=api,
        ui_backend=ui,
    )
    assert result["stages"]["preflight"]["failure"] == "unsupported_language"
    assert result["stages"]["api_connection"]["status"] == "blocked"
    assert api.calls == []
    assert ui.actions == []


def test_unsupported_version_stops_before_api_or_ui(tmp_path: Path) -> None:
    api = FakeAPI([_before()])
    ui = EvidenceRecordingBackend()
    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(
            title="Live2D Cubism Editor 6", executable_name="CubismEditor6.exe"
        ),
        api=api,
        ui_backend=ui,
    )
    assert result["stages"]["preflight"]["failure"] == "unsupported_version"
    assert api.calls == []
    assert ui.actions == []


def test_failure_screenshot_and_blocked_propagation(tmp_path: Path) -> None:
    control = FakeControlBackend()
    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=control,
        api=FakeAPI([_before()]),
        ui_backend=EvidenceRecordingBackend(),
    )
    assert result["evidence"]["failure_screenshot"] == "outputs/cubism-smoke-failure.png"
    assert control.screenshots
    assert result["stages"]["final_snapshot"]["status"] == "blocked"


def test_output_outside_base_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside base-dir"):
        run_smoke_test(
            _psd(tmp_path),
            output=tmp_path.parent / "outside.yaml",
            base_dir=tmp_path,
            execute=True,
        )


def test_non_loopback_api_host_is_rejected_before_report_write(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "smoke.yaml"
    with pytest.raises(ValueError, match="loopback"):
        run_smoke_test(
            _psd(tmp_path),
            output=output,
            base_dir=tmp_path,
            execute=True,
            api_options=ConnectionOptions(host="192.0.2.10"),
        )
    assert not output.exists()


class FakeDialog:
    def __init__(
        self,
        combos: list[object],
        edits: list[object] | None = None,
        buttons: list[object] | None = None,
    ) -> None:
        self.combos = combos
        self.edits = edits or []
        self.buttons = buttons or []

    def descendants(self, *, control_type: str) -> list[object]:
        if control_type == "ComboBox":
            return self.combos
        if control_type == "Edit":
            return self.edits
        if control_type == "Button":
            return self.buttons
        return []


class FakeCombo:
    def select(self, label: str) -> None:
        return None


class FakeNamedControl:
    def __init__(self, name: str) -> None:
        self.name = name

    def window_text(self) -> str:
        return self.name

    def is_enabled(self) -> bool:
        return True

    def is_visible(self) -> bool:
        return True

    def set_edit_text(self, value: str) -> None:
        return None

    def invoke(self) -> None:
        return None


def _bare_windows_backend(dialog: FakeDialog) -> WindowsCubismBackend:
    backend = object.__new__(WindowsCubismBackend)
    backend._active_dialogs = {"auto_mesh": dialog}
    backend._profile = None
    return backend


def test_auto_mesh_stops_when_combobox_is_missing() -> None:
    with pytest.raises(UIAutomationError, match="not exposed"):
        _bare_windows_backend(FakeDialog([]))._configure_auto_mesh("Standard", 10)


def test_auto_mesh_stops_when_combobox_is_ambiguous() -> None:
    with pytest.raises(UIAutomationError, match="ambiguous"):
        _bare_windows_backend(FakeDialog([FakeCombo(), FakeCombo()]))._configure_auto_mesh(
            "Standard", 10
        )


def test_auto_mesh_stops_when_alpha_edit_is_missing() -> None:
    backend = _bare_windows_backend(FakeDialog([FakeCombo()]))
    with pytest.raises(UIAutomationError, match="named control not found"):
        backend._configure_auto_mesh("Standard", 10)


def test_auto_mesh_stops_when_alpha_edit_is_ambiguous() -> None:
    alpha = "Alpha value to be considered transparent"
    dialog = FakeDialog(
        [FakeCombo()],
        edits=[FakeNamedControl(alpha), FakeNamedControl(alpha)],
        buttons=[FakeNamedControl("OK")],
    )
    with pytest.raises(UIAutomationError, match="ambiguous named control"):
        _bare_windows_backend(dialog)._configure_auto_mesh("Standard", 10)


def test_auto_mesh_stops_when_confirm_button_is_ambiguous() -> None:
    dialog = FakeDialog(
        [FakeCombo()],
        edits=[FakeNamedControl("Alpha value to be considered transparent")],
        buttons=[FakeNamedControl("OK"), FakeNamedControl("OK")],
    )
    with pytest.raises(UIAutomationError, match="ambiguous named control"):
        _bare_windows_backend(dialog)._configure_auto_mesh("Standard", 10)


def test_unverifiable_control_state_is_not_an_operation_candidate() -> None:
    class UnverifiableControl(FakeNamedControl):
        def is_enabled(self) -> bool:
            raise RuntimeError("UIA state unavailable")

    dialog = FakeDialog([], edits=[UnverifiableControl("target")])
    backend = _bare_windows_backend(dialog)
    with pytest.raises(UIAutomationError, match="named control not found"):
        backend._find_matching_control(
            dialog, control_types=("Edit",), patterns=(r"^target$",)
        )


def test_dialog_lookup_is_scoped_to_verified_process_id() -> None:
    backend = object.__new__(WindowsCubismBackend)
    backend._cubism_pid = 42
    backend._profile = None
    backend._active_dialogs = {}
    captured: dict[str, Any] = {}

    def find_top_window(pattern: Any, timeout: float, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    backend._find_top_window = find_top_window
    backend._find_dialog("auto_mesh")
    assert captured["process_id"] == 42


class FailAutoMeshScreenshotBackend(EvidenceRecordingBackend):
    def perform(self, action: UIAction) -> None:
        if action.name == "capture_screenshot" and "auto-mesh" in str(action.args.get("path")):
            self.actions.append(action)
            raise RuntimeError("auto-mesh screenshot failed")
        super().perform(action)


def test_auto_mesh_post_apply_failure_runs_one_undo_recovery(tmp_path: Path) -> None:
    backend = FailAutoMeshScreenshotBackend()
    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(),
        api=FakeAPI([_before(), _after()]),
        ui_backend=backend,
    )
    undo_hotkeys = [
        action
        for action in backend.actions
        if action.name == "hotkey" and action.args.get("keys") == ["ctrl", "z"]
    ]
    assert len(undo_hotkeys) == 1
    assert result["stages"]["auto_mesh"]["recovery"]["status"] == "completed"
    assert result["stages"]["undo"]["status"] == "blocked"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ambiguous named control", "ambiguous_control"),
        ("named control not found", "control_not_exposed_by_uia"),
    ],
)
def test_smoke_classifies_auto_mesh_control_failures(
    tmp_path: Path, message: str, expected: str
) -> None:
    class FailingControlBackend(EvidenceRecordingBackend):
        def perform(self, action: UIAction) -> None:
            if action.name == "configure_auto_mesh":
                raise UIAutomationError(message)
            super().perform(action)

    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(),
        api=FakeAPI([_before(), _after()]),
        ui_backend=FailingControlBackend(),
    )
    assert result["stages"]["auto_mesh"]["failure"] == expected


def test_macro_error_classifies_failed_hotkey_for_report() -> None:
    class FailingHotkeyBackend(RecordingBackend):
        def perform(self, action: UIAction) -> None:
            if action.name == "hotkey":
                raise UIAutomationError("foreground changed")
            super().perform(action)

    with pytest.raises(MacroExecutionError) as error:
        execute_actions(
            [UIAction("hotkey", {"keys": ["ctrl", "o"]})],
            execute=True,
            backend=FailingHotkeyBackend(),
        )
    assert error.value.result["failed_action"]["name"] == "hotkey"


def test_smoke_classifies_failed_hotkey_as_shortcut_failure(tmp_path: Path) -> None:
    class FailingHotkeyBackend(EvidenceRecordingBackend):
        def perform(self, action: UIAction) -> None:
            if action.name == "hotkey":
                raise UIAutomationError("foreground changed")
            super().perform(action)

    result = run_smoke_test(
        _psd(tmp_path),
        output=tmp_path / "outputs" / "smoke.yaml",
        base_dir=tmp_path,
        execute=True,
        control_backend=FakeControlBackend(),
        api=FakeAPI([_before()]),
        ui_backend=FailingHotkeyBackend(),
    )
    assert result["stages"]["psd_import"]["failure"] == "shortcut_failed"


def test_global_hotkey_stops_when_foreground_pid_changed() -> None:
    class FakeWindow:
        def __init__(self, process_id: int, focused: bool) -> None:
            self.element_info = type("Info", (), {"process_id": process_id})()
            self.focused = focused

        def has_focus(self) -> bool:
            return self.focused

    class FakeDesktop:
        def windows(self) -> list[FakeWindow]:
            return [FakeWindow(99, True), FakeWindow(42, False)]

    class FakePyAutoGUI:
        def __init__(self) -> None:
            self.sent = False

        def hotkey(self, *keys: str) -> None:
            self.sent = True

    backend = object.__new__(WindowsCubismBackend)
    backend._cubism_pid = 42
    backend._desktop = FakeDesktop()
    backend._fixed_process_allowlist = re.compile(r"^CubismEditor.*\.exe$", re.IGNORECASE)
    backend._pyautogui = FakePyAutoGUI()
    with pytest.raises(UIAutomationError, match="foreground"):
        backend.perform(UIAction("hotkey", {"keys": ["ctrl", "o"]}))
    assert backend._pyautogui.sent is False


def test_smoke_schema_is_valid() -> None:
    schema = yaml.safe_load(
        Path("schemas/cubism_smoke_test_report.schema.yaml").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
