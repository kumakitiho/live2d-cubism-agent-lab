from __future__ import annotations

from pathlib import Path

import pytest

from tools.cubism_ui import (
    DEFAULT_PROCESS_NAME,
    RecordingBackend,
    build_auto_mesh_actions,
    build_import_psd_actions,
    build_save_actions,
    execute_actions,
    validate_psd_path,
)


def test_import_psd_dry_run_returns_expected_actions(tmp_path: Path) -> None:
    psd = tmp_path / "model.psd"
    psd.write_bytes(b"8BPS")
    actions = build_import_psd_actions(psd)
    assert [action.name for action in actions] == [
        "focus_cubism",
        "hotkey",
        "wait_for_dialog",
        "paste_file_path",
        "press_key",
        "wait_for_dialog",
        "choose_model_open_mode",
        "wait_for_dialog_closed",
        "wait_after_operation",
        "capture_screenshot",
    ]
    result = execute_actions(actions)
    assert result["status"] == "planned"
    assert result["completed_actions"] == 0


def test_recording_backend_executes_without_desktop() -> None:
    backend = RecordingBackend()
    actions = build_save_actions()
    result = execute_actions(actions, execute=True, backend=backend)
    assert result["status"] == "completed"
    assert result["applied_mutations"] == ["save"]
    assert backend.actions == actions


def test_auto_mesh_validates_alpha() -> None:
    with pytest.raises(ValueError, match="between 0 and 255"):
        build_auto_mesh_actions(alpha=256)


def test_default_process_pattern_does_not_match_browser() -> None:
    import re

    pattern = re.compile(DEFAULT_PROCESS_NAME, re.IGNORECASE)
    assert pattern.search("CubismEditor5.exe")
    assert not pattern.search("chrome.exe")
    assert not pattern.search("CubismViewer5.exe")
    assert not pattern.search("CubismUpdater.exe")


def test_psd_path_validation_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_psd_path(tmp_path / "missing.psd")


def test_psd_path_validation_rejects_wrong_extension(tmp_path: Path) -> None:
    png = tmp_path / "model.png"
    png.write_bytes(b"png")
    with pytest.raises(ValueError, match="not a PSD"):
        validate_psd_path(png)
