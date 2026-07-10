from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from tools.cubism_api import ConnectionOptions
from tools.cubism_control_tree_snapshot import (
    CubismIdentity,
    DiagnosticFailure,
    SnapshotNode,
    WindowCandidate,
    WindowsControlTreeBackend,
    identify_cubism_window,
    sanitize_control_text,
    serialize_control_tree,
)
from tools.cubism_environment_probe import run_environment_probe


def _root(*labels: str, process_id: int = 42) -> SnapshotNode:
    return SnapshotNode(
        name="Live2D Cubism Editor 5",
        automation_id="main",
        control_type="Window",
        class_name="QtWindow",
        enabled=True,
        visible=True,
        process_id=process_id,
        children=[
            SnapshotNode(
                name=label,
                automation_id=f"item-{index}",
                control_type="MenuItem",
                class_name="",
                enabled=True,
                visible=True,
                process_id=process_id,
                supported_patterns=["Invoke"],
            )
            for index, label in enumerate(labels)
        ],
    )


class FakeBackend:
    def __init__(self, roots: list[SnapshotNode] | None = None, *, fail_tree: bool = False) -> None:
        self.calls: list[str] = []
        self.roots = roots or [_root("File", "Edit", "Save")]
        self.fail_tree = fail_tree

    def discover(self) -> CubismIdentity:
        self.calls.append("discover")
        return CubismIdentity(
            process_id=42,
            executable_name="CubismEditor5.exe",
            executable=r"C:\Program Files\Live2D\CubismEditor5.exe",
            title="Live2D Cubism Editor 5",
            foreground=True,
            rect=(0, 0, 100, 100),
        )

    def collect_roots(self, identity: CubismIdentity) -> list[SnapshotNode]:
        self.calls.append("collect_roots")
        if self.fail_tree:
            raise DiagnosticFailure("control_not_exposed_by_uia", "tree unavailable")
        return self.roots

    def capture_screenshot(self, path: Path) -> None:
        self.calls.append("capture_screenshot")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")


def _api(_: Any) -> dict[str, Any]:
    return {
        "reachable": True,
        "endpoint": "ws://127.0.0.1:22033",
        "approved": True,
        "current_edit_mode": "Modeling",
        "current_document_uid": "doc-1",
        "current_model_uid": "model-1",
        "error": None,
    }


def test_dry_run_has_no_external_calls_or_file_writes(tmp_path: Path) -> None:
    backend = FakeBackend()
    output = tmp_path / "report.yaml"
    result = run_environment_probe(output=output, execute=False, backend=backend)
    assert result["status"] == "planned"
    assert result["external_connections"] is False
    assert result["files_written"] is False
    assert backend.calls == []
    assert not output.exists()


def test_process_name_and_title_must_both_match() -> None:
    fake = WindowCandidate(7, "chrome.exe", None, "Live2D Cubism Editor 5")
    with pytest.raises(DiagnosticFailure, match="executable name") as error:
        identify_cubism_window([fake])
    assert error.value.code == "wrong_process"


def test_process_override_cannot_widen_fixed_cubism_allowlist() -> None:
    fake = WindowCandidate(7, "chrome.exe", None, "Live2D Cubism Editor 5")
    with pytest.raises(DiagnosticFailure) as error:
        identify_cubism_window(
            [fake], process_name=r"^chrome\.exe$", window_title=r".*"
        )
    assert error.value.code == "wrong_process"


def test_matching_editor_is_identified_by_process_id() -> None:
    editor = WindowCandidate(42, "CubismEditor5.exe", "editor.exe", "Cubism Editor 5")
    identity = identify_cubism_window([editor])
    assert identity.process_id == 42
    assert identity.executable_name == "CubismEditor5.exe"


def test_control_tree_excludes_other_process_and_redacts_sensitive_text() -> None:
    root = _root("File", process_id=42)
    root.children.extend(
        [
            SnapshotNode(
                name=r"C:\Users\alice\private.psd",
                automation_id="path",
                control_type="Text",
                class_name="",
                enabled=True,
                visible=True,
                process_id=42,
            ),
            SnapshotNode(
                name="secret-value",
                automation_id="password",
                control_type="Edit",
                class_name="",
                enabled=True,
                visible=True,
                process_id=42,
            ),
            SnapshotNode(
                name="Foreign dialog",
                automation_id="foreign",
                control_type="Window",
                class_name="",
                enabled=True,
                visible=True,
                process_id=99,
            ),
        ]
    )
    identity = CubismIdentity(42, "CubismEditor5.exe", None, "Cubism Editor", True, None)
    payload = serialize_control_tree([root], identity)
    encoded = json.dumps(payload)
    assert "alice" not in encoded
    assert "private.psd" not in encoded
    assert "secret-value" not in encoded
    assert "Foreign dialog" not in encoded
    assert "<redacted-input>" in encoded


def test_secret_assignment_is_redacted() -> None:
    assert sanitize_control_text("token=abc123") == "token=<redacted>"
    assert sanitize_control_text("Authorization: Bearer abc.def.ghi") == "Bearer <redacted>"


def test_execute_writes_schema_valid_report_and_sanitized_tree(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "report.yaml"
    tree = tmp_path / "outputs" / "tree.json"
    result = run_environment_probe(
        output=output,
        control_tree_output=tree,
        base_dir=tmp_path,
        execute=True,
        backend=FakeBackend(),
        api_probe=_api,
        timestamp="2026-01-01T00:00:00+00:00",
        platform_report={"platform": "win32", "windows_version": "Windows-11"},
        screen_report={"width": 100, "height": 100, "dpi_scale": 1.0, "monitors": []},
    )
    assert result["diagnosis"]["status"] == "ready"
    assert result["uia"]["profile_match"] == "cubism-5-en"
    assert output.exists() and tree.exists()
    schema = yaml.safe_load(
        Path("schemas/cubism_environment_report.schema.yaml").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(yaml.safe_load(output.read_text(encoding="utf-8")))


def test_tree_failure_still_writes_diagnostic_report_and_screenshot(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "report.yaml"
    screenshot = tmp_path / "outputs" / "failure.png"
    result = run_environment_probe(
        output=output,
        screenshot=screenshot,
        base_dir=tmp_path,
        execute=True,
        backend=FakeBackend(fail_tree=True),
        api_probe=_api,
        platform_report={"platform": "win32", "windows_version": "Windows-11"},
        screen_report={"width": 100, "height": 100, "dpi_scale": 1.0, "monitors": []},
    )
    assert result["diagnosis"]["status"] == "blocked"
    assert "control_not_exposed_by_uia" in result["diagnosis"]["blockers"]
    assert output.exists() and screenshot.exists()


def test_process_scoped_screenshot_allows_main_title_change(tmp_path: Path) -> None:
    class Rectangle:
        left = 0
        top = 0
        right = 100
        bottom = 100

    class Image:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

    class Window:
        element_info = type("Info", (), {"process_id": 42, "name": ""})()

        def window_text(self) -> str:
            return "Live2D Cubism Editor 5 - imported-model"

        def rectangle(self) -> Rectangle:
            return Rectangle()

        def capture_as_image(self) -> Image:
            return Image()

    class Desktop:
        def windows(self) -> list[Window]:
            return [Window()]

    class Process:
        def name(self) -> str:
            return "CubismEditor5.exe"

    class Psutil:
        @staticmethod
        def Process(process_id: int) -> Process:
            assert process_id == 42
            return Process()

    backend = object.__new__(WindowsControlTreeBackend)
    backend._identity = CubismIdentity(
        42,
        "CubismEditor5.exe",
        None,
        "Live2D Cubism Editor 5",
        True,
        None,
    )
    backend._window_title = r".*(?:Live2D Cubism|Cubism Editor|Cubism).*"
    backend._desktop = Desktop()
    backend._psutil = Psutil()
    screenshot = tmp_path / "cubism.png"
    backend.capture_screenshot(screenshot)
    assert screenshot.read_bytes() == b"png"


def test_output_outside_base_dir_is_rejected_before_probe(tmp_path: Path) -> None:
    backend = FakeBackend()
    with pytest.raises(ValueError, match="inside base-dir"):
        run_environment_probe(
            output=tmp_path.parent / "outside.yaml",
            base_dir=tmp_path,
            execute=True,
            backend=backend,
        )
    assert backend.calls == []


def test_non_loopback_api_host_is_rejected_before_probe(tmp_path: Path) -> None:
    backend = FakeBackend()
    with pytest.raises(ValueError, match="loopback"):
        run_environment_probe(
            output=tmp_path / "report.yaml",
            base_dir=tmp_path,
            execute=True,
            backend=backend,
            api_options=ConnectionOptions(host="example.com"),
        )
    assert backend.calls == []


def test_environment_schema_and_sample_validate() -> None:
    schema = yaml.safe_load(
        Path("schemas/cubism_environment_report.schema.yaml").read_text(encoding="utf-8")
    )
    sample = yaml.safe_load(
        Path("examples/cubism_environment_report.sample.yaml").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(sample)
