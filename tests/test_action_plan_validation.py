from __future__ import annotations

from pathlib import Path

from tools.action_plan import load_action_plan, validate_action_plan


def test_sample_action_plan_is_valid() -> None:
    plan = load_action_plan(Path("examples/action_plan.sample.yaml"))
    assert validate_action_plan(plan) == []


def test_invalid_mode_is_reported() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [{"id": "bad", "mode": "mouse", "command": "click"}],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("must be one of" in message for message in messages)


def test_forbidden_click_command_is_reported() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "bad",
                "mode": "ui_macro",
                "command": "cubism_ui.click",
                "args": {},
            }
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("forbidden UI primitive" in message for message in messages)
    assert any("not allowed" in message for message in messages)


def test_screen_coordinates_are_reported() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "bad",
                "mode": "ui_macro",
                "command": "cubism_ui.focus",
                "args": {"coordinates": [100, 200]},
            }
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("coordinates are forbidden" in message for message in messages)


def test_manual_checkpoint_requires_instruction() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [{"id": "review", "mode": "manual_checkpoint"}],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("instruction" in message for message in messages)


def test_screenshot_requires_path() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "shot",
                "mode": "ui_macro",
                "command": "cubism_ui.screenshot",
                "args": {},
            }
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("args.path: is required" in message for message in messages)


def test_screenshot_must_stay_in_outputs(tmp_path: Path) -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "shot",
                "mode": "ui_macro",
                "command": "cubism_ui.screenshot",
                "args": {"path": str(tmp_path / "review.png")},
            }
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("must stay within outputs" in message for message in messages)


def test_unknown_command_argument_is_rejected() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "save",
                "mode": "ui_macro",
                "command": "cubism_ui.save",
                "args": {"force": True},
            }
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("args.force: is not allowed" in message for message in messages)


def test_auto_mesh_cannot_follow_unverified_import() -> None:
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "import",
                "mode": "ui_macro",
                "command": "cubism_ui.import_psd",
                "args": {"psd_path": "assets/models/model.psd"},
            },
            {
                "id": "mesh",
                "mode": "ui_macro",
                "command": "cubism_ui.apply_auto_mesh",
            },
        ],
    }
    messages = [issue.format() for issue in validate_action_plan(plan)]
    assert any("auto-mesh after import requires" in message for message in messages)
