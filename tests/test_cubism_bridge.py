from __future__ import annotations

from pathlib import Path

from tools.cubism_bridge import (
    _can_run_undo_recovery,
    _verify_imported_document,
    run_action_plan_data,
)


def test_bridge_writes_dry_run_report(tmp_path: Path) -> None:
    psd = tmp_path / "model.psd"
    psd.write_bytes(b"8BPS")
    report = tmp_path / "report.md"
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "import",
                "mode": "ui_macro",
                "command": "cubism_ui.import_psd",
                "args": {"psd_path": str(psd)},
            },
            {
                "id": "review",
                "mode": "manual_checkpoint",
                "instruction": "見た目を確認する",
            },
        ],
    }
    result = run_action_plan_data(plan, report_path=report)
    assert result["status"] == "planned"
    assert report.exists()
    assert "Cubism Action Plan Report" in report.read_text(encoding="utf-8")


def test_bridge_stops_at_manual_checkpoint_in_execute_mode(tmp_path: Path) -> None:
    report = tmp_path / "manual.md"
    plan = {
        "schema_version": 1,
        "project": "sample",
        "steps": [
            {
                "id": "review",
                "mode": "manual_checkpoint",
                "instruction": "見た目を確認する",
            }
        ],
    }
    result = run_action_plan_data(plan, execute=True, report_path=report)
    assert result["status"] == "stopped_for_manual_checkpoint"


def test_undo_recovery_requires_confirmed_auto_mesh_mutation() -> None:
    assert not _can_run_undo_recovery("cubism_ui.undo", {"applied_mutations": []})
    assert _can_run_undo_recovery(
        "cubism_ui.undo",
        {"applied_mutations": ["auto_mesh"]},
    )


def test_import_verification_checks_document_count_and_current_model() -> None:
    steps = [
        {
            "id": "before",
            "result": {
                "response": {
                    "Data": {
                        "ModelingDocuments": [{"DocumentUID": "doc-1", "Views": []}]
                    }
                }
            },
        },
        {
            "id": "import",
            "result": {"applied_mutations": ["import_psd"]},
        },
        {
            "id": "after",
            "result": {
                "response": {
                    "Documents": {
                        "ModelingDocuments": [
                            {"DocumentUID": "doc-1", "Views": []},
                            {
                                "DocumentUID": "doc-2",
                                "Views": [{"ModelUID": "model-2"}],
                            },
                        ]
                    },
                    "CurrentModel": {"ModelUID": "model-2"},
                }
            },
        },
    ]
    result = _verify_imported_document(
        {"before_step": "before", "import_step": "import", "after_step": "after"},
        steps,
        execute=True,
    )
    assert result["status"] == "completed"
    assert result["new_document_uid"] == "doc-2"


def test_import_verification_rejects_current_model_from_old_document() -> None:
    steps = [
        {
            "id": "before",
            "result": {
                "response": {
                    "Data": {
                        "ModelingDocuments": [
                            {
                                "DocumentUID": "doc-1",
                                "Views": [{"ModelUID": "model-old"}],
                            }
                        ]
                    }
                }
            },
        },
        {"id": "import", "result": {"applied_mutations": ["import_psd"]}},
        {
            "id": "after",
            "result": {
                "response": {
                    "Documents": {
                        "ModelingDocuments": [
                            {
                                "DocumentUID": "doc-1",
                                "Views": [{"ModelUID": "model-old"}],
                            },
                            {"DocumentUID": "doc-2", "Views": []},
                        ]
                    },
                    "CurrentModel": {"ModelUID": "model-old"},
                }
            },
        },
    ]
    result = _verify_imported_document(
        {"before_step": "before", "import_step": "import", "after_step": "after"},
        steps,
        execute=True,
    )
    assert result["status"] == "error"
