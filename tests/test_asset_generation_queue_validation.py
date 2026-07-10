from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_generation_queue_validator import main, validate_asset_generation_queue


def _sample() -> dict[str, object]:
    return load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))


def _approve_all_jobs_and_gates(data: dict[str, object]) -> None:
    jobs = data["jobs"]
    merge_gate = data["merge_gate"]
    assert isinstance(jobs, list)
    assert isinstance(merge_gate, dict)
    for job in jobs:
        job["status"] = "approved"
        for key in job["validation"]:
            job["validation"][key] = True
    for key in merge_gate["validation"]:
        merge_gate["validation"][key] = True


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_cli_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    layer_map = load_yaml_mapping(Path("examples/layer_map.sample.yaml"))
    feedback = load_yaml_mapping(Path("examples/asset_feedback.sample.yaml"))
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    manifest = load_yaml_mapping(Path("examples/asset_manifest.sample.yaml"))

    feedback["model_refs"]["layer_map"] = "layer_map.yaml"
    feedback["feedback"][0]["status"] = "resolved"
    queue["feedback_inputs"] = ["feedback.yaml"]
    queue["merge_gate"]["output_manifest"] = "manifest.yaml"
    _approve_all_jobs_and_gates(queue)

    layer_map_path = tmp_path / "layer_map.yaml"
    feedback_path = tmp_path / "feedback.yaml"
    queue_path = tmp_path / "queue.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    _write_yaml(layer_map_path, layer_map)
    _write_yaml(feedback_path, feedback)
    _write_yaml(queue_path, queue)
    _write_yaml(manifest_path, manifest)
    return queue_path, feedback_path, layer_map_path, manifest_path


def test_queue_sample_is_valid_but_not_merge_ready() -> None:
    report = validate_asset_generation_queue(_sample())

    assert report.valid
    assert not report.merge_ready


def test_queue_requires_all_five_part_families() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    jobs[:] = [job for job in jobs if job["part_family"] != "mouth"]

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("missing required part family: mouth" in issue.message for issue in report.issues)


def test_queue_jobs_must_be_parallelizable() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    jobs[0]["can_run_in_parallel"] = False

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("can_run_in_parallel" in issue.path for issue in report.issues)


def test_approved_jobs_and_merge_validations_are_merge_ready() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    data["feedback_inputs"] = []
    for job in jobs:
        job["feedback_refs"] = []
    _approve_all_jobs_and_gates(data)

    report = validate_asset_generation_queue(data)

    assert report.valid
    assert report.merge_ready


def test_approved_job_requires_all_job_validations() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    jobs[0]["status"] = "approved"

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("all required validations must be true" in issue.message for issue in report.issues)


def test_merge_gate_rejects_unknown_job() -> None:
    data = deepcopy(_sample())
    merge_gate = data["merge_gate"]
    assert isinstance(merge_gate, dict)
    merge_gate["required_jobs"].append("accessories")

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("unknown job id" in issue.message for issue in report.issues)


def test_queue_rejects_target_layer_owned_by_multiple_jobs() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    jobs[1]["targets"].append(jobs[0]["targets"][0])

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("target layer is already owned" in issue.message for issue in report.issues)


def test_required_jobs_cannot_depend_on_each_other() -> None:
    data = deepcopy(_sample())
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    jobs[1]["depends_on"] = ["eyes"]

    report = validate_asset_generation_queue(data)

    assert not report.valid
    assert any("must start without dependencies" in issue.message for issue in report.issues)


def test_unresolved_high_feedback_blocks_merge_even_if_gate_claims_resolved() -> None:
    data = deepcopy(_sample())
    feedback_inputs = data["feedback_inputs"]
    assert isinstance(feedback_inputs, list)
    feedback_ref = feedback_inputs[0]
    feedback = load_yaml_mapping(Path(str(feedback_ref)))
    _approve_all_jobs_and_gates(data)

    report = validate_asset_generation_queue(
        data,
        feedback_documents={str(feedback_ref): feedback},
    )

    assert not report.valid
    assert not report.merge_ready
    assert any("cannot be true" in issue.message for issue in report.issues)


def test_unresolved_medium_feedback_also_blocks_merge() -> None:
    data = deepcopy(_sample())
    feedback_inputs = data["feedback_inputs"]
    assert isinstance(feedback_inputs, list)
    feedback_ref = feedback_inputs[0]
    feedback = load_yaml_mapping(Path(str(feedback_ref)))
    feedback["feedback"][0]["severity"] = "medium"
    _approve_all_jobs_and_gates(data)

    report = validate_asset_generation_queue(
        data,
        feedback_documents={str(feedback_ref): feedback},
    )

    assert not report.valid
    assert not report.merge_ready
    assert any("unresolved or unverified" in issue.message for issue in report.issues)


def test_queue_project_must_match_feedback_project() -> None:
    data = deepcopy(_sample())
    feedback_inputs = data["feedback_inputs"]
    assert isinstance(feedback_inputs, list)
    feedback_ref = feedback_inputs[0]
    feedback = load_yaml_mapping(Path(str(feedback_ref)))
    feedback["feedback"][0]["status"] = "resolved"
    data["project"] = "different-project"
    _approve_all_jobs_and_gates(data)

    report = validate_asset_generation_queue(
        data,
        feedback_documents={str(feedback_ref): feedback},
    )

    assert not report.valid
    assert not report.merge_ready
    assert any(
        "feedback project must match queue project" in issue.message for issue in report.issues
    )


def test_resolved_high_feedback_can_pass_merge_gate() -> None:
    data = deepcopy(_sample())
    feedback_inputs = data["feedback_inputs"]
    assert isinstance(feedback_inputs, list)
    feedback_ref = feedback_inputs[0]
    feedback = load_yaml_mapping(Path(str(feedback_ref)))
    feedback["feedback"][0]["status"] = "resolved"
    _approve_all_jobs_and_gates(data)

    report = validate_asset_generation_queue(
        data,
        feedback_documents={str(feedback_ref): feedback},
    )

    assert report.valid
    assert report.merge_ready


def test_queue_cli_loads_relative_feedback_and_manifest_from_base_dir(tmp_path: Path) -> None:
    queue_path, _feedback_path, _layer_map_path, _manifest_path = _write_cli_fixture(tmp_path)

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--manifest",
            "manifest.yaml",
            "--require-merge-ready",
        ]
    )

    assert exit_code == 0


def test_queue_cli_rejects_missing_feedback_file(tmp_path: Path) -> None:
    queue_path, feedback_path, _layer_map_path, _manifest_path = _write_cli_fixture(tmp_path)
    feedback_path.unlink()

    assert main([str(queue_path), "--base-dir", str(tmp_path)]) == 2


def test_queue_cli_rejects_feedback_layer_map_project_mismatch(tmp_path: Path) -> None:
    queue_path, _feedback_path, layer_map_path, _manifest_path = _write_cli_fixture(tmp_path)
    layer_map = load_yaml_mapping(layer_map_path)
    layer_map["project"] = "different-project"
    _write_yaml(layer_map_path, layer_map)

    assert main([str(queue_path), "--base-dir", str(tmp_path)]) == 2


def test_queue_cli_rejects_missing_feedback_layer_map_reference(tmp_path: Path) -> None:
    queue_path, feedback_path, _layer_map_path, _manifest_path = _write_cli_fixture(tmp_path)
    feedback = load_yaml_mapping(feedback_path)
    feedback["model_refs"]["layer_map"] = "missing-layer-map.yaml"
    _write_yaml(feedback_path, feedback)

    assert main([str(queue_path), "--base-dir", str(tmp_path)]) == 2


def test_queue_cli_rejects_handoff_manifest_project_mismatch(tmp_path: Path) -> None:
    queue_path, _feedback_path, _layer_map_path, manifest_path = _write_cli_fixture(tmp_path)
    manifest = load_yaml_mapping(manifest_path)
    manifest["project"] = "different-project"
    _write_yaml(manifest_path, manifest)

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--manifest",
            "manifest.yaml",
            "--require-merge-ready",
        ]
    )

    assert exit_code == 1


def test_queue_cli_rejects_handoff_manifest_source_mismatch(tmp_path: Path) -> None:
    queue_path, _feedback_path, _layer_map_path, manifest_path = _write_cli_fixture(tmp_path)
    manifest = load_yaml_mapping(manifest_path)
    manifest["source_image"]["path"] = "assets/source/different.png"
    _write_yaml(manifest_path, manifest)

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--manifest",
            "manifest.yaml",
            "--require-merge-ready",
        ]
    )

    assert exit_code == 1


def test_queue_cli_rejects_manifest_path_not_declared_by_queue(tmp_path: Path) -> None:
    queue_path, _feedback_path, _layer_map_path, manifest_path = _write_cli_fixture(tmp_path)
    other_manifest_path = tmp_path / "other-manifest.yaml"
    other_manifest_path.write_bytes(manifest_path.read_bytes())

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--manifest",
            "other-manifest.yaml",
        ]
    )

    assert exit_code == 1
