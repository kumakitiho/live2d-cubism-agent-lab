from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from PIL import Image, ImageChops

import tools.asset_generation_orchestrator as orchestrator_module
from tools.asset_generation_orchestrator import main as orchestrator_main
from tools.asset_pipeline_common import write_yaml


def _save_mask(
    path: Path,
    size: tuple[int, int],
    points: set[tuple[int, int]],
) -> None:
    image = Image.new("L", size, 0)
    for point in points:
        image.putpixel(point, 255)
    image.save(path)


def _asset(
    layer_id: str,
    *,
    generation_method: str,
    draw_order: int,
) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        "layer_name": layer_id,
        "layer_path": layer_id,
        "role": layer_id,
        "side": "C",
        "source_file": f"canonical/{layer_id}.png",
        "target_mask": f"{layer_id}.target.png",
        "protect_mask": f"{layer_id}.protect.png",
        "edge_extension_mask": f"{layer_id}.edge.png",
        "inpaint_mask": f"{layer_id}.inpaint.png",
        "prompt_id": f"prompt-{layer_id}",
        "generation_method": generation_method,
        "dependencies": [],
        "draw_order": draw_order,
        "overlap_margin_px": 0,
        "quality_status": "pending",
        "refinement_attempts": 0,
        "required": True,
        "inferred": generation_method == "inpaint",
        "review_required": generation_method == "inpaint",
        "readiness": "planned",
        "include_in_import": True,
        "segmentation": {
            "semantic_prompt": layer_id,
            "candidate_count": 1,
            "fixture_masks": [f"{layer_id}.target.png"],
        },
    }


def _fixture(tmp_path: Path, *, failed_extract: bool = False) -> Path:
    size = (10, 8)
    source = Image.new("RGBA", size, (0, 0, 0, 0))
    visible = {(3, y) for y in range(2, 6)} | {(7, y) for y in range(2, 6)}
    hidden = {(x, y) for x in range(4, 7) for y in range(2, 6)}
    for point in visible:
        source.putpixel(point, (80, 100, 120, 255))
    source.save(tmp_path / "source.png")
    _save_mask(tmp_path / "face.target.png", size, visible | hidden)
    _save_mask(tmp_path / "face.protect.png", size, visible)
    _save_mask(tmp_path / "face.edge.png", size, set())
    _save_mask(tmp_path / "face.inpaint.png", size, hidden)
    assets = [_asset("face", generation_method="inpaint", draw_order=10)]
    if failed_extract:
        failed_point = {(1, 1)}
        _save_mask(tmp_path / "empty.target.png", size, failed_point)
        _save_mask(tmp_path / "empty.protect.png", size, set())
        _save_mask(tmp_path / "empty.edge.png", size, set())
        _save_mask(tmp_path / "empty.inpaint.png", size, set())
        assets.append(_asset("empty", generation_method="extract", draw_order=20))
    queue = {
        "schema_version": 3,
        "validation_mode": "strict",
        "project": "integration-fixture",
        "source_image": {"path": "source.png", "rights_status": "confirmed"},
        "canvas": {
            "width": size[0],
            "height": size[1],
            "color_mode": "RGBA",
            "bit_depth": 8,
            "color_profile": "sRGB",
        },
        "assets": assets,
    }
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return queue_path


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "queue.yaml",
        "--base-dir",
        str(tmp_path),
        "--segmentation-backend",
        "mock",
        "--inpainting-backend",
        "mock",
        "--run-id",
        "run-001",
        "--output-dir",
        "generated/runs/run-001",
        *extra,
    ]


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _install_quality_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: str,
    failed_parts: int,
    visual_score: float,
) -> list[list[str]]:
    original_quality_main = orchestrator_module.quality_main
    calls: list[list[str]] = []

    def scenario_quality(argv: list[str] | None = None) -> int:
        assert argv is not None
        calls.append(argv)
        code = original_quality_main(argv)
        assert code == 0
        base_dir = Path(argv[argv.index("--base-dir") + 1])
        output = base_dir / argv[argv.index("--output") + 1]
        report = _load(output)
        parts = report["parts"]
        assert isinstance(parts, list) and parts
        for part in parts:
            part["quality_status"] = "pass"
            part["failed_checks"] = []
        if failed_parts:
            assert failed_parts == 1
            report["thresholds"]["max_edge_continuity_score"] = 0.1
            parts[0]["metrics"]["edge_continuity_score"] = 0.5
            parts[0]["quality_status"] = "fail"
            parts[0]["failed_checks"] = ["edge_continuity_score"]
        report["thresholds"]["max_visual_reconstruction_difference_score"] = 0.1
        report["summary"].update(
            {
                "result": result,
                "failed_parts": failed_parts,
                "visual_reconstruction_difference_score": visual_score,
            }
        )
        write_yaml(output, report, force=True)
        return 0

    monkeypatch.setattr(orchestrator_module, "quality_main", scenario_quality)
    return calls


def test_dry_run_changes_no_files_and_loads_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry-run must not execute a backend")

    monkeypatch.setattr(orchestrator_module, "segmentation_main", fail)
    monkeypatch.setattr(orchestrator_module, "inpainting_main", fail)

    assert orchestrator_main(_arguments(tmp_path)) == 0
    assert queue.read_bytes() == before
    assert not (tmp_path / "generated").exists()


def test_mock_end_to_end_preserves_canonical_queue_and_masks(tmp_path: Path) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()

    assert orchestrator_main(_arguments(tmp_path, "--auto-approve-mock", "--execute")) == 0

    run_root = tmp_path / "generated/runs/run-001"
    state = _load(run_root / "run.yaml")
    run_schema = _load(Path("schemas/asset_generation_run.schema.yaml"))
    Draft202012Validator(run_schema).validate(state)
    assert state["outcome"] == "completed"
    assert all(
        state["stages"][name]["status"] in {"completed", "skipped"} for name in state["stages"]
    )
    assert queue.read_bytes() == before
    candidate = _load(tmp_path / state["queue_candidate"])
    selected_path = tmp_path / candidate["assets"][0]["source_file"]
    selected = Image.open(selected_path).convert("RGBA")
    source = Image.open(tmp_path / "source.png").convert("RGBA")
    protect = Image.open(tmp_path / "face.protect.png").convert("L")
    inpaint = Image.open(tmp_path / "face.inpaint.png").convert("L")
    outside = ImageChops.invert(inpaint)
    transparent = Image.new("RGBA", source.size)
    assert (
        ImageChops.difference(
            Image.composite(source, transparent, protect),
            Image.composite(selected, transparent, protect),
        ).getbbox()
        is None
    )
    assert (
        ImageChops.difference(
            Image.composite(source, transparent, outside),
            Image.composite(selected, transparent, outside),
        ).getbbox()
        is None
    )
    assert (run_root / "queue-candidates/diff-summary.yaml").is_file()
    provenance = _load(run_root / "inpainting/provenance.yaml")
    assert provenance["candidates"][0]["run_id"] == "run-001"
    assert "prompt" in provenance["candidates"][0]
    segmentation_provenance = _load(run_root / "segmentation/provenance.yaml")
    segmentation_candidate = segmentation_provenance["candidates"][0]
    assert segmentation_candidate["source_request"]["request"]["layer_id"] == "face"
    assert segmentation_candidate["quality_metrics"]
    assert segmentation_candidate["selection_reason"] == "assignment_selected_rank_1"


def test_inpainting_backend_defaults_and_cli_overrides(tmp_path: Path) -> None:
    mock = orchestrator_module.build_inpainting_backend_config("mock", base_dir=tmp_path)
    diffusers = orchestrator_module.build_inpainting_backend_config(
        "diffusers",
        base_dir=tmp_path,
    )
    flux = orchestrator_module.build_inpainting_backend_config(
        "flux_fill",
        base_dir=tmp_path,
    )
    overridden = orchestrator_module.build_inpainting_backend_config(
        "diffusers",
        base_dir=tmp_path,
        width=768,
        height=640,
        padding=48,
        max_edge_continuity_score=0.2,
        max_boundary_color_difference_score=0.3,
        max_visual_reconstruction_difference_score=0.4,
    )

    assert mock["model_size"] == [64, 64]
    assert mock["padding"] == 2
    assert mock["quality_thresholds"]["max_edge_continuity_score"] == 1.0
    assert mock["quality_thresholds"]["max_boundary_color_difference_score"] == 1.0
    assert mock["quality_thresholds"]["max_visual_reconstruction_difference_score"] == 1.0
    assert diffusers["model_size"] == [512, 512]
    assert diffusers["padding"] == 32
    assert diffusers["quality_thresholds"] == (orchestrator_module.DEFAULT_QUALITY_THRESHOLDS)
    assert flux["model_size"] == [1024, 1024]
    assert flux["quality_thresholds"] == orchestrator_module.DEFAULT_QUALITY_THRESHOLDS
    assert overridden["model_size"] == [768, 640]
    assert overridden["padding"] == 48
    assert overridden["quality_thresholds"]["max_edge_continuity_score"] == 0.2
    assert overridden["quality_thresholds"]["max_boundary_color_difference_score"] == 0.3
    assert overridden["quality_thresholds"]["max_visual_reconstruction_difference_score"] == 0.4


def test_configuration_digest_includes_resolved_backend_defaults(tmp_path: Path) -> None:
    args = orchestrator_module.build_parser().parse_args(_arguments(tmp_path))
    limits = orchestrator_module.ResourceLimits()
    first = orchestrator_module.build_inpainting_backend_config("mock", base_dir=tmp_path)
    second = deepcopy(first)
    second["model_size"] = [96, 96]

    assert orchestrator_module._configuration_sha256(
        args,
        limits,
        first,
    ) != orchestrator_module._configuration_sha256(args, limits, second)


def test_gpu_memory_estimates_reach_scheduled_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path)
    observed: dict[str, int] = {}
    original_run = orchestrator_module.ResourceScheduler.run

    def tracking_run(
        scheduler: orchestrator_module.ResourceScheduler,
        tasks: list[orchestrator_module.ScheduledTask],
    ) -> dict[str, Any]:
        observed.update({task.name: task.gpu_memory_mb for task in tasks})
        return original_run(scheduler, tasks)

    monkeypatch.setattr(orchestrator_module.ResourceScheduler, "run", tracking_run)

    assert (
        orchestrator_main(
            _arguments(
                tmp_path,
                "--segmentation-gpu-memory-mb",
                "1024",
                "--inpainting-gpu-memory-mb",
                "2048",
                "--inpainting-width",
                "80",
                "--inpainting-height",
                "72",
                "--inpainting-padding",
                "6",
                "--inpainting-max-edge-continuity-score",
                "0.8",
                "--auto-approve-mock",
                "--execute",
            )
        )
        == 0
    )
    assert observed["segmentation"] == 1024
    assert observed["inpaint:face"] == 2048
    request = _load(tmp_path / "generated/runs/run-001/inpainting/face/request.yaml")
    assert request["backend_config"]["model_size"] == [80, 72]
    assert request["backend_config"]["padding"] == 6
    assert request["backend_config"]["quality_thresholds"]["max_edge_continuity_score"] == 0.8


def test_segmentation_only_stops_at_assignment_review(tmp_path: Path) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"

    assert orchestrator_main(args) == 0

    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert state["outcome"] == "waiting_for_review"
    assert state["stages"]["assignment"]["status"] == "waiting_for_review"
    assert state["stages"]["extraction"]["status"] == "planned"
    assert queue.read_bytes() == before


def test_inpainting_only_uses_existing_masks_and_stops_at_selection(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--segmentation-backend"))] = "disabled"

    assert orchestrator_main(args) == 0

    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert state["stages"]["segmentation"]["status"] == "skipped"
    assert state["stages"]["extraction"]["status"] == "completed"
    assert state["stages"]["inpainting"]["status"] == "waiting_for_review"


def test_auto_approve_mock_is_rejected_for_real_backend(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--auto-approve-mock")
    args[args.index("mock", args.index("--inpainting-backend"))] = "diffusers"

    assert orchestrator_main(args) == 2
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("unsafe", ["absolute_path", "credential"])
def test_queue_secrets_and_absolute_paths_are_rejected(
    tmp_path: Path,
    unsafe: str,
) -> None:
    queue_path = _fixture(tmp_path)
    queue = _load(queue_path)
    if unsafe == "absolute_path":
        queue["source_image"]["path"] = str((tmp_path / "source.png").resolve())
    else:
        queue["assets"][0]["api_token"] = "secret"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert orchestrator_main(_arguments(tmp_path)) == 2
    assert not (tmp_path / "generated").exists()


def test_resume_skips_completed_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    assignment_path = tmp_path / "generated/runs/run-001/assignments/assignment.yaml"
    assignment = _load(assignment_path)
    assignment["review_status"] = "approved"
    assignment["assignments"][0]["status"] = "approved"
    assignment["assignments"][0]["requires_review"] = False
    assignment_path.write_text(
        yaml.safe_dump(assignment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("completed segmentation must not run again")

    monkeypatch.setattr(orchestrator_module, "segmentation_main", fail)

    assert orchestrator_main([*args, "--resume"]) == 0
    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert state["outcome"] == "completed"


def test_resume_rejects_stale_queue(tmp_path: Path) -> None:
    queue = _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    queue.write_bytes(queue.read_bytes() + b"\n# changed\n")

    assert orchestrator_main([*args, "--resume"]) == 2


def test_resume_rejects_backend_configuration_change(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    changed = deepcopy(args)
    changed[changed.index("mock", changed.index("--segmentation-backend"))] = "disabled"

    assert orchestrator_main([*changed, "--resume", "--auto-approve-mock"]) == 2
    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert state["stages"]["assignment"]["status"] == "waiting_for_review"


def test_resume_rejects_mixed_run_id_artifact(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    assignment_path = tmp_path / "generated/runs/run-001/assignments/assignment.yaml"
    assignment = _load(assignment_path)
    assignment["segmentation_run_id"] = "other-run"
    assignment_path.write_text(yaml.safe_dump(assignment), encoding="utf-8")

    assert orchestrator_main([*args, "--resume"]) == 2


def test_resume_rejects_assignment_candidate_tampering(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    assignment_path = tmp_path / "generated/runs/run-001/assignments/assignment.yaml"
    assignment = _load(assignment_path)
    assignment["review_status"] = "approved"
    assignment["assignments"][0]["status"] = "approved"
    assignment["assignments"][0]["requires_review"] = False
    assignment["assignments"][0]["target_mask"] = "face.target.png"
    assignment_path.write_text(
        yaml.safe_dump(assignment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert orchestrator_main([*args, "--resume"]) == 2


def test_resume_rejects_tampered_inpainting_selection(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--segmentation-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    selection_path = tmp_path / "generated/runs/run-001/inpainting/face/selection.yaml"
    selection = _load(selection_path)
    selection["review"] = {"status": "approved", "reviewer": "human", "notes": "ok"}
    selection["selected_candidate"]["output_file"] = "tampered.png"
    selection_path.write_text(
        yaml.safe_dump(selection, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert orchestrator_main([*args, "--resume"]) == 2


def test_output_collision_and_base_dir_escape_are_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--execute")
    args[args.index("mock", args.index("--inpainting-backend"))] = "disabled"
    assert orchestrator_main(args) == 0
    assert orchestrator_main(args) == 2
    escaped = deepcopy(args)
    output_index = escaped.index("--output-dir") + 1
    escaped[output_index] = "../outside"
    escaped[escaped.index("--run-id") + 1] = "outside-run"
    assert orchestrator_main(escaped) == 2


def test_layer_id_path_traversal_is_rejected_before_output(tmp_path: Path) -> None:
    queue_path = _fixture(tmp_path)
    queue = _load(queue_path)
    queue["assets"][0]["layer_id"] = "../../outside"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert orchestrator_main(_arguments(tmp_path, "--execute")) == 2
    assert not (tmp_path / "generated").exists()
    assert not (tmp_path.parent / "outside.png").exists()


def test_refinement_requeues_only_failed_part(tmp_path: Path) -> None:
    _fixture(tmp_path, failed_extract=True)

    assert orchestrator_main(_arguments(tmp_path, "--auto-approve-mock", "--execute")) == 0

    run_root = tmp_path / "generated/runs/run-001"
    state = _load(run_root / "run.yaml")
    plan = _load(run_root / "refinement/plan.yaml")
    assert state["outcome"] == "refinement_required"
    assert [job["layer_id"] for job in plan["jobs"]] == ["empty"]


def test_completed_stage_digest_manifests_cover_required_artifacts(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert orchestrator_main(_arguments(tmp_path, "--auto-approve-mock", "--execute")) == 0
    state = _load(tmp_path / "generated/runs/run-001/run.yaml")

    extraction = set(state["stages"]["extraction"]["artifact_sha256"])
    inpainting = set(state["stages"]["inpainting"]["artifact_sha256"])
    quality = set(state["stages"]["quality"]["artifact_sha256"])
    refinement = set(state["stages"]["refinement"]["artifact_sha256"])
    assert any(path.endswith("queue-candidates/after-extraction.yaml") for path in extraction)
    assert any(path.endswith("extracted-parts/face.png") for path in extraction)
    assert any(path.endswith("inpainting/face/request.yaml") for path in inpainting)
    assert any(path.endswith("inpainting/face/result.yaml") for path in inpainting)
    assert any(path.endswith("inpainting/face/selection.yaml") for path in inpainting)
    assert any(path.endswith("quality/result.yaml") for path in quality)
    assert any(path.endswith("quality/difference.png") for path in quality)
    assert any(path.endswith("refinement/plan.yaml") for path in refinement)
    assert any(path.endswith("queue-candidates/refined.yaml") for path in refinement)


@pytest.mark.parametrize(
    ("stage_name", "artifact"),
    [
        ("extraction", "extracted-parts/face.png"),
        ("inpainting", "inpainting/face/request.yaml"),
        ("quality", "quality/difference.png"),
        ("refinement", "refinement/plan.yaml"),
    ],
)
def test_resume_rejects_completed_stage_artifact_tampering(
    tmp_path: Path,
    stage_name: str,
    artifact: str,
) -> None:
    _fixture(tmp_path)
    args = _arguments(tmp_path, "--auto-approve-mock", "--execute")
    assert orchestrator_main(args) == 0
    artifact_path = tmp_path / "generated/runs/run-001" / artifact
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    assert orchestrator_main([*args, "--resume"]) == 2
    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert state["outcome"] == "failed"
    assert state["stages"][stage_name]["status"] == "failed"


def test_unattributed_global_quality_failure_waits_for_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()
    quality_calls = 0

    def unattributed_quality(argv: list[str] | None = None) -> int:
        nonlocal quality_calls
        quality_calls += 1
        assert argv is not None
        base_dir = Path(argv[argv.index("--base-dir") + 1])
        output = base_dir / argv[argv.index("--output") + 1]
        difference = base_dir / argv[argv.index("--difference-output") + 1]
        Image.new("RGBA", (10, 8), (0, 0, 0, 0)).save(difference)
        write_yaml(
            output,
            {
                "schema_version": 2,
                "thresholds": {
                    "max_visual_reconstruction_difference_score": 0.1,
                },
                "summary": {
                    "result": "fail",
                    "failed_parts": 0,
                    "visual_reconstruction_difference_score": 0.5,
                },
            },
        )
        return 0

    monkeypatch.setattr(orchestrator_module, "quality_main", unattributed_quality)

    args = _arguments(tmp_path, "--auto-approve-mock", "--execute")
    assert orchestrator_main(args) == 0
    state = _load(tmp_path / "generated/runs/run-001/run.yaml")
    Draft202012Validator(_load(Path("schemas/asset_generation_run.schema.yaml"))).validate(state)
    assert state["outcome"] == "manual_review_required"
    assert state["stages"]["quality"]["status"] == "waiting_for_review"
    assert state["stages"]["quality"]["manual_review_required"] is True
    assert state["stages"]["refinement"]["status"] == "blocked"
    assert quality_calls == 1

    assert orchestrator_main([*args, "--resume"]) == 0
    assert quality_calls == 1
    resumed = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert resumed["outcome"] == "manual_review_required"

    difference = tmp_path / "generated/runs/run-001/quality/difference.png"
    difference.write_bytes(difference.read_bytes() + b"tampered")
    assert orchestrator_main([*args, "--resume"]) == 2
    assert quality_calls == 1
    failed = _load(tmp_path / "generated/runs/run-001/run.yaml")
    assert failed["outcome"] == "failed"
    assert failed["stages"]["quality"]["status"] == "failed"
    assert queue.read_bytes() == before


@pytest.mark.parametrize(
    "visual_score",
    [
        pytest.param(0.5, id="part-and-global-failure"),
        pytest.param(0.05, id="part-failure-only"),
    ],
)
def test_attributed_part_failure_reaches_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visual_score: float,
) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()
    quality_calls = _install_quality_scenario(
        monkeypatch,
        result="fail",
        failed_parts=1,
        visual_score=visual_score,
    )

    assert orchestrator_main(_arguments(tmp_path, "--auto-approve-mock", "--execute")) == 0

    run_root = tmp_path / "generated/runs/run-001"
    state = _load(run_root / "run.yaml")
    plan = _load(run_root / "refinement/plan.yaml")
    assert len(quality_calls) == 1
    assert state["outcome"] == "refinement_required"
    assert state["stages"]["quality"]["status"] == "completed"
    assert "manual_review_required" not in state["stages"]["quality"]
    assert state["stages"]["refinement"]["status"] == "completed"
    assert plan["summary"]["failed_parts"] == 1
    assert [job["layer_id"] for job in plan["jobs"]] == ["face"]
    assert queue.read_bytes() == before


def test_complete_quality_pass_finishes_without_refinement_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _fixture(tmp_path)
    before = queue.read_bytes()
    _install_quality_scenario(
        monkeypatch,
        result="pass",
        failed_parts=0,
        visual_score=0.05,
    )

    assert orchestrator_main(_arguments(tmp_path, "--auto-approve-mock", "--execute")) == 0

    run_root = tmp_path / "generated/runs/run-001"
    state = _load(run_root / "run.yaml")
    plan = _load(run_root / "refinement/plan.yaml")
    assert state["outcome"] == "completed"
    assert state["stages"]["quality"]["status"] == "completed"
    assert state["stages"]["refinement"]["status"] == "completed"
    assert plan["summary"]["failed_parts"] == 0
    assert plan["jobs"] == []
    assert queue.read_bytes() == before


@pytest.mark.parametrize(
    "summary",
    [
        {"result": "warn", "failed_parts": 0},
        {"result": "fail", "failed_parts": True},
        {"result": "fail", "failed_parts": -1},
        {"result": "fail", "failed_parts": "1"},
    ],
)
def test_quality_summary_rejects_invalid_result_and_failed_parts(
    summary: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        orchestrator_module._validated_quality_summary({"summary": summary})


def test_run_schema_and_sample_validate() -> None:
    schema = _load(Path("schemas/asset_generation_run.schema.yaml"))
    sample = _load(Path("examples/asset_generation_run.sample.yaml"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(sample)


def test_common_yaml_writer_uses_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replaced: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def tracking_replace(source: Path, target: Path) -> Path:
        replaced.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)
    output = tmp_path / "quality.yaml"

    write_yaml(output, {"status": "completed"})

    assert replaced and replaced[0][1] == output
    assert output.read_text(encoding="utf-8") == "status: completed\n"
    assert not list(tmp_path.glob("*.tmp"))
