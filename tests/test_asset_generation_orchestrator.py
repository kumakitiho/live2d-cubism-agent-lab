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
