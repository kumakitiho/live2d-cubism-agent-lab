from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from PIL import Image, ImageChops
from referencing import Registry, Resource

import tools.generative_inpainter as inpainter_module
from tools.artifact_validation import load_yaml_mapping
from tools.backends.inpainting.base import BackendStatus
from tools.backends.inpainting.diffusers_backend import DiffusersInpaintingBackend
from tools.backends.inpainting.mock import MockInpaintingBackend
from tools.generative_inpainter import (
    composite_generated_crop,
    evaluate_inpainting_candidate,
    file_sha256,
    inpaint_crop_box,
    validate_inpainting_result,
)
from tools.generative_inpainter import (
    main as inpainter_main,
)
from tools.inpainting_candidate_ranker import (
    apply_selection_to_queue,
    rank_candidates,
)
from tools.inpainting_candidate_ranker import (
    main as ranker_main,
)
from tools.inpainting_prompt_builder import NEGATIVE_CONSTRAINTS, build_inpainting_prompt


def _save_mask(path: Path, size: tuple[int, int], values: dict[tuple[int, int], int]) -> None:
    mask = Image.new("L", size, 0)
    for point, value in values.items():
        mask.putpixel(point, value)
    mask.save(path)


def _fixture(tmp_path: Path, *, soft_center: bool = False) -> Path:
    size = (10, 8)
    baseline = Image.new("RGBA", size, (0, 0, 0, 0))
    visible = {(3, y) for y in range(2, 6)} | {(7, y) for y in range(2, 6)}
    for point in visible:
        baseline.putpixel(point, (80, 100, 120, 255))
    baseline.save(tmp_path / "source.png")
    baseline.save(tmp_path / "part.png")
    inpaint_points = {(x, y) for x in range(4, 7) for y in range(2, 6)}
    inpaint_values = {point: 255 for point in inpaint_points}
    if soft_center:
        inpaint_values[(5, 3)] = 128
    _save_mask(tmp_path / "inpaint.png", size, inpaint_values)
    _save_mask(tmp_path / "protect.png", size, {point: 255 for point in visible})
    _save_mask(tmp_path / "target.png", size, {point: 255 for point in visible | inpaint_points})
    _save_mask(tmp_path / "edge.png", size, {(2, 3): 255})
    request: dict[str, Any] = {
        "schema_version": 1,
        "project": "fixture-project",
        "run_id": "fixture-run",
        "layer_id": "face_hidden_fill",
        "source_image": "source.png",
        "current_part": "part.png",
        "target_mask": "target.png",
        "protect_mask": "protect.png",
        "edge_extension_mask": "edge.png",
        "inpaint_mask": "inpaint.png",
        "prompt": "match source; only modify masked region",
        "negative_prompt": "pose change, canvas resize",
        "backend": "mock",
        "backend_config": {
            "padding": 2,
            "model_size": [16, 12],
            "quality_thresholds": {
                "max_edge_continuity_score": 1.0,
                "max_boundary_color_difference_score": 1.0,
                "max_visual_reconstruction_difference_score": 1.0,
            },
        },
        "candidate_count": 3,
        "seed_policy": {"mode": "explicit_list", "seeds": [11, 12, 13]},
        "output_dir": "generated/candidates",
    }
    request_path = tmp_path / "request.yaml"
    request_path.write_text(
        yaml.safe_dump(request, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return request_path


def _run_fixture(tmp_path: Path, *, soft_center: bool = False) -> dict[str, Any]:
    _fixture(tmp_path, soft_center=soft_center)
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 0
    )
    return load_yaml_mapping(tmp_path / "generated/result.yaml")


def test_mock_end_to_end_preserves_protect_outside_canvas_origin_and_provenance(
    tmp_path: Path,
) -> None:
    result = _run_fixture(tmp_path)
    assert validate_inpainting_result(result) == []
    assert result["canvas"] == {"width": 10, "height": 8, "origin": [0, 0]}
    assert result["inferred"] is True
    assert result["review_required"] is True
    assert result["masks"]["generation_permission"] == "inpaint_mask_only"
    assert len(result["candidates"]) == 3
    assert len({candidate["candidate_id"] for candidate in result["candidates"]}) == 3
    assert [candidate["seed"] for candidate in result["candidates"]] == [11, 12, 13]
    baseline = Image.open(tmp_path / "part.png").convert("RGBA")
    inpaint = Image.open(tmp_path / "inpaint.png").convert("L")
    protect = Image.open(tmp_path / "protect.png").convert("L")
    outside = ImageChops.invert(inpaint.point(lambda value: 255 if value else 0, mode="L"))
    outputs: list[bytes] = []
    for metadata in result["candidates"]:
        candidate_path = tmp_path / metadata["output_file"]
        preview_path = tmp_path / metadata["preview_file"]
        assert candidate_path.is_file()
        assert preview_path.is_file()
        assert metadata["output_sha256"] == file_sha256(candidate_path)
        assert metadata["preview_sha256"] == file_sha256(preview_path)
        candidate = Image.open(candidate_path).convert("RGBA")
        outputs.append(candidate.tobytes())
        assert ImageChops.difference(
            Image.composite(baseline, Image.new("RGBA", baseline.size), protect),
            Image.composite(candidate, Image.new("RGBA", baseline.size), protect),
        ).getbbox() is None
        assert ImageChops.difference(
            Image.composite(baseline, Image.new("RGBA", baseline.size), outside),
            Image.composite(candidate, Image.new("RGBA", baseline.size), outside),
        ).getbbox() is None
        assert metadata["crop_box"] == [2, 0, 9, 8]
        assert metadata["resize_from"] == [7, 8]
        assert metadata["resize_to"] == [16, 12]
        assert metadata["requires_review"] is True
        assert metadata["quality_metrics"]["inpaint_region_source_difference_score"] > 0
    assert len(set(outputs)) == 3
    assert not list((tmp_path / "generated").rglob("*.tmp"))


def test_soft_mask_compositing_produces_partial_alpha(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, soft_center=True)
    candidate = Image.open(tmp_path / result["candidates"][0]["output_file"]).convert("RGBA")
    assert 0 < candidate.getpixel((5, 3))[3] < 255


def test_crop_padding_restore_and_protect_restoration_helpers() -> None:
    baseline = Image.new("RGBA", (6, 5), (10, 20, 30, 255))
    inpaint = Image.new("L", baseline.size, 0)
    inpaint.putpixel((3, 2), 128)
    protect = Image.new("L", baseline.size, 0)
    protect.putpixel((3, 2), 255)
    box = inpaint_crop_box(inpaint, 1)
    assert box == (2, 1, 5, 4)
    generated = Image.new("RGBA", (3, 3), (200, 10, 20, 255))
    result = composite_generated_crop(baseline, generated, inpaint, protect, box)
    assert result.tobytes() == baseline.tobytes()


def test_dry_run_does_not_generate_or_write(tmp_path: Path, monkeypatch: Any) -> None:
    _fixture(tmp_path)

    def fail_generate(*args: Any, **kwargs: Any) -> Image.Image:
        raise AssertionError("dry-run must not invoke backend generation")

    monkeypatch.setattr(
        "tools.backends.inpainting.mock.MockInpaintingBackend.generate", fail_generate
    )
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
            ]
        )
        == 0
    )
    assert not (tmp_path / "generated").exists()


def test_diffusers_dry_run_does_not_load_pipeline(tmp_path: Path, monkeypatch: Any) -> None:
    _fixture(tmp_path)
    backend = DiffusersInpaintingBackend()

    def fail_load(config: Any) -> Any:
        raise AssertionError("dry-run must not load a Diffusers pipeline")

    monkeypatch.setattr(backend, "_load_pipeline", fail_load)
    monkeypatch.setattr(inpainter_module, "create_backend", lambda name: backend)
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--backend",
                "diffusers",
                "--model-id",
                "local/model",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
            ]
        )
        == 0
    )
    assert backend._pipeline is None
    assert not (tmp_path / "generated").exists()


def test_unavailable_backend_stops_before_output(tmp_path: Path, monkeypatch: Any) -> None:
    _fixture(tmp_path)

    class UnavailableBackend:
        name = "diffusers"
        recommended_size = 512

        def status(self) -> BackendStatus:
            return BackendStatus("diffusers", False, "optional dependency unavailable")

    monkeypatch.setattr(inpainter_module, "create_backend", lambda name: UnavailableBackend())
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--backend",
                "diffusers",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "generated").exists()


def test_candidate_count_override_requires_enough_explicit_seeds(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--candidate-count",
                "4",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "generated").exists()


def test_output_collision_prevents_partial_publication(tmp_path: Path) -> None:
    _fixture(tmp_path)
    output = tmp_path / "generated/result.yaml"
    output.parent.mkdir(parents=True)
    output.write_text("owned: true\n", encoding="utf-8")
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert output.read_text(encoding="utf-8") == "owned: true\n"
    assert not list((tmp_path / "generated").rglob("*.png"))


def test_failed_later_candidate_publishes_no_partial_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _fixture(tmp_path)
    original_generate = MockInpaintingBackend.generate
    calls = 0

    def fail_second(self: MockInpaintingBackend, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture backend failed on candidate two")
        return original_generate(self, *args, **kwargs)

    monkeypatch.setattr(MockInpaintingBackend, "generate", fail_second)
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "generated").exists()
    assert not list(tmp_path.glob(".inpainting-stage-*"))


def test_source_protect_mismatch_stops_before_generation(tmp_path: Path) -> None:
    _fixture(tmp_path)
    source = Image.open(tmp_path / "source.png").convert("RGBA")
    source.putpixel((3, 3), (200, 20, 20, 255))
    source.save(tmp_path / "source.png")
    assert (
        inpainter_main(
            [
                "request.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "generated").exists()


def _evaluation_fixture() -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    baseline = Image.new("RGBA", (5, 3), (0, 0, 0, 0))
    baseline.putpixel((1, 1), (100, 100, 100, 255))
    candidate = baseline.copy()
    candidate.putpixel((2, 1), (100, 100, 100, 255))
    target = Image.new("L", baseline.size, 0)
    target.putpixel((1, 1), 255)
    target.putpixel((2, 1), 255)
    protect = Image.new("L", baseline.size, 0)
    protect.putpixel((1, 1), 255)
    edge = Image.new("L", baseline.size, 0)
    inpaint = Image.new("L", baseline.size, 0)
    inpaint.putpixel((2, 1), 255)
    return baseline, candidate, target, protect, edge, inpaint


def test_inpaint_source_difference_is_informational_not_rejected() -> None:
    baseline, candidate, target, protect, edge, inpaint = _evaluation_fixture()
    metrics, reasons = evaluate_inpainting_candidate(
        candidate,
        baseline,
        target,
        protect,
        edge,
        inpaint,
        source_image=candidate,
        quality_thresholds={
            "max_edge_continuity_score": 1.0,
            "max_boundary_color_difference_score": 1.0,
        },
    )
    assert metrics["inpaint_region_source_difference_score"] > 0
    assert reasons == []


def test_transparent_rgb_differences_are_ignored_by_preservation_gate() -> None:
    baseline, candidate, target, protect, edge, inpaint = _evaluation_fixture()
    baseline.putpixel((0, 0), (0, 0, 0, 0))
    candidate.putpixel((0, 0), (255, 128, 64, 0))
    protect.putpixel((0, 0), 255)
    metrics, reasons = evaluate_inpainting_candidate(
        candidate,
        baseline,
        target,
        protect,
        edge,
        inpaint,
        source_image=candidate,
        quality_thresholds={
            "max_edge_continuity_score": 1.0,
            "max_boundary_color_difference_score": 1.0,
        },
    )
    assert metrics["protect_difference_px"] == 0
    assert reasons == []


def test_candidate_evaluation_detects_boundary_edge_and_mask_leak() -> None:
    baseline, candidate, target, protect, edge, inpaint = _evaluation_fixture()
    candidate.putpixel((2, 1), (255, 0, 0, 32))
    candidate.putpixel((4, 2), (20, 20, 20, 255))
    _, reasons = evaluate_inpainting_candidate(
        candidate,
        baseline,
        target,
        protect,
        edge,
        inpaint,
        quality_thresholds={
            "max_edge_continuity_score": 0.01,
            "max_boundary_color_difference_score": 0.01,
        },
    )
    assert "inpaint_mask_outside_changed" in reasons
    assert "edge_continuity_failure" in reasons
    assert "boundary_color_failure" in reasons


def test_visual_reconstruction_uses_candidate_beneath_source_context() -> None:
    baseline, candidate, target, protect, edge, inpaint = _evaluation_fixture()
    visible_source = candidate.copy()
    hidden_metrics, hidden_reasons = evaluate_inpainting_candidate(
        candidate,
        baseline,
        target,
        protect,
        edge,
        inpaint,
        source_image=visible_source,
        quality_thresholds={"max_visual_reconstruction_difference_score": 0.0},
    )
    leaking_metrics, leaking_reasons = evaluate_inpainting_candidate(
        candidate,
        baseline,
        target,
        protect,
        edge,
        inpaint,
        source_image=baseline,
        quality_thresholds={"max_visual_reconstruction_difference_score": 0.0},
    )
    assert hidden_metrics["visual_reconstruction_score"] == 0.0
    assert "visual_reconstruction_failure" not in hidden_reasons
    assert leaking_metrics["visual_reconstruction_score"] > 0.0
    assert "visual_reconstruction_failure" in leaking_reasons


def test_ranking_duplicate_ids_all_fail_and_deterministic_selection(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path)
    first = rank_candidates(result)
    second = rank_candidates(result)
    assert first["selected_candidate"]["candidate_id"] == second["selected_candidate"][
        "candidate_id"
    ]
    assert first["review_required"] is True
    assert first["review"]["status"] == "pending"
    duplicate = deepcopy(result)
    duplicate["candidates"][1]["candidate_id"] = duplicate["candidates"][0]["candidate_id"]
    try:
        rank_candidates(duplicate)
    except ValueError as exc:
        assert "duplicate candidate_id" in str(exc)
    else:
        raise AssertionError("duplicate candidate IDs must be rejected")
    all_failed = deepcopy(result)
    for candidate in all_failed["candidates"]:
        candidate["quality_status"] = "fail"
        candidate["rejection_reasons"] = ["schema_invalid"]
    try:
        rank_candidates(all_failed)
    except ValueError as exc:
        assert "all inpainting candidates failed" in str(exc)
    else:
        raise AssertionError("all-failed results must not create a selection")
    strict_violation = deepcopy(result)
    for candidate in strict_violation["candidates"]:
        candidate["quality_metrics"]["inpaint_outside_difference_score"] = 0.01
    try:
        rank_candidates(strict_violation)
    except ValueError as exc:
        assert "all inpainting candidates failed" in str(exc)
    else:
        raise AssertionError("strict preservation failures must override a pass label")
    schema_invalid = deepcopy(result)
    schema_invalid["candidates"][0]["output_file"] = 42
    try:
        rank_candidates(schema_invalid)
    except ValueError as exc:
        assert "output_file must be a PNG path" in str(exc)
    else:
        raise AssertionError("schema-invalid candidate results must be rejected")


def test_rank_cli_and_approved_apply_only_change_selected_part(tmp_path: Path) -> None:
    _run_fixture(tmp_path)
    request_bytes = (tmp_path / "request.yaml").read_bytes()
    assert (
        ranker_main(
            [
                "generated/result.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "request.yaml",
                "--execute",
                "--force",
            ]
        )
        == 2
    )
    assert (tmp_path / "request.yaml").read_bytes() == request_bytes
    assert (
        ranker_main(
            [
                "generated/result.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/selection.yaml",
                "--execute",
            ]
        )
        == 0
    )
    selection = load_yaml_mapping(tmp_path / "generated/selection.yaml")
    queue = {
        "schema_version": 3,
        "project": "fixture-project",
        "assets": [
            {"layer_id": "face_hidden_fill", "source_file": "old.png"},
            {"layer_id": "eye_L", "source_file": "eye.png", "marker": [1, 2, 3]},
        ],
    }
    original = deepcopy(queue)
    try:
        apply_selection_to_queue(queue, selection)
    except ValueError as exc:
        assert "review.status must be approved" in str(exc)
    else:
        raise AssertionError("pending selections must not be applied")
    selection["review"] = {"status": "approved", "reviewer": "human", "notes": "ok"}
    updated = apply_selection_to_queue(queue, selection)
    assert queue == original
    assert updated["assets"][1] == original["assets"][1]
    selected = updated["assets"][0]
    assert selected["source_file"] == selection["selected_candidate"]["output_file"]
    assert selected["generation_method"] == "inpaint"
    assert selected["inferred"] is True
    assert selected["review_required"] is True
    assert selected["readiness"] == "generated"
    tampered = deepcopy(selection)
    tampered["selected_candidate"]["quality_status"] = "fail"
    tampered["selected_candidate"]["rejection_reasons"] = ["mask_leak"]
    try:
        apply_selection_to_queue(queue, tampered)
    except ValueError as exc:
        assert "pass all automatic quality gates" in str(exc)
    else:
        raise AssertionError("a tampered failed candidate must not be applied")
    strict_tampered = deepcopy(selection)
    strict_tampered["selected_candidate"]["quality_metrics"][
        "protect_difference_px"
    ] = 1
    try:
        apply_selection_to_queue(queue, strict_tampered)
    except ValueError as exc:
        assert "strict preservation gate" in str(exc)
    else:
        raise AssertionError("strict preservation metrics must be rechecked on apply")
    path_injected = deepcopy(selection)
    path_injected["selected_candidate"]["output_file"] = "../../outside.png"
    try:
        apply_selection_to_queue(queue, path_injected)
    except ValueError as exc:
        assert "safe relative path" in str(exc)
    else:
        raise AssertionError("candidate paths outside base-dir must not reach the queue")

    queue_path = tmp_path / "generated/queue.yaml"
    approved_path = tmp_path / "generated/selection.approved.yaml"
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    approved_path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    original_queue_bytes = queue_path.read_bytes()
    result_bytes = (tmp_path / "generated/result.yaml").read_bytes()
    assert (
        ranker_main(
            [
                "apply",
                "generated/queue.yaml",
                "generated/selection.approved.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/result.yaml",
                "--execute",
                "--force",
            ]
        )
        == 2
    )
    assert (tmp_path / "generated/result.yaml").read_bytes() == result_bytes
    assert (
        ranker_main(
            [
                "apply",
                "generated/queue.yaml",
                "generated/selection.approved.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/queue.inpainted.yaml",
                "--execute",
            ]
        )
        == 0
    )
    assert queue_path.read_bytes() == original_queue_bytes
    applied = load_yaml_mapping(tmp_path / "generated/queue.inpainted.yaml")
    assert applied["assets"][1] == queue["assets"][1]
    selected_png = tmp_path / selection["selected_candidate"]["output_file"]
    replacement = Image.open(selected_png).convert("RGBA")
    replacement.putpixel((0, 0), (255, 0, 0, 255))
    replacement.save(selected_png)
    assert (
        ranker_main(
            [
                "apply",
                "generated/queue.yaml",
                "generated/selection.approved.yaml",
                "--base-dir",
                str(tmp_path),
                "--output",
                "generated/queue.tampered.yaml",
                "--execute",
            ]
        )
        == 2
    )
    assert not (tmp_path / "generated/queue.tampered.yaml").exists()


def test_prompt_builder_contains_required_context_and_prohibitions() -> None:
    spec = load_yaml_mapping(Path("examples/character_spec.sample.yaml"))
    queue = load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))
    result = build_inpainting_prompt(spec, queue, "face_hidden_fill")
    for key in (
        "character_identity",
        "line_style",
        "palette",
        "lighting_direction",
        "part_role",
        "side",
        "surrounding_geometry",
        "hidden_region_purpose",
        "background",
        "alignment",
        "edit_scope",
    ):
        assert key in result["components"]
    for forbidden in NEGATIVE_CONSTRAINTS:
        assert forbidden in result["negative_prompt"]


def test_sample_and_schemas_are_tracked_contracts() -> None:
    request = load_yaml_mapping(Path("examples/inpainting_request.sample.yaml"))
    assert request["backend_config"]["local_files_only"] is True
    assert request["candidate_count"] == len(request["seed_policy"]["seeds"])
    for name in ("request", "result", "selection"):
        schema = load_yaml_mapping(Path(f"schemas/inpainting_{name}.schema.yaml"))
        assert schema["$schema"].endswith("2020-12/schema")
    result_schema = load_yaml_mapping(Path("schemas/inpainting_result.schema.yaml"))
    selection_schema = load_yaml_mapping(Path("schemas/inpainting_selection.schema.yaml"))
    assert result_schema["$defs"]["candidate"]["additionalProperties"] is False
    assert result_schema["$defs"]["quality_metrics"]["additionalProperties"] is False
    assert selection_schema["properties"]["selected_candidate"]["$ref"].endswith(
        "#/$defs/candidate"
    )


def test_generated_artifacts_validate_against_public_schemas(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path)
    selection = rank_candidates(result)
    request = load_yaml_mapping(tmp_path / "request.yaml")
    request_schema = load_yaml_mapping(Path("schemas/inpainting_request.schema.yaml"))
    result_schema = load_yaml_mapping(Path("schemas/inpainting_result.schema.yaml"))
    selection_schema = load_yaml_mapping(Path("schemas/inpainting_selection.schema.yaml"))
    for schema in (request_schema, result_schema, selection_schema):
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(request_schema).validate(request)
    Draft202012Validator(result_schema).validate(result)
    registry = Registry().with_resource(
        "inpainting_result.schema.yaml", Resource.from_contents(result_schema)
    )
    Draft202012Validator(selection_schema, registry=registry).validate(selection)
