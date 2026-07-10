from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tools.asset_manifest_validator import load_asset_manifest, validate_asset_manifest
from tools.psd_asset_builder import build_psd, create_psd_build_plan


def _sample_manifest() -> dict[str, object]:
    return load_asset_manifest(Path("examples/asset_manifest.sample.yaml"))


def _materialize_handoff_files(manifest: dict[str, object], root: Path) -> None:
    source_image = manifest["source_image"]
    output = manifest["output"]
    canvas = manifest["canvas"]
    parts = manifest["parts"]
    assert isinstance(source_image, dict)
    assert isinstance(output, dict)
    assert isinstance(canvas, dict)
    assert isinstance(parts, list)

    source_path = root / str(source_image["path"])
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"\x89PNG\r\n\x1a\nsource-image")

    for part in parts:
        part_path = root / str(part["source_file"])
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"\x89PNG\r\n\x1a\ntransparent-part")

    psd_path = root / str(output["model_import_psd"])
    psd_path.parent.mkdir(parents=True, exist_ok=True)
    psd_path.write_bytes(b"8BPS-non-empty-placeholder")

    layer_map_path = root / str(output["layer_map"])
    layer_map_path.parent.mkdir(parents=True, exist_ok=True)
    layer_map = {
        "schema_version": 1,
        "project": manifest["project"],
        "canvas": {"width": canvas["width"], "height": canvas["height"]},
        "layers": [
            {
                "layer_id": part["layer_id"],
                "name": part["layer_name"],
            }
            for part in parts
            if part["include_in_import"] is True
        ],
    }
    layer_map_path.write_text(
        yaml.safe_dump(layer_map, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_sample_manifest_is_structurally_valid_but_not_handoff_ready() -> None:
    report = validate_asset_manifest(_sample_manifest())

    assert report.valid
    assert not report.handoff_ready
    assert any("masks_resolved" in issue.path for issue in report.warnings)


def test_duplicate_layer_id_is_rejected() -> None:
    manifest = deepcopy(_sample_manifest())
    parts = manifest["parts"]
    assert isinstance(parts, list)
    duplicate = deepcopy(parts[0])
    duplicate["layer_name"] = "different_name"
    duplicate["order"] = 70
    parts.append(duplicate)

    report = validate_asset_manifest(manifest)

    assert not report.valid
    assert any("duplicate layer id" in issue.message for issue in report.errors)


def test_inferred_asset_must_require_review() -> None:
    manifest = deepcopy(_sample_manifest())
    parts = manifest["parts"]
    assert isinstance(parts, list)
    inferred_part = next(part for part in parts if part["inferred"] is True)
    inferred_part["review_required"] = False

    report = validate_asset_manifest(manifest)

    assert not report.valid
    assert any("when inferred is true" in issue.message for issue in report.errors)


def test_fully_approved_manifest_with_matching_artifacts_is_handoff_ready(
    tmp_path: Path,
) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert report.handoff_ready


def test_approved_flags_without_artifacts_cannot_handoff(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert not report.handoff_ready
    assert any(issue.path == "output.model_import_psd" for issue in report.warnings)


def test_layer_map_mismatch_blocks_handoff(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    output = manifest["output"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(output, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)

    layer_map_path = tmp_path / str(output["layer_map"])
    layer_map = yaml.safe_load(layer_map_path.read_text(encoding="utf-8"))
    layer_map["project"] = "different-project"
    layer_map_path.write_text(yaml.safe_dump(layer_map), encoding="utf-8")

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert not report.handoff_ready
    assert any("must match manifest project" in issue.message for issue in report.warnings)


def test_layer_map_id_name_pair_swap_blocks_handoff(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    output = manifest["output"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(output, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)

    layer_map_path = tmp_path / str(output["layer_map"])
    layer_map = yaml.safe_load(layer_map_path.read_text(encoding="utf-8"))
    layers = layer_map["layers"]
    layers[0]["name"], layers[1]["name"] = layers[1]["name"], layers[0]["name"]
    layer_map_path.write_text(yaml.safe_dump(layer_map), encoding="utf-8")

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert not report.handoff_ready
    assert any("pairs must exactly match" in issue.message for issue in report.warnings)


def test_duplicate_layer_map_row_blocks_handoff(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    output = manifest["output"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(output, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)

    layer_map_path = tmp_path / str(output["layer_map"])
    layer_map = yaml.safe_load(layer_map_path.read_text(encoding="utf-8"))
    layer_map["layers"].append(deepcopy(layer_map["layers"][0]))
    layer_map_path.write_text(yaml.safe_dump(layer_map), encoding="utf-8")

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert not report.handoff_ready
    assert any("present and unique" in issue.message for issue in report.warnings)


def test_required_part_excluded_from_import_blocks_handoff(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    parts[0]["include_in_import"] = False
    _materialize_handoff_files(manifest, tmp_path)

    report = validate_asset_manifest(manifest, base_dir=tmp_path)

    assert report.valid
    assert not report.handoff_ready
    assert any("required part is excluded" in issue.message for issue in report.warnings)


def test_redraw_asset_must_require_review() -> None:
    manifest = deepcopy(_sample_manifest())
    parts = manifest["parts"]
    assert isinstance(parts, list)
    parts[0]["generation_method"] = "redraw"
    parts[0]["review_required"] = False

    report = validate_asset_manifest(manifest)

    assert not report.valid
    assert any("redraw assets" in issue.message for issue in report.errors)


def test_psd_builder_returns_ordered_plan_without_creating_psd(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    target = tmp_path / "model_import.psd"
    plan = create_psd_build_plan(
        manifest,
        manifest_path=Path("examples/asset_manifest.sample.yaml"),
        output_psd=target,
    )

    layers = plan["layers"]
    assert isinstance(layers, list)
    assert [layer["order"] for layer in layers] == [10, 20, 30, 40, 50, 60]
    assert plan["status"] == "plan_only"
    assert plan["ready_to_build"] is False
    assert plan["can_build"] is False
    assert not target.exists()


def test_psd_backend_is_not_called_for_unready_plan(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    plan = create_psd_build_plan(
        manifest,
        manifest_path=Path("examples/asset_manifest.sample.yaml"),
        output_psd=tmp_path / "model_import.psd",
        base_dir=tmp_path,
    )

    class RecordingBackend:
        called = False

        def build(
            self,
            *,
            output_path: Path,
            canvas: Mapping[str, Any],
            layers: Sequence[Mapping[str, Any]],
        ) -> Path:
            del output_path, canvas, layers
            self.called = True
            return tmp_path / "model_import.psd"

    backend = RecordingBackend()

    try:
        build_psd(plan, backend)
    except RuntimeError as exc:
        assert "not ready" in str(exc)
    else:
        raise AssertionError("unready build plan must be rejected")
    assert not backend.called


def test_psd_builder_rejects_source_with_wrong_signature(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())
    constraints = manifest["import_constraints"]
    source_image = manifest["source_image"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(source_image, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)
    (tmp_path / str(source_image["path"])).write_bytes(b"not-a-png")

    plan = create_psd_build_plan(
        manifest,
        manifest_path=Path("examples/asset_manifest.sample.yaml"),
        base_dir=tmp_path,
    )

    assert plan["ready_to_build"] is False
    assert "source image is missing, empty, or has an invalid signature" in plan["build_blockers"]


def test_psd_builder_resolves_relative_output_from_base_dir(tmp_path: Path) -> None:
    manifest = deepcopy(_sample_manifest())

    plan = create_psd_build_plan(
        manifest,
        manifest_path=Path("examples/asset_manifest.sample.yaml"),
        base_dir=tmp_path,
    )

    assert Path(str(plan["output_psd"])) == tmp_path / "generated/model_import.psd"


def _ready_build_plan(manifest: dict[str, object], tmp_path: Path) -> dict[str, object]:
    constraints = manifest["import_constraints"]
    parts = manifest["parts"]
    assert isinstance(constraints, dict)
    assert isinstance(parts, list)
    for key in constraints:
        constraints[key] = True
    for part in parts:
        part["readiness"] = "approved"
    _materialize_handoff_files(manifest, tmp_path)
    return create_psd_build_plan(
        manifest,
        manifest_path=Path("examples/asset_manifest.sample.yaml"),
        base_dir=tmp_path,
    )


def test_psd_backend_must_return_expected_output_path(tmp_path: Path) -> None:
    plan = _ready_build_plan(deepcopy(_sample_manifest()), tmp_path)

    class WrongPathBackend:
        def build(
            self,
            *,
            output_path: Path,
            canvas: Mapping[str, Any],
            layers: Sequence[Mapping[str, Any]],
        ) -> Path:
            del output_path, canvas, layers
            wrong = tmp_path / "wrong.psd"
            wrong.write_bytes(b"8BPS-wrong-path")
            return wrong

    try:
        build_psd(plan, WrongPathBackend())
    except RuntimeError as exc:
        assert "different output path" in str(exc)
    else:
        raise AssertionError("backend must return the configured output path")


def test_psd_backend_output_must_have_psd_signature(tmp_path: Path) -> None:
    plan = _ready_build_plan(deepcopy(_sample_manifest()), tmp_path)

    class InvalidPsdBackend:
        def build(
            self,
            *,
            output_path: Path,
            canvas: Mapping[str, Any],
            layers: Sequence[Mapping[str, Any]],
        ) -> Path:
            del canvas, layers
            output_path.write_bytes(b"not-a-psd")
            return output_path

    try:
        build_psd(plan, InvalidPsdBackend())
    except RuntimeError as exc:
        assert "PSD signature" in str(exc)
    else:
        raise AssertionError("backend output must have a PSD signature")
