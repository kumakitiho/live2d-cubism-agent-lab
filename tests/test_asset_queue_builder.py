from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.artifact_validation import load_yaml_mapping
from tools.asset_queue_builder import derive_asset_manifest, derive_layer_map, main


def _queue() -> dict[str, Any]:
    return load_yaml_mapping(Path("examples/asset_generation_queue.sample.yaml"))


def test_checked_in_manifest_and_layer_map_are_exact_queue_derivatives() -> None:
    queue = _queue()
    queue_ref = "examples/asset_generation_queue.sample.yaml"

    assert derive_asset_manifest(queue, queue_ref=queue_ref) == load_yaml_mapping(
        Path("examples/asset_manifest.sample.yaml")
    )
    assert derive_layer_map(queue, queue_ref=queue_ref) == load_yaml_mapping(
        Path("examples/layer_map.sample.yaml")
    )


def test_queue_asset_change_updates_both_derivatives() -> None:
    queue = deepcopy(_queue())
    queue["assets"][0]["layer_name"] = "eye_sclera_L"

    manifest = derive_asset_manifest(queue)
    layer_map = derive_layer_map(queue)

    assert manifest["parts"][0]["layer_name"] == "eye_sclera_L"
    assert layer_map["layers"][0]["name"] == "eye_sclera_L"


def test_builder_writes_declared_derivatives_from_queue(tmp_path: Path) -> None:
    queue = deepcopy(_queue())
    queue["feedback_inputs"] = []
    queue["jobs"][-1]["feedback_refs"] = []
    queue["derivatives"]["asset_manifest"] = "derived/asset_manifest.yaml"
    queue["derivatives"]["layer_map"] = "derived/layer_map.yaml"
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--execute",
        ]
    )

    assert exit_code == 0
    manifest = load_yaml_mapping(tmp_path / "derived/asset_manifest.yaml")
    layer_map = load_yaml_mapping(tmp_path / "derived/layer_map.yaml")
    assert manifest == derive_asset_manifest(queue, queue_ref=queue_path.as_posix())
    assert layer_map == derive_layer_map(queue, queue_ref=queue_path.as_posix())


def test_builder_rejects_output_outside_base_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "workspace"
    base_dir.mkdir()
    queue = deepcopy(_queue())
    queue["feedback_inputs"] = []
    queue["jobs"][-1]["feedback_refs"] = []
    queue["derivatives"]["asset_manifest"] = "../outside.yaml"
    queue["derivatives"]["layer_map"] = "derived/layer_map.yaml"
    queue_path = base_dir / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main([str(queue_path), "--base-dir", str(base_dir), "--execute"])

    assert exit_code == 2
    assert not (tmp_path / "outside.yaml").exists()


def test_builder_preflights_both_outputs_before_writing(tmp_path: Path) -> None:
    queue = deepcopy(_queue())
    queue["feedback_inputs"] = []
    queue["jobs"][-1]["feedback_refs"] = []
    queue["derivatives"]["asset_manifest"] = "derived/asset_manifest.yaml"
    queue["derivatives"]["layer_map"] = "derived/layer_map.yaml"
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    layer_map_path = tmp_path / "derived/layer_map.yaml"
    layer_map_path.parent.mkdir(parents=True)
    layer_map_path.write_text("existing", encoding="utf-8")

    exit_code = main([str(queue_path), "--base-dir", str(tmp_path), "--execute"])

    assert exit_code == 2
    assert not (tmp_path / "derived/asset_manifest.yaml").exists()
    assert layer_map_path.read_text(encoding="utf-8") == "existing"


def test_builder_rolls_back_both_outputs_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = deepcopy(_queue())
    queue["feedback_inputs"] = []
    queue["jobs"][-1]["feedback_refs"] = []
    queue["derivatives"]["asset_manifest"] = "derived/asset_manifest.yaml"
    queue["derivatives"]["layer_map"] = "derived/layer_map.yaml"
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(queue, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "derived/asset_manifest.yaml"
    layer_map_path = tmp_path / "derived/layer_map.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("old-manifest", encoding="utf-8")
    layer_map_path.write_text("old-layer-map", encoding="utf-8")

    original_replace = Path.replace

    def fail_layer_map_temp_replace(source: Path, target: Path) -> Path:
        if source.name.endswith(".tmp") and target == layer_map_path:
            raise OSError("simulated replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_layer_map_temp_replace)

    exit_code = main(
        [
            str(queue_path),
            "--base-dir",
            str(tmp_path),
            "--execute",
            "--force",
        ]
    )

    assert exit_code == 2
    assert manifest_path.read_text(encoding="utf-8") == "old-manifest"
    assert layer_map_path.read_text(encoding="utf-8") == "old-layer-map"
