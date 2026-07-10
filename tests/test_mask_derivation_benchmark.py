from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest
import yaml
from PIL import Image

from tools.mask_derivation.algorithms import detect_candidate_conflicts
from tools.mask_derivation.pipeline import derive_masks


@pytest.mark.benchmark
def test_full_canvas_2048_island_detection_wall_clock() -> None:
    target = Image.new("L", (2048, 2048), 255)

    started = perf_counter()
    conflicts, _mask = detect_candidate_conflicts(
        target,
        {"protect": target},
        max_area_ratio=2.0,
        min_island_area_px=2,
    )
    elapsed = perf_counter() - started

    assert not any(conflict["type"] == "thin_isolated_region" for conflict in conflicts)
    assert elapsed < 3.0


@pytest.mark.benchmark
def test_dry_run_2048_retains_no_candidate_images(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    Image.new("RGBA", (2048, 2048), (40, 60, 80, 255)).save(source_path)
    target = Image.new("L", (2048, 2048), 0)
    target.paste(180, (128, 128, 1920, 1920))
    target.save(target_path)
    queue = {
        "schema_version": 3,
        "project": "benchmark",
        "source_image": {"path": source_path.name},
        "canvas": {"width": 2048, "height": 2048},
        "assets": [
            {
                "layer_id": "large_part",
                "role": "torso",
                "target_mask": target_path.name,
                "protect_mask": "canonical/protect.png",
                "edge_extension_mask": "canonical/edge.png",
                "inpaint_mask": "canonical/inpaint.png",
                "draw_order": 1,
            }
        ],
    }
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), "utf-8")

    started = perf_counter()
    _result, artifacts = derive_masks(
        queue,
        queue_path=queue_path,
        base_dir=tmp_path,
        output_dir=tmp_path / "derived",
        retain_artifacts=False,
    )
    elapsed = perf_counter() - started

    assert artifacts.png_bytes == {}
    assert elapsed < 10.0
