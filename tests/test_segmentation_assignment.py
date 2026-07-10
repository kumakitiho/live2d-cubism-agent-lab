from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import yaml

from tools.backends.segmentation.integrity import canonical_mapping_sha256
from tools.segmentation_assignment_planner import (
    QUEUE_UPDATE_FIELDS,
    apply_assignment_plan,
    build_assignment_plan,
    render_queue_with_selected_updates,
)


def _asset(layer_id: str, side: str) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        "role": "eye_white",
        "side": side,
        "target_mask": f"canonical/{layer_id}.target.png",
        "protect_mask": f"canonical/{layer_id}.protect.png",
        "edge_extension_mask": f"canonical/{layer_id}.edge.png",
        "inpaint_mask": f"canonical/{layer_id}.inpaint.png",
        "unrelated": {"must": "stay-byte-equivalent"},
    }


def _queue() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "project": "assignment-test",
        "assets": [_asset("eye_L", "L"), _asset("eye_R", "R")],
        "jobs": [{"id": "eyes", "status": "planned"}],
    }


def _ranked_candidate(layer_id: str, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "layer_id": layer_id,
        "rank": 1,
        "soft_mask_file": f"segmentation/{candidate_id}.soft.png",
        "confidence": 0.9,
        "source_backend": "mock",
        "model_id": "mock-fixture-v1",
        "model_revision": "1",
        "requires_review": False,
        "rejection_reasons": [],
    }


def _ranked() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ranked",
        "project": "assignment-test",
        "run_id": "run-123",
        "asset_generation_queue": "queue.yaml",
        "asset_generation_queue_sha256": "a" * 64,
        "asset_generation_queue_content_sha256": canonical_mapping_sha256(_queue()),
        "source_image_sha256": "b" * 64,
        "candidates": [
            _ranked_candidate("eye_L", "candidate-L"),
            _ranked_candidate("eye_R", "candidate-R"),
        ],
    }


def test_assignment_plan_never_mutates_queue_or_auto_approves() -> None:
    queue = _queue()
    before = deepcopy(queue)

    plan = build_assignment_plan(queue, _ranked(), queue_ref="queue.yaml")

    assert queue == before
    assert plan["review_status"] == "pending"
    assert plan["summary"]["canonical_queue_modified"] is False
    assert all(assignment["status"] == "needs_review" for assignment in plan["assignments"])
    assert all(assignment["requires_review"] is True for assignment in plan["assignments"])
    assert all(
        assignment["edge_extension_mask"] != assignment["inpaint_mask"]
        for assignment in plan["assignments"]
    )


def test_unreviewed_assignment_plan_cannot_be_applied() -> None:
    plan = build_assignment_plan(_queue(), _ranked(), queue_ref="queue.yaml")

    with pytest.raises(ValueError, match="review_status: approved"):
        apply_assignment_plan(_queue(), plan)


def test_assignment_plan_cannot_be_applied_to_stale_queue_content() -> None:
    queue = _queue()
    plan = build_assignment_plan(queue, _ranked(), queue_ref="queue.yaml")
    plan["review_status"] = "approved"
    plan["assignments"][0]["status"] = "approved"
    plan["assignments"][0]["requires_review"] = False
    stale_queue = deepcopy(queue)
    stale_queue["assets"][0]["unrelated"] = {"changed": True}

    with pytest.raises(ValueError, match="different queue content"):
        apply_assignment_plan(stale_queue, plan)


def test_only_approved_part_is_updated_and_unselected_part_is_equivalent() -> None:
    queue = _queue()
    original = deepcopy(queue)
    plan = build_assignment_plan(queue, _ranked(), queue_ref="queue.yaml")
    plan["review_status"] = "approved"
    plan["assignments"][0]["status"] = "approved"
    plan["assignments"][0]["requires_review"] = False

    updated = apply_assignment_plan(queue, plan)

    assert queue == original
    assert updated["assets"][1] == original["assets"][1]
    assert updated["jobs"] == original["jobs"]
    changed_fields = {
        key
        for key in updated["assets"][0]
        if updated["assets"][0].get(key) != original["assets"][0].get(key)
    }
    assert changed_fields <= set(QUEUE_UPDATE_FIELDS)
    assert updated["assets"][0]["target_mask"] == "segmentation/candidate-L.soft.png"
    assert updated["assets"][0]["protect_mask"] == original["assets"][0]["protect_mask"]
    assert updated["assets"][0]["edge_extension_mask"] != updated["assets"][0]["inpaint_mask"]
    assert updated["assets"][0]["segmentation_run_id"] == "run-123"


def test_duplicate_selected_candidate_is_rejected() -> None:
    plan = build_assignment_plan(_queue(), _ranked(), queue_ref="queue.yaml")
    plan["review_status"] = "approved"
    for assignment in plan["assignments"]:
        assignment["status"] = "approved"
        assignment["requires_review"] = False
    plan["assignments"][1]["selected_candidate_id"] = "candidate-L"

    with pytest.raises(ValueError, match="duplicate selected candidate ID"):
        apply_assignment_plan(_queue(), plan)


def test_approved_assignment_keeps_edge_extension_separate_from_inpaint() -> None:
    plan = build_assignment_plan(_queue(), _ranked(), queue_ref="queue.yaml")
    plan["review_status"] = "approved"
    assignment = plan["assignments"][0]
    assignment["status"] = "approved"
    assignment["requires_review"] = False
    assignment["edge_extension_mask"] = assignment["inpaint_mask"]

    with pytest.raises(ValueError, match="edge_extension_mask must differ"):
        apply_assignment_plan(_queue(), plan)


def test_selected_only_queue_render_preserves_unselected_part_bytes() -> None:
    queue_text = """schema_version: 3
project: \"assignment-test\"
assets:
  - layer_id: eye_L
    role: eye_white
    side: L
    target_mask: canonical/eye_L.target.png
    protect_mask: canonical/eye_L.protect.png
    edge_extension_mask: canonical/eye_L.edge.png
    inpaint_mask: canonical/eye_L.inpaint.png
  - layer_id: eye_R  # keep this exact comment
    role: eye_white
    side: R
    target_mask: 'canonical/eye_R.target.png'
    protect_mask: canonical/eye_R.protect.png
    edge_extension_mask: canonical/eye_R.edge.png
    inpaint_mask: canonical/eye_R.inpaint.png
    unrelated: {must: \"stay-byte-equivalent\"} # keep flow style
jobs:
  - {id: eyes, status: planned}
"""
    queue: dict[str, Any] = yaml.safe_load(queue_text)
    updated = deepcopy(queue)
    updated["assets"][0]["target_mask"] = "segmentation/approved.soft.png"
    updated["assets"][0]["segmentation_run_id"] = "run-123"
    unselected_before = queue_text[
        queue_text.index("  - layer_id: eye_R") : queue_text.index("jobs:")
    ]

    rendered = render_queue_with_selected_updates(
        queue_text,
        updated,
        selected_layer_ids={"eye_L"},
    )

    unselected_after = rendered[rendered.index("  - layer_id: eye_R") : rendered.index("jobs:")]
    assert unselected_after == unselected_before
    assert yaml.safe_load(rendered) == updated


def test_selected_only_queue_render_rejects_flow_style_assets() -> None:
    queue_text = "assets: [{layer_id: eye_L, target_mask: old.png}]\n"
    updated = {"assets": [{"layer_id": "eye_L", "target_mask": "new.png"}]}

    with pytest.raises(ValueError, match="block-style assets"):
        render_queue_with_selected_updates(
            queue_text,
            updated,
            selected_layer_ids={"eye_L"},
        )
