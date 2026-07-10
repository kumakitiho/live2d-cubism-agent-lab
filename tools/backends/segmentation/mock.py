from __future__ import annotations

import hashlib
from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageFilter

from tools.backends.segmentation.contracts import (
    BackendAvailability,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResult,
)


def _request_digest(request: SegmentationRequest) -> str:
    digest = hashlib.sha256()
    digest.update(request.request_id.encode("utf-8"))
    digest.update(request.layer_id.encode("utf-8"))
    digest.update(request.semantic_prompt.encode("utf-8"))
    digest.update(repr(request.point_prompts).encode("ascii"))
    digest.update(repr(request.box_prompt).encode("ascii"))
    digest.update(request.source_image.convert("RGBA").tobytes())
    for fixture in request.fixture_masks:
        digest.update(fixture.convert("L").tobytes())
    return digest.hexdigest()


def _clamped_box(
    box: tuple[float, float, float, float],
    canvas: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = canvas
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1)))
    top = max(0, min(height - 1, round(y1)))
    right = max(left + 1, min(width, round(x2)))
    bottom = max(top + 1, min(height, round(y2)))
    return left, top, right, bottom


def _default_mask(
    request: SegmentationRequest,
    *,
    index: int,
    digest: str,
) -> Image.Image:
    width, height = request.canvas
    if request.box_prompt is not None:
        base_box = _clamped_box(request.box_prompt, request.canvas)
    else:
        span_x = max(2, width // 3)
        span_y = max(2, height // 3)
        if request.side == "L":
            center_x = width * 0.3
        elif request.side == "R":
            center_x = width * 0.7
        else:
            center_x = width * 0.5
        center_y = height * (0.35 + (int(digest[0:2], 16) / 255.0) * 0.3)
        base_box = _clamped_box(
            (
                center_x - span_x / 2,
                center_y - span_y / 2,
                center_x + span_x / 2,
                center_y + span_y / 2,
            ),
            request.canvas,
        )
    offset = index - (request.candidate_count - 1) / 2
    shift_x = round(offset * max(1, width * 0.025))
    shift_y = round(offset * max(1, height * 0.015))
    x1, y1, x2, y2 = base_box
    shifted = _clamped_box((x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y), request.canvas)
    mask = Image.new("L", request.canvas, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(shifted, fill=220)
    for point in request.point_prompts:
        radius = max(1, min(width, height) // 24)
        point_box = (
            round(point.x) - radius,
            round(point.y) - radius,
            round(point.x) + radius,
            round(point.y) + radius,
        )
        draw.ellipse(point_box, fill=255 if point.label == 1 else 0)
    blur_radius = max(0.5, min(width, height) / 128)
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _candidate_masks(
    request: SegmentationRequest,
    digest: str,
) -> Iterable[tuple[int, Image.Image]]:
    if request.fixture_masks:
        for index, fixture in enumerate(request.fixture_masks[: request.candidate_count]):
            yield index, fixture.convert("L").copy()
        return
    for index in range(request.candidate_count):
        yield index, _default_mask(request, index=index, digest=digest)


class MockSegmentationBackend:
    """Deterministic CPU-only backend for CI and contract tests."""

    backend_id = "mock"

    def check_availability(self) -> BackendAvailability:
        return BackendAvailability(True)

    def release(self) -> None:
        """Mock backend has no retained model resources."""

    def segment(
        self,
        request: SegmentationRequest,
        *,
        execute: bool = False,
    ) -> SegmentationResult:
        provenance = {
            "model_id": "mock-fixture-v1",
            "model_revision": "1",
            "device": "cpu",
            "prompt": request.prompt_provenance(),
            "model_downloaded": False,
        }
        if not execute:
            return SegmentationResult(
                status="not_run",
                backend=self.backend_id,
                provenance=provenance,
                message="dry-run: mock segmentation was not executed",
            )

        digest = _request_digest(request)
        candidates: list[SegmentationCandidate] = []
        for index, mask in _candidate_masks(request, digest):
            confidence = max(0.0, min(1.0, 0.96 - index * 0.08))
            if confidence < request.minimum_confidence:
                continue
            bbox = mask.getbbox()
            if bbox is None:
                continue
            candidate_digest = hashlib.sha256(f"{digest}:{index}".encode("ascii")).hexdigest()[:12]
            candidates.append(
                SegmentationCandidate(
                    candidate_id=f"{request.layer_id}-{candidate_digest}",
                    mask=mask,
                    confidence=confidence,
                    stability_score=max(0.0, 0.94 - index * 0.05),
                    bbox_xyxy=bbox,
                    label=request.semantic_prompt,
                    metadata={
                        "fixture_index": index,
                        "prompt_provenance": request.prompt_provenance(),
                    },
                )
            )
        return SegmentationResult(
            status="completed",
            backend=self.backend_id,
            candidates=tuple(candidates),
            provenance=provenance,
        )
