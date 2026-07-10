from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from PIL import Image

SegmentationStatus = Literal["completed", "unavailable", "not_run", "failed"]


@dataclass(frozen=True)
class PointPrompt:
    """A SAM-style point prompt in source-canvas coordinates."""

    x: float
    y: float
    label: int = 1

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("point prompt coordinates must be non-negative")
        if self.label not in {0, 1}:
            raise ValueError("point prompt label must be 0 or 1")

    def as_dict(self) -> dict[str, float | int]:
        return {"x": self.x, "y": self.y, "label": self.label}


@dataclass
class SegmentationRequest:
    """Backend-neutral request. Images are already aligned to the canonical canvas."""

    request_id: str
    layer_id: str
    source_image: Image.Image
    semantic_prompt: str = ""
    point_prompts: tuple[PointPrompt, ...] = ()
    box_prompt: tuple[float, float, float, float] | None = None
    existing_mask: Image.Image | None = None
    candidate_count: int = 3
    minimum_confidence: float = 0.0
    side: str = "none"
    expected_region: Mapping[str, float] | None = None
    fixture_masks: tuple[Image.Image, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.layer_id.strip():
            raise ValueError("request_id and layer_id must be non-empty")
        if self.source_image.width <= 0 or self.source_image.height <= 0:
            raise ValueError("source image canvas must be non-empty")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.side not in {"L", "R", "C", "none"}:
            raise ValueError("side must be L, R, C, or none")
        if self.box_prompt is not None:
            x1, y1, x2, y2 = self.box_prompt
            if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError("box_prompt must be a positive xyxy box")
            if x2 > self.source_image.width or y2 > self.source_image.height:
                raise ValueError("box_prompt must stay inside the source canvas")
        for point in self.point_prompts:
            if point.x >= self.source_image.width or point.y >= self.source_image.height:
                raise ValueError("point prompt must stay inside the source canvas")
        for field_name, image in (
            ("existing_mask", self.existing_mask),
            *(('fixture_masks', image) for image in self.fixture_masks),
        ):
            if image is not None and image.size != self.source_image.size:
                raise ValueError(f"{field_name} canvas mismatch")

    @property
    def canvas(self) -> tuple[int, int]:
        return self.source_image.size

    def prompt_provenance(self) -> dict[str, Any]:
        return {
            "semantic_prompt": self.semantic_prompt,
            "point_prompts": [point.as_dict() for point in self.point_prompts],
            "box_prompt": list(self.box_prompt) if self.box_prompt is not None else None,
            "existing_mask": self.existing_mask is not None,
            "candidate_count": self.candidate_count,
            "minimum_confidence": self.minimum_confidence,
        }


@dataclass
class SegmentationCandidate:
    candidate_id: str
    mask: Image.Image
    confidence: float
    stability_score: float
    bbox_xyxy: tuple[int, int, int, int]
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")
        if not 0.0 <= self.stability_score <= 1.0:
            raise ValueError("candidate stability_score must be between 0 and 1")
        x1, y1, x2, y2 = self.bbox_xyxy
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("candidate bbox_xyxy must be a positive xyxy box")
        if x2 > self.mask.width or y2 > self.mask.height:
            raise ValueError("candidate bbox_xyxy must stay inside the mask canvas")


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None


@dataclass
class SegmentationResult:
    status: SegmentationStatus
    backend: str
    candidates: tuple[SegmentationCandidate, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status != "completed" and self.candidates:
            raise ValueError("only completed results may contain candidates")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("segmentation result contains duplicate candidate IDs")


@runtime_checkable
class SegmentationBackend(Protocol):
    backend_id: str

    def check_availability(self) -> BackendAvailability: ...

    def segment(
        self,
        request: SegmentationRequest,
        *,
        execute: bool = False,
    ) -> SegmentationResult: ...
