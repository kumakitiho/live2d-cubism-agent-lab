"""Optional segmentation backends and their dependency-light contracts."""

from tools.backends.segmentation.contracts import (
    BackendAvailability,
    PointPrompt,
    SegmentationBackend,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResult,
)
from tools.backends.segmentation.grounded_sam2 import (
    GroundedSam2SegmentationBackend,
    GroundingDetection,
    GroundingDinoBackend,
)
from tools.backends.segmentation.mock import MockSegmentationBackend
from tools.backends.segmentation.sam2 import Sam2Config, Sam2SegmentationBackend

__all__ = [
    "BackendAvailability",
    "GroundedSam2SegmentationBackend",
    "GroundingDetection",
    "GroundingDinoBackend",
    "MockSegmentationBackend",
    "PointPrompt",
    "Sam2Config",
    "Sam2SegmentationBackend",
    "SegmentationBackend",
    "SegmentationCandidate",
    "SegmentationRequest",
    "SegmentationResult",
]
