from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from tools.backends.segmentation.contracts import (
    BackendAvailability,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResult,
)
from tools.backends.segmentation.sam2 import Sam2SegmentationBackend


@dataclass(frozen=True)
class GroundingDetection:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("grounding bbox must be a positive xyxy box")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("grounding score must be between 0 and 1")
        if not self.label.strip():
            raise ValueError("grounding label must be non-empty")


class GroundingBackend(Protocol):
    backend_id: str

    def check_availability(self) -> BackendAvailability: ...

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        *,
        minimum_confidence: float,
    ) -> Sequence[GroundingDetection]: ...

    def provenance(self) -> dict[str, Any]: ...


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class GroundingDinoBackend:
    """Local-only Grounding DINO adapter using optional Transformers weights."""

    backend_id = "grounding-dino-local"

    def __init__(
        self,
        model_path: Path | None,
        *,
        model_revision: str | None = None,
        device: str = "cpu",
        text_threshold: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = device
        self.text_threshold = text_threshold
        self._processor: Any | None = None
        self._model: Any | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "model_revision": self.model_revision,
            "device": self.device,
            "local_files_only": True,
            "model_downloaded": False,
        }

    def check_availability(self) -> BackendAvailability:
        if self.model_path is None:
            return BackendAvailability(False, "Grounding DINO requires --grounding-model")
        if not self.model_path.exists():
            return BackendAvailability(
                False,
                f"local Grounding DINO model not found: {self.model_path}",
            )
        missing = [
            name for name in ("transformers", "torch") if not _module_available(name)
        ]
        if missing:
            return BackendAvailability(
                False,
                f"Grounding DINO optional dependencies are unavailable: {', '.join(missing)}",
            )
        return BackendAvailability(True)

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        assert self.model_path is not None
        transformers = importlib.import_module("transformers")
        processor_class = transformers.AutoProcessor
        model_class = transformers.AutoModelForZeroShotObjectDetection
        load_options = {
            "local_files_only": True,
            "revision": self.model_revision,
        }
        self._processor = processor_class.from_pretrained(
            str(self.model_path.resolve()),
            **load_options,
        )
        self._model = model_class.from_pretrained(
            str(self.model_path.resolve()),
            **load_options,
        ).to(self.device)
        self._model.eval()
        return self._processor, self._model

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        *,
        minimum_confidence: float,
    ) -> Sequence[GroundingDetection]:
        if not text_prompt.strip():
            raise ValueError("Grounding DINO requires a semantic prompt")
        availability = self.check_availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        processor, model = self._load()
        inputs = processor(
            images=image.convert("RGB"),
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)
        torch = importlib.import_module("torch")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [(image.height, image.width)]
        post_process = processor.post_process_grounded_object_detection
        parameters = inspect.signature(post_process).parameters
        threshold_options = (
            {"box_threshold": minimum_confidence}
            if "box_threshold" in parameters
            else {"threshold": minimum_confidence}
        )
        processed = post_process(
            outputs,
            inputs.input_ids,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
            **threshold_options,
        )[0]
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels = processed.get("text_labels", processed.get("labels", []))
        detections: list[GroundingDetection] = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            box_values = tuple(float(value) for value in box.tolist())
            if len(box_values) != 4:
                continue
            confidence = float(score.item() if hasattr(score, "item") else score)
            if confidence < minimum_confidence:
                continue
            detections.append(
                GroundingDetection(
                    bbox_xyxy=(
                        box_values[0],
                        box_values[1],
                        box_values[2],
                        box_values[3],
                    ),
                    score=confidence,
                    label=str(label),
                )
            )
        return detections


class GroundedSam2SegmentationBackend:
    """Compose replaceable text grounding with the SAM 2 adapter."""

    backend_id = "grounded-sam2"

    def __init__(
        self,
        grounding_backend: GroundingBackend,
        sam2_backend: Sam2SegmentationBackend,
    ) -> None:
        self.grounding_backend = grounding_backend
        self.sam2_backend = sam2_backend

    def check_availability(self) -> BackendAvailability:
        grounding = self.grounding_backend.check_availability()
        if not grounding.available:
            return grounding
        sam2 = self.sam2_backend.check_availability()
        if not sam2.available:
            return sam2
        return BackendAvailability(True)

    def segment(
        self,
        request: SegmentationRequest,
        *,
        execute: bool = False,
    ) -> SegmentationResult:
        provenance = {
            "grounding": self.grounding_backend.provenance(),
            "sam2": self.sam2_backend.config.provenance(),
            "prompt": request.prompt_provenance(),
            "model_downloaded": False,
        }
        if not request.semantic_prompt.strip():
            return SegmentationResult(
                status="failed",
                backend=self.backend_id,
                provenance=provenance,
                message="grounded SAM 2 requires semantic_prompt",
            )
        if not execute:
            return SegmentationResult(
                status="not_run",
                backend=self.backend_id,
                provenance=provenance,
                message="dry-run: grounding and SAM 2 models were not loaded",
            )
        availability = self.check_availability()
        if not availability.available:
            return SegmentationResult(
                status="unavailable",
                backend=self.backend_id,
                provenance=provenance,
                message=availability.reason,
            )
        try:
            detections = self.grounding_backend.detect(
                request.source_image,
                request.semantic_prompt,
                minimum_confidence=request.minimum_confidence,
            )
            candidates: list[SegmentationCandidate] = []
            for detection_index, detection in enumerate(detections):
                sam_request = SegmentationRequest(
                    request_id=f"{request.request_id}:grounding:{detection_index}",
                    layer_id=request.layer_id,
                    source_image=request.source_image,
                    semantic_prompt=request.semantic_prompt,
                    point_prompts=request.point_prompts,
                    box_prompt=detection.bbox_xyxy,
                    existing_mask=request.existing_mask,
                    candidate_count=request.candidate_count,
                    minimum_confidence=request.minimum_confidence,
                    side=request.side,
                    expected_region=request.expected_region,
                )
                sam_result = self.sam2_backend.segment(sam_request, execute=True)
                if sam_result.status != "completed":
                    return SegmentationResult(
                        status=sam_result.status,
                        backend=self.backend_id,
                        provenance=provenance,
                        message=sam_result.message,
                    )
                for candidate in sam_result.candidates:
                    combined_confidence = candidate.confidence * detection.score
                    if combined_confidence < request.minimum_confidence:
                        continue
                    candidates.append(
                        SegmentationCandidate(
                            candidate_id=(
                                f"{candidate.candidate_id}-grounding-{detection_index}"
                            ),
                            mask=candidate.mask,
                            confidence=combined_confidence,
                            stability_score=candidate.stability_score,
                            bbox_xyxy=candidate.bbox_xyxy,
                            label=detection.label,
                            metadata={
                                **candidate.metadata,
                                "grounding": {
                                    "bbox_xyxy": list(detection.bbox_xyxy),
                                    "score": detection.score,
                                    "label": detection.label,
                                    "backend": self.grounding_backend.backend_id,
                                },
                            },
                        )
                    )
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return SegmentationResult(
                status="failed",
                backend=self.backend_id,
                provenance=provenance,
                message=f"Grounded SAM 2 execution failed: {exc}",
            )
        return SegmentationResult(
            status="completed",
            backend=self.backend_id,
            candidates=tuple(candidates),
            provenance=provenance,
        )
