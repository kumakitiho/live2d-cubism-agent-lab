from __future__ import annotations

import hashlib
import importlib
import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageFilter

from tools.backends.segmentation.contracts import (
    BackendAvailability,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResult,
)


@dataclass(frozen=True)
class Sam2Config:
    model_id: str | None = None
    model_revision: str | None = None
    checkpoint: Path | None = None
    model_config: str | None = None
    device: str = "cpu"

    def provenance(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "model_config": self.model_config,
            "device": self.device,
        }


class Sam2Runtime(Protocol):
    def segment(self, request: SegmentationRequest) -> Sequence[SegmentationCandidate]: ...


Sam2RuntimeFactory = Callable[[Sam2Config], Sam2Runtime]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _resolve_model_source(
    config: Sam2Config,
) -> tuple[tuple[str, Path] | None, str | None]:
    if config.checkpoint is not None:
        if not config.checkpoint.is_file():
            return None, f"SAM 2 checkpoint not found: {config.checkpoint}"
        if config.model_config is None or not config.model_config.strip():
            return (
                None,
                "SAM 2 checkpoint execution requires a packaged Hydra --model-config name",
            )
        return (config.model_config, config.checkpoint.resolve()), None
    if config.model_id:
        if not _module_available("sam2") or not _module_available("huggingface_hub"):
            return (
                None,
                "SAM 2 model-ID resolution requires installed sam2 and huggingface_hub; "
                "automatic model download is disabled",
            )
        try:
            build_module = importlib.import_module("sam2.build_sam")
            hub = importlib.import_module("huggingface_hub")
            config_name, checkpoint_name = build_module.HF_MODEL_ID_TO_FILENAMES[
                config.model_id
            ]
            cached_checkpoint = hub.hf_hub_download(
                repo_id=config.model_id,
                filename=checkpoint_name,
                revision=config.model_revision,
                local_files_only=True,
            )
        except Exception as exc:
            return (
                None,
                "SAM 2 model ID/checkpoint is unsupported or not cached locally; "
                f"automatic model download is disabled: {exc}",
            )
        return (str(config_name), Path(str(cached_checkpoint)).resolve()), None
    return None, "SAM 2 requires --checkpoint or a locally cached --model-id"


def _candidate_id(request: SegmentationRequest, config: Sam2Config, index: int) -> str:
    digest = hashlib.sha256(
        (
            f"{request.request_id}:{request.layer_id}:{config.model_id}:"
            f"{config.model_revision}:{config.checkpoint}:{index}"
        ).encode()
    ).hexdigest()[:12]
    return f"{request.layer_id}-{digest}"


class _LocalSam2Runtime:
    """Thin adapter over a locally installed Meta SAM 2 checkout."""

    def __init__(self, config: Sam2Config) -> None:
        self.config = config
        self._model: Any | None = None
        self._predictor: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._predictor is not None:
            return self._model, self._predictor
        model_source, reason = _resolve_model_source(self.config)
        if model_source is None:
            raise RuntimeError(reason)
        config_name, checkpoint = model_source
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        predictor_class = predictor_module.SAM2ImagePredictor
        build_module = importlib.import_module("sam2.build_sam")
        self._model = build_module.build_sam2(
            config_name,
            str(checkpoint),
            device=self.config.device,
        )
        self._predictor = predictor_class(self._model)
        return self._model, self._predictor

    @staticmethod
    def _soft_mask_from_logits(logits: Any, canvas: tuple[int, int]) -> Image.Image:
        numpy = importlib.import_module("numpy")
        values = numpy.asarray(logits, dtype="float32")
        values = numpy.squeeze(values)
        probability = 1.0 / (1.0 + numpy.exp(-numpy.clip(values, -20.0, 20.0)))
        pixels = numpy.rint(probability * 255.0).astype("uint8")
        return Image.fromarray(pixels, mode="L").resize(canvas, Image.Resampling.BILINEAR)

    @staticmethod
    def _soft_mask_from_segmentation(segmentation: Any, canvas: tuple[int, int]) -> Image.Image:
        numpy = importlib.import_module("numpy")
        pixels = numpy.asarray(segmentation, dtype="uint8") * 255
        mask = Image.fromarray(pixels, mode="L")
        if mask.size != canvas:
            mask = mask.resize(canvas, Image.Resampling.NEAREST)
        return mask.filter(ImageFilter.GaussianBlur(radius=0.5))

    @staticmethod
    def _mask_input(existing_mask: Image.Image) -> Any:
        numpy = importlib.import_module("numpy")
        resized = existing_mask.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        probability = numpy.asarray(resized, dtype="float32") / 255.0
        probability = numpy.clip(probability, 1e-4, 1.0 - 1e-4)
        logits = numpy.log(probability / (1.0 - probability))
        return logits[None, :, :]

    def _automatic_candidates(
        self,
        request: SegmentationRequest,
        model: Any,
    ) -> list[SegmentationCandidate]:
        numpy = importlib.import_module("numpy")
        generator_module = importlib.import_module("sam2.automatic_mask_generator")
        generator_class = generator_module.SAM2AutomaticMaskGenerator
        generator = generator_class(model)
        raw_candidates = generator.generate(numpy.asarray(request.source_image.convert("RGB")))
        ranked = sorted(
            raw_candidates,
            key=lambda item: float(item.get("predicted_iou", 0.0)),
            reverse=True,
        )
        candidates: list[SegmentationCandidate] = []
        for index, raw in enumerate(ranked[: request.candidate_count]):
            confidence = float(raw.get("predicted_iou", 0.0))
            if confidence < request.minimum_confidence:
                continue
            mask = self._soft_mask_from_segmentation(raw["segmentation"], request.canvas)
            bbox = mask.getbbox()
            if bbox is None:
                continue
            candidates.append(
                SegmentationCandidate(
                    candidate_id=_candidate_id(request, self.config, index),
                    mask=mask,
                    confidence=max(0.0, min(1.0, confidence)),
                    stability_score=max(
                        0.0,
                        min(1.0, float(raw.get("stability_score", confidence))),
                    ),
                    bbox_xyxy=bbox,
                    label=request.semantic_prompt,
                    metadata={
                        "mode": "automatic_mask_generation",
                        "prompt_provenance": request.prompt_provenance(),
                    },
                )
            )
        return candidates

    def _prompted_candidates(
        self,
        request: SegmentationRequest,
        predictor: Any,
    ) -> list[SegmentationCandidate]:
        numpy = importlib.import_module("numpy")
        points = (
            numpy.asarray([[point.x, point.y] for point in request.point_prompts])
            if request.point_prompts
            else None
        )
        labels = (
            numpy.asarray([point.label for point in request.point_prompts])
            if request.point_prompts
            else None
        )
        box = numpy.asarray(request.box_prompt) if request.box_prompt is not None else None
        mask_input = (
            self._mask_input(request.existing_mask) if request.existing_mask is not None else None
        )
        _masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=request.candidate_count > 1,
        )
        scored = sorted(
            enumerate(zip(scores, logits, strict=True)),
            key=lambda item: float(item[1][0]),
            reverse=True,
        )
        candidates: list[SegmentationCandidate] = []
        for output_index, (confidence_raw, candidate_logits) in scored[: request.candidate_count]:
            confidence = float(confidence_raw)
            if confidence < request.minimum_confidence:
                continue
            mask = self._soft_mask_from_logits(candidate_logits, request.canvas)
            bbox = mask.getbbox()
            if bbox is None:
                continue
            candidates.append(
                SegmentationCandidate(
                    candidate_id=_candidate_id(request, self.config, output_index),
                    mask=mask,
                    confidence=max(0.0, min(1.0, confidence)),
                    stability_score=max(0.0, min(1.0, confidence)),
                    bbox_xyxy=bbox,
                    label=request.semantic_prompt,
                    metadata={
                        "mode": "prompted_or_refined",
                        "prompt_provenance": request.prompt_provenance(),
                    },
                )
            )
        return candidates

    def segment(self, request: SegmentationRequest) -> Sequence[SegmentationCandidate]:
        model, predictor = self._load()
        predictor.set_image(request.source_image.convert("RGB"))
        has_prompt = bool(
            request.point_prompts
            or request.box_prompt is not None
            or request.existing_mask is not None
        )
        if has_prompt:
            return self._prompted_candidates(request, predictor)
        return self._automatic_candidates(request, model)


class Sam2SegmentationBackend:
    backend_id = "sam2"

    def __init__(
        self,
        config: Sam2Config,
        *,
        runtime_factory: Sam2RuntimeFactory | None = None,
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory

    def check_availability(self) -> BackendAvailability:
        model_source, source_reason = _resolve_model_source(self.config)
        if model_source is None:
            return BackendAvailability(False, source_reason)
        if self._runtime_factory is not None:
            return BackendAvailability(True)
        missing = [name for name in ("sam2", "numpy") if not _module_available(name)]
        if missing:
            return BackendAvailability(
                False,
                f"SAM 2 optional dependencies are unavailable: {', '.join(missing)}",
            )
        return BackendAvailability(True)

    def segment(
        self,
        request: SegmentationRequest,
        *,
        execute: bool = False,
    ) -> SegmentationResult:
        provenance: dict[str, Any] = {
            **self.config.provenance(),
            "prompt": request.prompt_provenance(),
            "model_downloaded": False,
        }
        if not execute:
            return SegmentationResult(
                status="not_run",
                backend=self.backend_id,
                provenance=provenance,
                message="dry-run: SAM 2 model was not loaded",
            )
        try:
            availability = self.check_availability()
        except Exception as exc:
            return SegmentationResult(
                status="unavailable",
                backend=self.backend_id,
                provenance=provenance,
                message=f"SAM 2 availability check failed: {exc}",
            )
        if not availability.available:
            return SegmentationResult(
                status="unavailable",
                backend=self.backend_id,
                provenance=provenance,
                message=availability.reason,
            )
        try:
            runtime = (
                self._runtime_factory(self.config)
                if self._runtime_factory is not None
                else _LocalSam2Runtime(self.config)
            )
            candidates = tuple(runtime.segment(request))
        except (
            AttributeError,
            ImportError,
            KeyError,
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
                message=f"SAM 2 execution failed: {exc}",
            )
        return SegmentationResult(
            status="completed",
            backend=self.backend_id,
            candidates=candidates,
            provenance=provenance,
        )
