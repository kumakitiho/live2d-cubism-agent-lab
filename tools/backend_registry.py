from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BackendKind = Literal["segmentation", "inpainting"]


@dataclass(frozen=True)
class BackendAvailabilityInfo:
    """Dependency-light availability result for either backend family."""

    kind: BackendKind
    name: str
    available: bool
    reason: str

    @property
    def status(self) -> Literal["available", "unavailable"]:
        return "available" if self.available else "unavailable"

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
        }


def _normalized(name: str) -> str:
    value = name.strip().lower().replace("-", "_")
    aliases = {"grounded": "grounded_sam2", "flux": "flux_fill"}
    return aliases.get(value, value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ValueError("backend path options must be non-empty strings or Path values")


class BackendRegistry:
    """Lazy common registry; resolving an adapter never loads a model."""

    segmentation_names = ("mock", "sam2", "grounded_sam2")
    inpainting_names = ("mock", "diffusers", "flux_fill")

    def get_segmentation(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        normalized = _normalized(name)
        options = dict(config or {})
        if normalized == "mock":
            from tools.backends.segmentation.mock import MockSegmentationBackend

            return MockSegmentationBackend()
        if normalized == "sam2":
            from tools.backends.segmentation.sam2 import Sam2Config, Sam2SegmentationBackend

            return Sam2SegmentationBackend(
                Sam2Config(
                    model_id=options.get("model_id"),
                    model_revision=options.get("model_revision"),
                    checkpoint=_optional_path(options.get("checkpoint")),
                    model_config=options.get("model_config"),
                    device=str(options.get("device", "cpu")),
                )
            )
        if normalized == "grounded_sam2":
            from tools.backends.segmentation.grounded_sam2 import (
                GroundedSam2SegmentationBackend,
                GroundingDinoBackend,
            )
            from tools.backends.segmentation.sam2 import Sam2Config, Sam2SegmentationBackend

            sam2 = Sam2SegmentationBackend(
                Sam2Config(
                    model_id=options.get("model_id"),
                    model_revision=options.get("model_revision"),
                    checkpoint=_optional_path(options.get("checkpoint")),
                    model_config=options.get("model_config"),
                    device=str(options.get("device", "cpu")),
                )
            )
            grounding = GroundingDinoBackend(
                _optional_path(options.get("grounding_model")),
                model_revision=options.get("grounding_model_revision"),
                device=str(options.get("device", "cpu")),
            )
            return GroundedSam2SegmentationBackend(grounding, sam2)
        raise ValueError(f"unknown segmentation backend: {name}")

    def get_inpainting(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        del config
        normalized = _normalized(name)
        if normalized == "mock":
            from tools.backends.inpainting.mock import MockInpaintingBackend

            return MockInpaintingBackend()
        if normalized == "diffusers":
            from tools.backends.inpainting.diffusers_backend import DiffusersInpaintingBackend

            return DiffusersInpaintingBackend()
        if normalized == "flux_fill":
            from tools.backends.inpainting.flux_fill import FluxFillInpaintingBackend

            return FluxFillInpaintingBackend()
        raise ValueError(f"unknown inpainting backend: {name}")

    def availability(
        self,
        kind: BackendKind,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> BackendAvailabilityInfo:
        normalized = _normalized(name)
        try:
            if kind == "segmentation":
                backend = self.get_segmentation(normalized, config)
                status = backend.check_availability()
                available = bool(status.available)
                reason = status.reason or "backend is available"
            elif kind == "inpainting":
                backend = self.get_inpainting(normalized, config)
                status = backend.status()
                available = bool(status.available)
                reason = str(status.detail)
            else:
                raise ValueError(f"unknown backend kind: {kind}")
        except (ImportError, ModuleNotFoundError) as exc:
            available = False
            reason = f"optional dependency unavailable: {exc}"
        return BackendAvailabilityInfo(kind, normalized, available, reason)

    def list_availability(
        self,
        kind: BackendKind,
    ) -> list[BackendAvailabilityInfo]:
        names = self.segmentation_names if kind == "segmentation" else self.inpainting_names
        return [self.availability(kind, name) for name in names]


def release_backend(backend: object) -> None:
    """Release a model if an adapter exposes the optional release hook."""

    release = getattr(backend, "release", None)
    if callable(release):
        release()


registry = BackendRegistry()


__all__ = [
    "BackendAvailabilityInfo",
    "BackendKind",
    "BackendRegistry",
    "registry",
    "release_backend",
]
