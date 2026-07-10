from __future__ import annotations

from tools.backends.inpainting.base import BackendStatus, InpaintingBackend
from tools.backends.inpainting.diffusers_backend import DiffusersInpaintingBackend
from tools.backends.inpainting.flux_fill import FluxFillInpaintingBackend
from tools.backends.inpainting.mock import MockInpaintingBackend


def create_backend(name: str) -> InpaintingBackend:
    normalized = name.strip().lower().replace("-", "_")
    backends: dict[str, type[InpaintingBackend]] = {
        "mock": MockInpaintingBackend,
        "diffusers": DiffusersInpaintingBackend,
        "flux_fill": FluxFillInpaintingBackend,
        "flux": FluxFillInpaintingBackend,
    }
    backend_type = backends.get(normalized)
    if backend_type is None:
        raise ValueError(f"unknown inpainting backend: {name}")
    return backend_type()


def backend_statuses() -> list[BackendStatus]:
    return [
        MockInpaintingBackend().status(),
        DiffusersInpaintingBackend().status(),
        FluxFillInpaintingBackend().status(),
    ]


__all__ = [
    "BackendStatus",
    "InpaintingBackend",
    "backend_statuses",
    "create_backend",
]
