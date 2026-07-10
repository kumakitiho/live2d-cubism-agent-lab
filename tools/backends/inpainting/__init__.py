from __future__ import annotations

from typing import cast

from tools.backend_registry import registry
from tools.backends.inpainting.base import BackendStatus, InpaintingBackend


def create_backend(name: str) -> InpaintingBackend:
    return cast(InpaintingBackend, registry.get_inpainting(name))


def backend_statuses() -> list[BackendStatus]:
    return [registry.get_inpainting(name).status() for name in registry.inpainting_names]


__all__ = [
    "BackendStatus",
    "InpaintingBackend",
    "backend_statuses",
    "create_backend",
]
