from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


class BackendUnavailableError(RuntimeError):
    """Raised when an optional backend cannot be executed in this environment."""


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "available": self.available, "detail": self.detail}


class InpaintingBackend(Protocol):
    name: str
    recommended_size: int

    def status(self) -> BackendStatus: ...

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> Image.Image: ...


def model_size(config: Mapping[str, Any], *, default: int) -> tuple[int, int]:
    value = config.get("model_size", default)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError("backend_config.model_size must be a positive integer or [width, height]")


def config_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < minimum:
        raise ValueError(f"backend_config.{key} must be a number >= {minimum}")
    return float(value)


def config_positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"backend_config.{key} must be a positive integer")
    return value
