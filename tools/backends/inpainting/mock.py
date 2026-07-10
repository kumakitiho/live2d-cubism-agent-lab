from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from PIL import Image

from tools.backends.inpainting.base import BackendStatus


class MockInpaintingBackend:
    """Small deterministic backend used by CI and contract tests."""

    name = "mock"
    recommended_size = 64

    def status(self) -> BackendStatus:
        return BackendStatus(self.name, True, "deterministic CPU fixture backend")

    def release(self) -> None:
        """Mock backend has no retained model resources."""

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> Image.Image:
        del prompt, negative_prompt, config
        rgba = image.convert("RGBA")
        binary_mask = mask.convert("L")
        if rgba.size != binary_mask.size:
            raise ValueError("backend image and mask must use the same canvas")
        result = rgba.copy()
        pixels: Any = result.load()
        mask_pixels: Any = binary_mask.load()
        rng = random.Random(seed)
        phase = rng.randrange(1, 251)
        for y in range(result.height):
            for x in range(result.width):
                if mask_pixels[x, y] == 0:
                    continue
                pixels[x, y] = (
                    24 + ((x * 37 + y * 17 + phase) % 208),
                    24 + ((x * 13 + y * 41 + phase * 3) % 208),
                    24 + ((x * 29 + y * 11 + phase * 7) % 208),
                    255,
                )
        return result
