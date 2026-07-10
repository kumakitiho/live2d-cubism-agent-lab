from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from tools.backends.inpainting.base import config_positive_int
from tools.backends.inpainting.diffusers_backend import DiffusersInpaintingBackend


class FluxFillInpaintingBackend(DiffusersInpaintingBackend):
    """Optional Diffusers FLUX Fill adapter; model selection and licensing stay user-owned."""

    name = "flux_fill"
    recommended_size = 1024
    pipeline_class_name = "FluxFillPipeline"

    def _call_arguments(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        arguments = super()._call_arguments(
            image,
            mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            config=config,
        )
        arguments.pop("negative_prompt", None)
        arguments.pop("strength", None)
        arguments.update(
            {
                "height": image.height,
                "width": image.width,
                "max_sequence_length": config_positive_int(
                    config, "max_sequence_length", 512
                ),
            }
        )
        return arguments
