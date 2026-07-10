from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from typing import Any

from PIL import Image

from tools.backends.inpainting.base import (
    BackendStatus,
    BackendUnavailableError,
    config_float,
    config_positive_int,
)


class DiffusersInpaintingBackend:
    """Lazy adapter around Diffusers AutoPipelineForInpainting."""

    name = "diffusers"
    recommended_size = 512
    pipeline_class_name = "AutoPipelineForInpainting"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._loaded_key: tuple[object, ...] | None = None

    def status(self) -> BackendStatus:
        missing = [
            name for name in ("diffusers", "torch") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return BackendStatus(
                self.name,
                False,
                f"optional dependencies unavailable: {', '.join(missing)}",
            )
        return BackendStatus(self.name, True, "optional dependencies are importable")

    def _pipeline_key(self, config: Mapping[str, Any]) -> tuple[object, ...]:
        return (
            config.get("model_id"),
            config.get("model_revision"),
            config.get("dtype", "float32"),
            bool(config.get("offline", False)),
            bool(config.get("local_files_only", True)),
            config.get("scheduler"),
            config.get("device", "cpu"),
            bool(config.get("cpu_offload", False)),
            config.get("attention_slicing", False),
        )

    def _load_pipeline(self, config: Mapping[str, Any]) -> Any:
        status = self.status()
        if not status.available:
            raise BackendUnavailableError(status.detail)
        model_id = config.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("backend_config.model_id is required")
        key = self._pipeline_key(config)
        if self._pipeline is not None and self._loaded_key == key:
            return self._pipeline

        diffusers = importlib.import_module("diffusers")
        torch = importlib.import_module("torch")
        pipeline_class = getattr(diffusers, self.pipeline_class_name, None)
        if pipeline_class is None:
            raise BackendUnavailableError(
                f"installed diffusers does not provide {self.pipeline_class_name}"
            )
        dtype_name = config.get("dtype", "float32")
        if not isinstance(dtype_name, str) or not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported backend_config.dtype: {dtype_name}")
        local_files_only = bool(config.get("local_files_only", True)) or bool(
            config.get("offline", False)
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype_name),
            "local_files_only": local_files_only,
        }
        revision = config.get("model_revision")
        if revision is not None:
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("backend_config.model_revision must be a non-empty string")
            load_kwargs["revision"] = revision
        pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)

        scheduler_name = config.get("scheduler")
        if scheduler_name is not None:
            if not isinstance(scheduler_name, str) or not scheduler_name.strip():
                raise ValueError("backend_config.scheduler must be a non-empty string")
            scheduler_class = getattr(diffusers, scheduler_name, None)
            if scheduler_class is None or not hasattr(scheduler_class, "from_config"):
                raise ValueError(f"unsupported diffusers scheduler: {scheduler_name}")
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

        device = config.get("device", "cpu")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("backend_config.device must be a non-empty string")
        if bool(config.get("cpu_offload", False)):
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline = pipeline.to(device)
        attention_slicing = config.get("attention_slicing", False)
        if attention_slicing:
            slice_size = "auto" if attention_slicing is True else attention_slicing
            pipeline.enable_attention_slicing(slice_size)
        self._pipeline = pipeline
        self._loaded_key = key
        return pipeline

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
        torch = importlib.import_module("torch")
        device = str(config.get("device", "cpu"))
        generator_device = "cpu" if bool(config.get("cpu_offload", False)) else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "image": image.convert("RGB"),
            "mask_image": mask.convert("L"),
            "generator": generator,
            "num_inference_steps": config_positive_int(config, "inference_steps", 30),
            "guidance_scale": config_float(config, "guidance_scale", 7.5),
            "strength": config_float(config, "strength", 1.0),
        }

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
        pipeline = self._load_pipeline(config)
        result = pipeline(
            **self._call_arguments(
                image,
                mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                config=config,
            )
        )
        images = getattr(result, "images", None)
        if not isinstance(images, list) or not images:
            raise RuntimeError("diffusers pipeline returned no images")
        generated = images[0]
        if not isinstance(generated, Image.Image):
            raise RuntimeError("diffusers pipeline returned a non-image result")
        return generated.convert("RGBA")
