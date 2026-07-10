from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from PIL import Image

from tools.backends.inpainting import backend_statuses, create_backend
from tools.backends.inpainting.diffusers_backend import DiffusersInpaintingBackend
from tools.backends.inpainting.flux_fill import FluxFillInpaintingBackend
from tools.backends.inpainting.mock import MockInpaintingBackend


def test_optional_dependencies_are_not_imported_for_normal_import() -> None:
    module = importlib.import_module("tools.generative_inpainter")
    assert callable(module.main)
    backend = DiffusersInpaintingBackend()
    assert backend._pipeline is None


def test_mock_backend_is_deterministic_and_changes_only_mask() -> None:
    image = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
    mask = Image.new("L", image.size, 0)
    mask.putpixel((2, 1), 255)
    backend = MockInpaintingBackend()
    first = backend.generate(
        image, mask, prompt="part", negative_prompt="", seed=41, config={}
    )
    second = backend.generate(
        image, mask, prompt="part", negative_prompt="", seed=41, config={}
    )
    third = backend.generate(
        image, mask, prompt="part", negative_prompt="", seed=42, config={}
    )
    assert first.tobytes() == second.tobytes()
    assert first.getpixel((2, 1)) != third.getpixel((2, 1))
    assert first.getpixel((0, 0)) == image.getpixel((0, 0))
    assert first.getpixel((2, 1)) != image.getpixel((2, 1))


def test_backend_registry_and_unavailable_status_are_explicit() -> None:
    assert create_backend("mock").name == "mock"
    assert create_backend("flux-fill").name == "flux_fill"
    assert isinstance(create_backend("diffusers"), DiffusersInpaintingBackend)
    assert isinstance(create_backend("flux"), FluxFillInpaintingBackend)
    statuses = {status.name: status for status in backend_statuses()}
    assert set(statuses) == {"mock", "diffusers", "flux_fill"}
    assert statuses["mock"].available is True
    assert all(status.detail for status in statuses.values())


def _fake_optional_runtime(monkeypatch: Any) -> tuple[Any, Any]:
    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device
            self.seed: int | None = None

        def manual_seed(self, seed: int) -> FakeGenerator:
            self.seed = seed
            return self

    class FakeTorch:
        float32 = object()
        float16 = object()
        bfloat16 = object()
        Generator = FakeGenerator

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: Any) -> str:
            return f"scheduled:{config}"

    class FakePipeline:
        def __init__(self) -> None:
            self.scheduler = SimpleNamespace(config="base")
            self.offload_device: str | None = None
            self.attention_slice: object | None = None
            self.to_device: str | None = None
            self.calls: list[dict[str, Any]] = []

        def enable_model_cpu_offload(self, *, device: str) -> None:
            self.offload_device = device

        def enable_attention_slicing(self, value: object) -> None:
            self.attention_slice = value

        def to(self, device: str) -> FakePipeline:
            self.to_device = device
            return self

        def __call__(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGBA", kwargs["image"].size, "red")])

    class FakePipelineClass:
        load_calls: list[tuple[str, dict[str, Any]]] = []
        instance = FakePipeline()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakePipeline:
            cls.load_calls.append((model_id, kwargs))
            return cls.instance

    fake_diffusers = SimpleNamespace(
        AutoPipelineForInpainting=FakePipelineClass,
        FluxFillPipeline=FakePipelineClass,
        FakeScheduler=FakeScheduler,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_diffusers if name == "diffusers" else FakeTorch,
    )
    return FakePipelineClass, FakePipelineClass.instance


def test_diffusers_adapter_lazily_applies_backend_config_and_reuses_pipeline(
    monkeypatch: Any,
) -> None:
    pipeline_class, pipeline = _fake_optional_runtime(monkeypatch)
    backend = DiffusersInpaintingBackend()
    image = Image.new("RGBA", (8, 6), "white")
    mask = Image.new("L", image.size, 255)
    config = {
        "model_id": "local/model",
        "model_revision": "revision-1",
        "dtype": "float16",
        "offline": True,
        "local_files_only": False,
        "scheduler": "FakeScheduler",
        "device": "cuda",
        "cpu_offload": True,
        "attention_slicing": "max",
        "inference_steps": 12,
        "guidance_scale": 4.5,
        "strength": 0.75,
    }
    for seed in (101, 102):
        generated = backend.generate(
            image,
            mask,
            prompt="part",
            negative_prompt="forbidden",
            seed=seed,
            config=config,
        )
        assert generated.size == image.size
    assert len(pipeline_class.load_calls) == 1
    model_id, load_kwargs = pipeline_class.load_calls[0]
    assert model_id == "local/model"
    assert load_kwargs["revision"] == "revision-1"
    assert load_kwargs["local_files_only"] is True
    assert pipeline.scheduler == "scheduled:base"
    assert pipeline.offload_device == "cuda"
    assert pipeline.attention_slice == "max"
    assert pipeline.calls[0]["generator"].device == "cpu"
    assert pipeline.calls[0]["generator"].seed == 101
    assert pipeline.calls[0]["num_inference_steps"] == 12
    assert pipeline.calls[0]["guidance_scale"] == 4.5
    assert pipeline.calls[0]["strength"] == 0.75


def test_flux_fill_adapter_omits_regular_inpaint_strength(monkeypatch: Any) -> None:
    _, pipeline = _fake_optional_runtime(monkeypatch)
    backend = FluxFillInpaintingBackend()
    image = Image.new("RGBA", (8, 6), "white")
    backend.generate(
        image,
        Image.new("L", image.size, 255),
        prompt="part",
        negative_prompt="ignored by Flux Fill",
        seed=3,
        config={
            "model_id": "local/flux-fill",
            "device": "cpu",
            "max_sequence_length": 256,
        },
    )
    call = pipeline.calls[-1]
    assert "strength" not in call
    assert "negative_prompt" not in call
    assert call["height"] == 6
    assert call["width"] == 8
    assert call["max_sequence_length"] == 256
