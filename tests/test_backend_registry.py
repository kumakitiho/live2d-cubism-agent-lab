from __future__ import annotations

import sys

import pytest

from tools.backend_registry import BackendRegistry


def test_registry_gets_segmentation_and_inpainting_mock() -> None:
    registry = BackendRegistry()

    assert registry.get_segmentation("mock").backend_id == "mock"
    assert registry.get_inpainting("mock").name == "mock"


@pytest.mark.parametrize("kind", ["segmentation", "inpainting"])
def test_registry_rejects_unknown_backend(kind: str) -> None:
    registry = BackendRegistry()

    with pytest.raises(ValueError, match="unknown"):
        if kind == "segmentation":
            registry.get_segmentation("missing")
        else:
            registry.get_inpainting("missing")


def test_optional_segmentation_without_model_source_is_unavailable() -> None:
    status = BackendRegistry().availability("segmentation", "sam2")

    assert status.status == "unavailable"
    assert status.available is False
    assert status.reason


def test_registry_resolution_is_lazy_and_does_not_load_models() -> None:
    registry = BackendRegistry()
    before = set(sys.modules)

    inpainting = registry.get_inpainting("diffusers")
    segmentation = registry.get_segmentation("sam2")

    assert inpainting._pipeline is None
    assert "torch" not in set(sys.modules) - before
    assert not hasattr(segmentation, "_model")


def test_registry_accepts_public_backend_aliases() -> None:
    registry = BackendRegistry()

    assert registry.get_inpainting("flux-fill").name == "flux_fill"
    assert registry.get_segmentation("grounded-sam2").backend_id == "grounded-sam2"
