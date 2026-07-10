from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from tools.backends.segmentation import (
    BackendAvailability,
    GroundedSam2SegmentationBackend,
    GroundingDetection,
    MockSegmentationBackend,
    PointPrompt,
    Sam2Config,
    Sam2SegmentationBackend,
    SegmentationCandidate,
    SegmentationRequest,
)
from tools.backends.segmentation import sam2 as sam2_module


def _request(*, fixture: Image.Image | None = None) -> SegmentationRequest:
    return SegmentationRequest(
        request_id="request-eye-L",
        layer_id="eye_L",
        source_image=Image.new("RGBA", (8, 6), (30, 40, 50, 255)),
        semantic_prompt="left eye white",
        point_prompts=(PointPrompt(2, 2, 1),),
        box_prompt=(1, 1, 5, 5),
        candidate_count=1,
        minimum_confidence=0.1,
        side="L",
        fixture_masks=(fixture,) if fixture is not None else (),
    )


def test_mock_backend_contract_is_deterministic_and_preserves_soft_fixture() -> None:
    fixture = Image.new("L", (8, 6), 0)
    fixture.putpixel((2, 2), 73)
    backend = MockSegmentationBackend()

    first = backend.segment(_request(fixture=fixture), execute=True)
    second = backend.segment(_request(fixture=fixture), execute=True)

    assert first.status == second.status == "completed"
    assert first.provenance["device"] == "cpu"
    assert first.provenance["model_downloaded"] is False
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].mask.tobytes() == second.candidates[0].mask.tobytes()
    assert first.candidates[0].mask.getpixel((2, 2)) == 73


def test_mock_prompt_provenance_contains_point_box_and_semantic_prompt() -> None:
    result = MockSegmentationBackend().segment(_request(), execute=True)
    prompt = result.candidates[0].metadata["prompt_provenance"]

    assert prompt["semantic_prompt"] == "left eye white"
    assert prompt["point_prompts"] == [{"x": 2, "y": 2, "label": 1}]
    assert prompt["box_prompt"] == [1, 1, 5, 5]


def test_mock_dry_run_does_not_execute() -> None:
    result = MockSegmentationBackend().segment(_request(), execute=False)

    assert result.status == "not_run"
    assert result.candidates == ()


def test_sam2_import_is_optional_and_unavailable_is_explicit() -> None:
    backend = Sam2SegmentationBackend(
        Sam2Config(model_id="definitely-not-a-local-model", device="cpu")
    )

    dry_run = backend.segment(_request(), execute=False)
    unavailable = backend.segment(_request(), execute=True)

    assert dry_run.status == "not_run"
    assert unavailable.status == "unavailable"
    assert unavailable.message is not None
    assert "download is disabled" in unavailable.message


def test_sam2_broken_optional_import_returns_unavailable(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(sam2_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        sam2_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(KeyError("broken optional install")),
    )
    backend = Sam2SegmentationBackend(Sam2Config(model_id="facebook/sam2.1-hiera-small"))

    result = backend.segment(_request(), execute=True)

    assert result.status == "unavailable"
    assert result.message is not None
    assert "download is disabled" in result.message


def test_sam2_dry_run_never_constructs_runtime(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"fixture")
    calls: list[Sam2Config] = []

    class Runtime:
        def segment(self, request: SegmentationRequest) -> list[SegmentationCandidate]:
            raise AssertionError("runtime must not execute during dry-run")

    def factory(config: Sam2Config) -> Runtime:
        calls.append(config)
        return Runtime()

    backend = Sam2SegmentationBackend(
        Sam2Config(checkpoint=checkpoint, model_config="configs/sam2.1/test.yaml"),
        runtime_factory=factory,
    )
    result = backend.segment(_request(), execute=False)

    assert result.status == "not_run"
    assert calls == []


def test_sam2_injected_runtime_records_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"fixture")

    class Runtime:
        def segment(self, request: SegmentationRequest) -> list[SegmentationCandidate]:
            mask = Image.new("L", request.canvas, 100)
            return [
                SegmentationCandidate(
                    candidate_id="runtime-candidate",
                    mask=mask,
                    confidence=0.9,
                    stability_score=0.8,
                    bbox_xyxy=(0, 0, mask.width, mask.height),
                )
            ]

    backend = Sam2SegmentationBackend(
        Sam2Config(
            model_id="sam2.1-local",
            model_revision="rev-a",
            checkpoint=checkpoint,
            model_config="configs/sam2.1/test.yaml",
            device="cuda:0",
        ),
        runtime_factory=lambda _config: Runtime(),
    )
    result = backend.segment(_request(), execute=True)

    assert result.status == "completed"
    assert result.provenance["model_id"] == "sam2.1-local"
    assert result.provenance["model_revision"] == "rev-a"
    assert result.provenance["checkpoint"] == str(checkpoint)
    assert result.provenance["device"] == "cuda:0"


def test_grounded_sam2_preserves_multiple_detection_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"fixture")

    class Grounding:
        backend_id = "fixture-grounding"

        def check_availability(self) -> BackendAvailability:
            return BackendAvailability(True)

        def provenance(self) -> dict[str, Any]:
            return {"backend": self.backend_id, "model_downloaded": False}

        def detect(
            self,
            image: Image.Image,
            text_prompt: str,
            *,
            minimum_confidence: float,
        ) -> list[GroundingDetection]:
            assert image.size == (8, 6)
            assert text_prompt == "left eye white"
            assert minimum_confidence == 0.1
            return [
                GroundingDetection((0, 0, 3, 3), 0.9, "eye"),
                GroundingDetection((3, 0, 7, 3), 0.8, "eye-like"),
            ]

    class Runtime:
        def segment(self, request: SegmentationRequest) -> list[SegmentationCandidate]:
            assert request.box_prompt is not None
            mask = Image.new("L", request.canvas, 0)
            x1, y1, x2, y2 = (round(value) for value in request.box_prompt)
            for y in range(y1, y2):
                for x in range(x1, x2):
                    mask.putpixel((x, y), 200)
            return [
                SegmentationCandidate(
                    candidate_id=request.request_id,
                    mask=mask,
                    confidence=0.95,
                    stability_score=0.9,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            ]

    sam2 = Sam2SegmentationBackend(
        Sam2Config(checkpoint=checkpoint, model_config="configs/sam2.1/test.yaml"),
        runtime_factory=lambda _config: Runtime(),
    )
    backend = GroundedSam2SegmentationBackend(Grounding(), sam2)
    result = backend.segment(_request(), execute=True)

    assert result.status == "completed"
    assert len(result.candidates) == 2
    assert result.candidates[0].metadata["grounding"] == {
        "bbox_xyxy": [0, 0, 3, 3],
        "score": 0.9,
        "label": "eye",
        "backend": "fixture-grounding",
    }


def test_model_id_resolution_uses_official_mapping_and_local_only(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint = tmp_path / "cached.pt"
    checkpoint.write_bytes(b"cached")
    calls: list[dict[str, Any]] = []
    build_module = SimpleNamespace(
        HF_MODEL_ID_TO_FILENAMES={
            "facebook/sam2.1-hiera-small": (
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_small.pt",
            )
        }
    )

    def hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(checkpoint)

    hub_module = SimpleNamespace(hf_hub_download=hf_hub_download)
    monkeypatch.setattr(sam2_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        sam2_module.importlib,
        "import_module",
        lambda name: build_module if name == "sam2.build_sam" else hub_module,
    )

    source, reason = sam2_module._resolve_model_source(
        Sam2Config(
            model_id="facebook/sam2.1-hiera-small",
            model_revision="rev-local",
        )
    )

    assert reason is None
    assert source == ("configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint)
    assert calls == [
        {
            "repo_id": "facebook/sam2.1-hiera-small",
            "filename": "sam2.1_hiera_small.pt",
            "revision": "rev-local",
            "local_files_only": True,
        }
    ]


def test_local_runtime_passes_hydra_config_name_to_official_builder(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"fixture")
    model = object()
    calls: list[tuple[str, str, str]] = []

    class Predictor:
        def __init__(self, received_model: object) -> None:
            assert received_model is model

    def build_sam2(config_name: str, checkpoint_path: str, *, device: str) -> object:
        calls.append((config_name, checkpoint_path, device))
        return model

    modules = {
        "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=Predictor),
        "sam2.build_sam": SimpleNamespace(build_sam2=build_sam2),
    }
    monkeypatch.setattr(
        sam2_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    runtime = sam2_module._LocalSam2Runtime(
        Sam2Config(
            checkpoint=checkpoint,
            model_config="configs/sam2.1/sam2.1_hiera_s.yaml",
            device="cuda:0",
        )
    )

    loaded_model, predictor = runtime._load()

    assert loaded_model is model
    assert isinstance(predictor, Predictor)
    assert calls == [
        (
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(checkpoint.resolve()),
            "cuda:0",
        )
    ]
