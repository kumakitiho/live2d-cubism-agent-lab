from __future__ import annotations

import argparse
import io
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from PIL import Image, ImageChops

from tools.asset_pipeline_common import (
    atomic_save_png,
    referenced_artifact_paths,
    require_output_suffix,
    resolve_inside_base,
)
from tools.asset_queue_builder import normalize_queue_ref
from tools.backend_registry import registry, release_backend
from tools.backends.segmentation.contracts import (
    PointPrompt,
    SegmentationBackend,
    SegmentationCandidate,
    SegmentationRequest,
)
from tools.backends.segmentation.integrity import (
    bytes_sha256,
    canonical_mapping_sha256,
    file_sha256,
)
from tools.segmentation_preview import build_segmentation_preview


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_canvas(queue: Mapping[str, Any]) -> tuple[int, int]:
    canvas = _require_mapping(queue.get("canvas"), "queue canvas")
    width = canvas.get("width")
    height = canvas.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise ValueError("queue canvas width/height must be positive integers")
    return width, height


def _queue_assets(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    assets = queue.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("queue assets must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    layer_ids: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"queue assets[{index}] must be a mapping")
        layer_id = asset.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"queue assets[{index}].layer_id must be a non-empty string")
        if layer_id in layer_ids:
            raise ValueError(f"duplicate queue layer_id: {layer_id}")
        layer_ids.add(layer_id)
        result.append(asset)
    return result


def _source_path(queue: Mapping[str, Any], base_dir: Path) -> tuple[Path, str]:
    source = _require_mapping(queue.get("source_image"), "queue source_image")
    source_value = source.get("path")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError("queue source_image.path must be a non-empty string")
    return (
        resolve_inside_base(base_dir, source_value, "queue source_image.path"),
        source_value,
    )


def _load_source(
    queue: Mapping[str, Any],
    base_dir: Path,
    *,
    execute: bool,
) -> tuple[Image.Image, Path, str, str | None]:
    canvas = _require_canvas(queue)
    path, reference = _source_path(queue, base_dir)
    if not execute:
        return Image.new("RGBA", canvas, (0, 0, 0, 0)), path, reference, None
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    source_bytes = path.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as opened:
        image = opened.convert("RGBA")
    if image.size != canvas:
        raise ValueError(f"source image canvas mismatch: {image.size} != {canvas}")
    return image, path, reference, bytes_sha256(source_bytes)


def _load_queue_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    queue_bytes = path.read_bytes()
    raw = yaml.safe_load(queue_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw, bytes_sha256(queue_bytes)


def _segmentation_settings(asset: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = asset.get("segmentation")
    return settings if isinstance(settings, Mapping) else {}


def _point_prompts(value: object) -> tuple[PointPrompt, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("point_prompts must be a list")
    result: list[PointPrompt] = []
    for index, point in enumerate(value):
        if isinstance(point, Mapping):
            x = point.get("x")
            y = point.get("y")
            label = point.get("label", 1)
        elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
            values = list(point)
            if len(values) not in {2, 3}:
                raise ValueError(f"point_prompts[{index}] must contain x, y, and optional label")
            x, y = values[:2]
            label = values[2] if len(values) == 3 else 1
        else:
            raise ValueError(f"point_prompts[{index}] must be a mapping or sequence")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"point_prompts[{index}] x/y must be numbers")
        if not isinstance(label, int) or isinstance(label, bool):
            raise ValueError(f"point_prompts[{index}] label must be 0 or 1")
        result.append(PointPrompt(float(x), float(y), label))
    return tuple(result)


def _box_prompt(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("box_prompt must be an xyxy sequence")
    values = list(value)
    if len(values) != 4 or not all(isinstance(item, (int, float)) for item in values):
        raise ValueError("box_prompt must contain four numbers")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _expected_region(value: object) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("expected_region must be a mapping")
    result: dict[str, float] = {}
    for key in ("x_min", "y_min", "x_max", "y_max"):
        item = value.get(key)
        if not isinstance(item, (int, float)):
            raise ValueError(f"expected_region.{key} must be a number")
        result[key] = float(item)
    if not 0 <= result["x_min"] < result["x_max"] <= 1:
        raise ValueError("expected_region x bounds must be normalized")
    if not 0 <= result["y_min"] < result["y_max"] <= 1:
        raise ValueError("expected_region y bounds must be normalized")
    return result


def _load_optional_mask(
    value: object,
    *,
    base_dir: Path,
    canvas: tuple[int, int],
    field: str,
    execute: bool,
) -> Image.Image | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    if not execute:
        return None
    path = resolve_inside_base(base_dir, value, field)
    if not path.is_file():
        raise FileNotFoundError(f"mask not found: {path}")
    with Image.open(path) as opened:
        mask = opened.convert("L")
    if mask.size != canvas:
        raise ValueError(f"{field} canvas mismatch: {mask.size} != {canvas}")
    return mask


def _fixture_masks(
    values: Sequence[Path],
    *,
    base_dir: Path,
    canvas: tuple[int, int],
    execute: bool,
) -> tuple[Image.Image, ...]:
    if not execute:
        return ()
    result: list[Image.Image] = []
    for index, value in enumerate(values):
        path = resolve_inside_base(base_dir, str(value), f"fixture_masks[{index}]")
        if not path.is_file():
            raise FileNotFoundError(f"fixture mask not found: {path}")
        with Image.open(path) as opened:
            mask = opened.convert("L")
        if mask.size != canvas:
            raise ValueError(f"fixture mask canvas mismatch: {mask.size} != {canvas}")
        result.append(mask)
    return tuple(result)


def build_request(
    asset: Mapping[str, Any],
    source: Image.Image,
    *,
    base_dir: Path,
    execute: bool,
    candidate_count: int | None = None,
    minimum_confidence: float | None = None,
    fixture_mask_paths: Sequence[Path] = (),
) -> SegmentationRequest:
    settings = _segmentation_settings(asset)
    layer_id = str(asset["layer_id"])
    semantic_value = settings.get(
        "semantic_prompt",
        asset.get("semantic_prompt", asset.get("role", layer_id)),
    )
    semantic_prompt = str(semantic_value).replace("_", " ").strip()
    side_value = asset.get("side", "none")
    side = str(side_value) if side_value is not None else "none"
    configured_count = settings.get("candidate_count", 3)
    count = candidate_count if candidate_count is not None else configured_count
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError(f"asset {layer_id} candidate_count must be an integer")
    configured_confidence = settings.get("minimum_confidence", 0.0)
    confidence = minimum_confidence if minimum_confidence is not None else configured_confidence
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"asset {layer_id} minimum_confidence must be a number")
    per_asset_fixtures = settings.get("fixture_masks", [])
    if not isinstance(per_asset_fixtures, list) or not all(
        isinstance(value, str) for value in per_asset_fixtures
    ):
        raise ValueError(f"asset {layer_id} fixture_masks must be a list of paths")
    fixture_paths = [*fixture_mask_paths, *(Path(value) for value in per_asset_fixtures)]
    return SegmentationRequest(
        request_id=f"segmentation:{layer_id}",
        layer_id=layer_id,
        source_image=source,
        semantic_prompt=semantic_prompt,
        point_prompts=_point_prompts(settings.get("point_prompts", asset.get("point_prompts"))),
        box_prompt=_box_prompt(settings.get("box_prompt", asset.get("box_prompt"))),
        existing_mask=_load_optional_mask(
            settings.get("existing_mask", asset.get("existing_mask")),
            base_dir=base_dir,
            canvas=source.size,
            field=f"asset {layer_id} existing_mask",
            execute=execute,
        ),
        candidate_count=count,
        minimum_confidence=float(confidence),
        side=side,
        expected_region=_expected_region(
            settings.get("expected_region", asset.get("expected_region"))
        ),
        fixture_masks=_fixture_masks(
            fixture_paths,
            base_dir=base_dir,
            canvas=source.size,
            execute=execute,
        ),
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not normalized:
        raise ValueError("candidate output name is empty after normalization")
    return normalized


def _relative(path: Path, base_dir: Path) -> str:
    return path.resolve().relative_to(base_dir.resolve()).as_posix()


def _binary_mask(mask: Image.Image, threshold: int) -> Image.Image:
    return mask.convert("L").point(lambda value: 255 if value >= threshold else 0, mode="L")


def _pixel_count(binary: Image.Image) -> int:
    return binary.histogram()[255]


def _review_reasons(
    candidate: SegmentationCandidate,
    *,
    binary: Image.Image,
    source: Image.Image,
    request: SegmentationRequest,
    competing_candidates: int,
) -> list[str]:
    reasons: list[str] = []
    area = _pixel_count(binary)
    canvas_area = binary.width * binary.height
    if candidate.confidence < max(0.5, request.minimum_confidence):
        reasons.append("low_confidence")
    if area == 0:
        reasons.append("empty_binary_mask")
    if request.side == "none":
        reasons.append("ambiguous_side")
    if competing_candidates > 1:
        reasons.append("candidate_competition")
    area_ratio = area / canvas_area
    if area_ratio < 0.0001 or area_ratio > 0.75:
        reasons.append("abnormal_area")
    alpha_binary = source.getchannel("A").point(
        lambda value: 255 if value > 0 else 0,
        mode="L",
    )
    intersection = ImageChops.multiply(binary, alpha_binary)
    outside = area - _pixel_count(intersection)
    if area and outside / area > 0.05:
        reasons.append("mask_outside_source_alpha")
    return reasons


def _candidate_record(
    candidate: SegmentationCandidate,
    *,
    asset: Mapping[str, Any],
    request: SegmentationRequest,
    backend_id: str,
    backend_provenance: Mapping[str, Any],
    output_dir: Path,
    base_dir: Path,
    threshold: int,
    source: Image.Image,
    competing_candidates: int,
) -> tuple[dict[str, Any], list[tuple[Path, Image.Image]]]:
    if candidate.mask.size != source.size:
        raise ValueError(
            f"candidate {candidate.candidate_id} canvas mismatch: "
            f"{candidate.mask.size} != {source.size}"
        )
    soft = candidate.mask.convert("L").copy()
    binary = _binary_mask(soft, threshold)
    base_name = f"{_slug(request.layer_id)}.{_slug(candidate.candidate_id)}"
    soft_path = output_dir / f"{base_name}.soft.png"
    binary_path = output_dir / f"{base_name}.binary.png"
    preview_path = output_dir / f"{base_name}.preview.png"
    reasons = _review_reasons(
        candidate,
        binary=binary,
        source=source,
        request=request,
        competing_candidates=competing_candidates,
    )
    model_id = backend_provenance.get("model_id")
    model_revision = backend_provenance.get("model_revision")
    if backend_id == "grounded-sam2":
        sam2 = backend_provenance.get("sam2")
        if isinstance(sam2, Mapping):
            model_id = sam2.get("model_id")
            model_revision = sam2.get("model_revision")
    record = {
        "candidate_id": candidate.candidate_id,
        "layer_id": request.layer_id,
        "semantic_prompt": request.semantic_prompt,
        "mask_file": _relative(binary_path, base_dir),
        "soft_mask_file": _relative(soft_path, base_dir),
        "binary_mask_file": _relative(binary_path, base_dir),
        "preview_file": _relative(preview_path, base_dir),
        "confidence": round(candidate.confidence, 6),
        "stability_score": round(candidate.stability_score, 6),
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "area_px": _pixel_count(binary),
        "source_backend": backend_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_provenance": {
            **request.prompt_provenance(),
            **candidate.metadata,
        },
        "side": asset.get("side", "none"),
        "role": asset.get("role"),
        "semantic_assignment": {
            "layer_id": request.layer_id,
            "role": asset.get("role"),
            "side": asset.get("side", "none"),
            "status": "proposed",
        },
        "expected_region": dict(request.expected_region) if request.expected_region else None,
        "draw_order": asset.get("draw_order"),
        "requires_review": bool(reasons),
        "rejection_reasons": reasons,
    }
    return record, [
        (soft_path, soft),
        (binary_path, binary),
        (preview_path, build_segmentation_preview(source, soft)),
    ]


def _make_backend(args: argparse.Namespace) -> SegmentationBackend:
    return cast(
        SegmentationBackend,
        registry.get_segmentation(
            args.backend,
            {
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "checkpoint": (args.checkpoint.resolve() if args.checkpoint is not None else None),
                "model_config": args.model_config,
                "grounding_model": (
                    args.grounding_model.resolve() if args.grounding_model is not None else None
                ),
                "grounding_model_revision": args.grounding_model_revision,
                "device": args.device,
            },
        ),
    )


def _preflight_outputs(
    output: Path,
    png_outputs: Sequence[Path],
    *,
    protected_paths: set[Path],
    force: bool,
) -> None:
    all_outputs = [output.resolve(), *(path.resolve() for path in png_outputs)]
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("segmentation outputs contain a path collision")
    collisions = set(all_outputs) & {path.resolve() for path in protected_paths}
    if collisions:
        raise ValueError(
            "segmentation output must not overwrite canonical input/artifact paths: "
            + ", ".join(str(path) for path in sorted(collisions))
        )
    existing = [path for path in all_outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite without --force: " + ", ".join(str(path) for path in existing)
        )


def _atomic_write_yaml(path: Path, data: Mapping[str, Any], *, force: bool) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "segmentation YAML output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reviewable segmentation candidates from the canonical asset queue."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--backend", choices=("mock", "sam2", "grounded-sam2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--minimum-confidence", type=float)
    parser.add_argument("--binary-threshold", type=int, default=128)
    parser.add_argument("--fixture-mask", type=Path, action="append", default=[])
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model-config",
        help="Hydra config name packaged with the installed SAM 2 distribution",
    )
    parser.add_argument("--grounding-model", type=Path)
    parser.add_argument("--grounding-model-revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    backend: SegmentationBackend | None = None
    try:
        if args.candidate_count is not None and args.candidate_count <= 0:
            raise ValueError("--candidate-count must be positive")
        if args.minimum_confidence is not None and not 0 <= args.minimum_confidence <= 1:
            raise ValueError("--minimum-confidence must be between 0 and 1")
        if not 1 <= args.binary_threshold <= 255:
            raise ValueError("--binary-threshold must be between 1 and 255")
        queue_path = resolve_inside_base(base_dir, str(args.queue), "asset generation queue")
        queue, queue_sha256 = _load_queue_snapshot(queue_path)
        assets = _queue_assets(queue)
        source, source_path, source_reference, source_sha256 = _load_source(
            queue,
            base_dir,
            execute=args.execute,
        )
        output = resolve_inside_base(base_dir, str(args.output), "segmentation result output")
        require_output_suffix(output, {".yaml", ".yml"}, "segmentation result output")
        output_dir_value = args.output_dir or output.parent / f"{output.stem}_masks"
        output_dir = resolve_inside_base(
            base_dir,
            str(output_dir_value),
            "segmentation mask output directory",
        )
        backend = _make_backend(args)
        queue_ref = normalize_queue_ref(queue_path, base_dir)
        if args.run_id is not None and not args.run_id.strip():
            raise ValueError("--run-id must be a non-empty string")
        run_id = args.run_id or str(uuid.uuid4())
        records: list[dict[str, Any]] = []
        pending_pngs: list[tuple[Path, Image.Image]] = []
        requests: list[dict[str, Any]] = []
        backend_provenance: dict[str, Any] = {}
        seen_candidate_ids: set[str] = set()
        for asset in assets:
            request = build_request(
                asset,
                source,
                base_dir=base_dir,
                execute=args.execute,
                candidate_count=args.candidate_count,
                minimum_confidence=args.minimum_confidence,
                fixture_mask_paths=args.fixture_mask,
            )
            requests.append(
                {
                    "layer_id": request.layer_id,
                    "side": request.side,
                    **request.prompt_provenance(),
                }
            )
            result = backend.segment(request, execute=args.execute)
            backend_provenance = result.provenance
            if not args.execute:
                if result.status != "not_run":
                    raise RuntimeError(
                        f"backend returned unexpected dry-run status: {result.status}"
                    )
                continue
            if result.status != "completed":
                raise RuntimeError(
                    f"segmentation backend {result.status}: {result.message or 'no details'}"
                )
            if not result.candidates:
                raise RuntimeError(f"segmentation returned no candidates for {request.layer_id}")
            for candidate in result.candidates:
                if candidate.candidate_id in seen_candidate_ids:
                    raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
                seen_candidate_ids.add(candidate.candidate_id)
                record, images = _candidate_record(
                    candidate,
                    asset=asset,
                    request=request,
                    backend_id=backend.backend_id,
                    backend_provenance=result.provenance,
                    output_dir=output_dir,
                    base_dir=base_dir,
                    threshold=args.binary_threshold,
                    source=source,
                    competing_candidates=len(result.candidates),
                )
                records.append(record)
                pending_pngs.extend(images)
        document = {
            "schema_version": 1,
            "status": "completed" if args.execute else "planned",
            "project": queue.get("project"),
            "run_id": run_id,
            "asset_generation_queue": queue_ref,
            "asset_generation_queue_sha256": queue_sha256,
            "asset_generation_queue_content_sha256": canonical_mapping_sha256(queue),
            "source_image": {"path": source_reference},
            "source_image_sha256": source_sha256,
            "canvas": {"width": source.width, "height": source.height, "origin": [0, 0]},
            "backend": backend.backend_id,
            "backend_provenance": backend_provenance,
            "binary_threshold": args.binary_threshold,
            "requests": requests,
            "candidates": records,
            "summary": {
                "request_count": len(requests),
                "candidate_count": len(records),
                "automatic_assignment": False,
                "review_required": True,
            },
        }
        protected_paths = referenced_artifact_paths(
            queue,
            base_dir,
            document_path=queue_path,
        )
        protected_paths.add(source_path)
        if file_sha256(queue_path) != queue_sha256:
            raise RuntimeError("canonical queue changed during segmentation; refusing stale output")
        if args.execute:
            assert source_sha256 is not None
            if file_sha256(source_path) != source_sha256:
                raise RuntimeError(
                    "source image changed during segmentation; refusing stale output"
                )
        _preflight_outputs(
            output,
            [path for path, _image in pending_pngs],
            protected_paths=protected_paths,
            force=args.force,
        )
        if args.execute:
            for path, image in pending_pngs:
                atomic_save_png(image, path, force=args.force)
            _atomic_write_yaml(output, document, force=args.force)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if backend is not None:
            release_backend(backend)

    result_summary = {
        "status": document["status"],
        "output": str(output),
        "backend": document["backend"],
        "request_count": document["summary"]["request_count"],
        "candidate_count": document["summary"]["candidate_count"],
    }
    if args.json:
        print(json.dumps(result_summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(result_summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
