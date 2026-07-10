from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import yaml
from PIL import Image

from tools.asset_pipeline_common import (
    load_rgba,
    load_soft_mask,
    referenced_artifact_paths,
    resolve_inside_base,
)
from tools.backends.segmentation.integrity import (
    bytes_sha256,
    canonical_mapping_sha256,
    file_sha256,
)
from tools.mask_derivation.algorithms import (
    FRONT_HAIR_ROLES,
    binary,
    derive_edge_extension_mask,
    derive_forehead_inpaint_mask,
    derive_protect_mask,
    detect_candidate_conflicts,
    dilate,
    mask_intersection,
    mask_union,
    pixel_count,
    symmetry_warning,
)
from tools.mask_derivation_preview import build_mask_derivation_preview


@dataclass(frozen=True)
class DerivationConfig:
    protect_radius_px: int = 2
    edge_radius_px: int = 2
    fine_part_min_area_px: int = 1
    max_candidate_area_ratio: float = 2.0
    min_island_area_px: int = 2
    binary_threshold: int = 1

    def validate(self) -> None:
        for name, value in (
            ("protect_radius_px", self.protect_radius_px),
            ("edge_radius_px", self.edge_radius_px),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.fine_part_min_area_px < 1:
            raise ValueError("fine_part_min_area_px must be positive")
        if self.max_candidate_area_ratio <= 0:
            raise ValueError("max_candidate_area_ratio must be positive")
        if self.min_island_area_px < 1:
            raise ValueError("min_island_area_px must be positive")
        if not 1 <= self.binary_threshold <= 255:
            raise ValueError("binary_threshold must be between 1 and 255")


@dataclass(frozen=True)
class DerivationArtifacts:
    payloads: Mapping[Path, SpooledTemporaryFile[bytes]]
    output_paths: frozenset[Path]
    input_paths: frozenset[Path]

    def read_bytes(self, path: Path) -> bytes:
        stream = self.payloads[path]
        stream.seek(0)
        return stream.read()

    @property
    def png_bytes(self) -> dict[Path, bytes]:
        return {path: self.read_bytes(path) for path in self.payloads}

    @property
    def images(self) -> dict[Path, Image.Image]:
        """Decode artifacts on demand for tests and in-memory API consumers."""
        result: dict[Path, Image.Image] = {}
        for path in self.payloads:
            content = self.read_bytes(path)
            with Image.open(BytesIO(content)) as opened:
                result[path] = opened.copy()
        return result

    def close(self) -> None:
        for stream in self.payloads.values():
            stream.close()


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _canvas(queue: Mapping[str, Any]) -> tuple[int, int]:
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


def _assets(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = queue.get("assets")
    if not isinstance(raw, list) or not raw:
        raise ValueError("queue assets must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        asset = _require_mapping(item, f"queue assets[{index}]")
        layer_id = asset.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"queue assets[{index}].layer_id must be a non-empty string")
        if layer_id in seen:
            raise ValueError(f"duplicate queue layer_id: {layer_id}")
        seen.add(layer_id)
        result.append(asset)
    return result


def _source_path(queue: Mapping[str, Any], base_dir: Path) -> tuple[Path, str]:
    source = _require_mapping(queue.get("source_image"), "queue source_image")
    value = source.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("queue source_image.path must be a non-empty string")
    return resolve_inside_base(base_dir, value, "queue source_image.path"), value


def _target_path(asset: Mapping[str, Any], base_dir: Path) -> tuple[Path, str]:
    value = asset.get("target_mask")
    layer_id = asset.get("layer_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"layer {layer_id} target_mask must be a non-empty string")
    return resolve_inside_base(base_dir, value, f"layer {layer_id} target_mask"), value


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("layer ID cannot produce an output filename")
    return result


def _layer_token(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(value)}-{digest}"


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _store_payload(
    payloads: dict[Path, SpooledTemporaryFile[bytes]],
    path: Path,
    content: bytes,
) -> None:
    existing = payloads.get(path)
    if existing is not None:
        existing.close()
    stream = SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")  # noqa: SIM115
    stream.write(content)
    stream.seek(0)
    payloads[path] = stream


def _relative(path: Path, base_dir: Path) -> str:
    return path.resolve().relative_to(base_dir.resolve()).as_posix()


def deterministic_run_id(
    *,
    project: object,
    queue_sha256: str,
    source_sha256: str,
    target_hashes: Mapping[str, str],
    config: DerivationConfig,
    layer_ids: Sequence[str],
) -> str:
    layer_scope_sha256 = canonical_mapping_sha256({"selected_layers": sorted(layer_ids)})
    material = "|".join(
        [
            str(project),
            queue_sha256,
            source_sha256,
            canonical_mapping_sha256(dict(sorted(target_hashes.items()))),
            repr(config),
            layer_scope_sha256,
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"live2d-mask-derivation:{material}"))


def _candidate_paths(
    output_dir: Path,
    *,
    layer_id: str,
    mask_type: str,
    artifact_scope: str,
) -> tuple[Path, Path, Path]:
    base = f"{_layer_token(layer_id)}.{mask_type}.{artifact_scope}"
    return (
        output_dir / f"{base}.soft.png",
        output_dir / f"{base}.binary.png",
        output_dir / f"{base}.preview.png",
    )


def _candidate_record(
    *,
    layer_id: str,
    mask_type: str,
    mask: Image.Image,
    details: Mapping[str, Any],
    confidence: float,
    output_dir: Path,
    base_dir: Path,
    run_id: str,
    artifact_scope: str,
    conflict_types: Sequence[str],
    adjacent_layers: Sequence[str] = (),
) -> dict[str, Any]:
    soft_path, binary_path, preview_path = _candidate_paths(
        output_dir,
        layer_id=layer_id,
        mask_type=mask_type,
        artifact_scope=artifact_scope,
    )
    rejection_reasons = [
        value.split(":", 1)[-1] for value in conflict_types if value.startswith("reject:")
    ]
    review_reasons = [
        value.split(":", 1)[-1] for value in conflict_types if value.startswith("review:")
    ]
    rejected = bool(rejection_reasons)
    reasons = [*rejection_reasons, *review_reasons]
    warnings = [str(value) for value in details.get("warnings", [])]
    return {
        "candidate_id": f"{_layer_token(layer_id)}-{mask_type}-{artifact_scope}-001",
        "layer_id": layer_id,
        "mask_type": mask_type,
        "method": details.get("method"),
        "parameters": dict(details.get("parameters", {})),
        "soft_mask_file": _relative(soft_path, base_dir),
        "binary_mask_file": _relative(binary_path, base_dir),
        "preview_file": _relative(preview_path, base_dir),
        "run_id": run_id,
        "artifact_scope": artifact_scope,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "soft_coverage": details.get("soft_coverage", 0.0),
        "area_px": details.get("area_px", pixel_count(mask)),
        "coverage_ratio": details.get("coverage_ratio", 0.0),
        "status": "rejected" if rejected else "candidate",
        "requires_review": True,
        "warnings": list(dict.fromkeys([*warnings, *reasons])),
        "rejection_reasons": rejection_reasons,
        "review_reasons": review_reasons,
        "derivation_reasons": list(details.get("derivation_reasons", [])),
        "adjacent_layers": list(adjacent_layers),
        "overlap_purpose": (
            [f"hidden_overlap_under:{value}" for value in adjacent_layers]
            if mask_type == "edge_extension"
            else []
        ),
    }


def _candidate_conflict_labels(
    conflicts: Sequence[Mapping[str, Any]],
    candidate_type: str,
) -> list[str]:
    result: list[str] = []
    for conflict in conflicts:
        candidate_types = conflict.get("candidate_types")
        if isinstance(candidate_types, list) and candidate_type not in candidate_types:
            continue
        conflict_type = str(conflict.get("type"))
        severity = "reject" if conflict.get("severity") == "reject" else "review"
        result.append(f"{severity}:{conflict_type}")
    return list(dict.fromkeys(result))


def _adjacent_layers(
    asset: Mapping[str, Any],
    target: Image.Image,
    all_assets: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Image.Image],
    *,
    radius: int,
) -> tuple[list[str], list[str]]:
    layer_id = str(asset["layer_id"])
    own_order = asset.get("draw_order")
    neighborhood = dilate(target, max(1, radius + 1))
    foreground: list[str] = []
    contradictory: list[str] = []
    for other in all_assets:
        other_id = str(other["layer_id"])
        if other_id == layer_id:
            continue
        other_mask = masks.get(other_id)
        if other_mask is None or pixel_count(mask_intersection(neighborhood, other_mask)) == 0:
            continue
        other_order = other.get("draw_order")
        if (
            isinstance(own_order, int)
            and not isinstance(own_order, bool)
            and isinstance(other_order, int)
            and not isinstance(other_order, bool)
            and other_order > own_order
        ):
            foreground.append(other_id)
        else:
            contradictory.append(other_id)
    return sorted(foreground), sorted(contradictory)


def _canvas_clip_conflict(
    layer_id: str,
    target: Image.Image,
    radius: int,
) -> dict[str, Any] | None:
    bbox = binary(target).getbbox()
    if bbox is None or radius == 0:
        return None
    width, height = target.size
    reaches_boundary = (
        bbox[0] < radius
        or bbox[1] < radius
        or width - bbox[2] < radius
        or height - bbox[3] < radius
    )
    if reaches_boundary:
        return {
            "type": "canvas_clipped",
            "severity": "review",
            "area_px": 0,
            "layers": [layer_id],
            "candidate_types": ["edge_extension"],
            "reason": "requested dilation reaches the canvas boundary",
        }
    return None


def _source_alpha_clip_conflict(
    layer_id: str,
    clipped: Image.Image,
) -> dict[str, Any] | None:
    area = pixel_count(clipped)
    if area == 0:
        return None
    return {
        "type": "source_alpha_abnormal_extension",
        "severity": "review",
        "area_px": area,
        "layers": [layer_id],
        "candidate_types": ["edge_extension"],
        "reason": (
            "raw extension reached source-alpha or excluded regions and was retained "
            "as a review conflict"
        ),
    }


def _front_hair_occluders(
    asset: Mapping[str, Any],
    all_assets: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Image.Image],
) -> tuple[list[str], list[Image.Image]]:
    own_order = asset.get("draw_order")
    names: list[str] = []
    result: list[Image.Image] = []
    for other in all_assets:
        other_id = str(other["layer_id"])
        if other_id == asset.get("layer_id"):
            continue
        role = str(other.get("role", "")).lower().replace("-", "_").replace(" ", "_")
        other_order = other.get("draw_order")
        is_foreground = (
            isinstance(own_order, int)
            and not isinstance(own_order, bool)
            and isinstance(other_order, int)
            and not isinstance(other_order, bool)
            and other_order > own_order
        )
        if role in FRONT_HAIR_ROLES and is_foreground and other_id in masks:
            names.append(other_id)
            result.append(masks[other_id])
    return names, result


def _confidence(asset: Mapping[str, Any], base: float) -> float:
    segmentation = asset.get("segmentation_confidence", 1.0)
    if not isinstance(segmentation, (int, float)) or isinstance(segmentation, bool):
        segmentation = 1.0
    return base * max(0.0, min(1.0, float(segmentation)))


def derive_masks(
    queue: Mapping[str, Any],
    *,
    queue_path: Path,
    base_dir: Path,
    output_dir: Path,
    config: DerivationConfig | None = None,
    run_id: str | None = None,
    layer_ids: set[str] | None = None,
    retain_artifacts: bool = True,
) -> tuple[dict[str, Any], DerivationArtifacts]:
    """Derive review-only mask candidates and return unwritten image artifacts."""
    config = config or DerivationConfig()
    config.validate()
    base_dir = base_dir.resolve()
    queue_path = queue_path.resolve()
    output_dir = output_dir.resolve()
    try:
        queue_ref = queue_path.relative_to(base_dir).as_posix()
        output_dir.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("queue and output-dir must stay inside base-dir") from exc
    canvas = _canvas(queue)
    assets = _assets(queue)
    known_ids = {str(asset["layer_id"]) for asset in assets}
    if layer_ids is not None:
        unknown = layer_ids - known_ids
        if unknown:
            raise ValueError(f"unknown requested layers: {sorted(unknown)}")
        selected_assets = [asset for asset in assets if asset["layer_id"] in layer_ids]
    else:
        selected_assets = assets
    selected_layer_ids = sorted(str(asset["layer_id"]) for asset in selected_assets)
    source_path, source_ref = _source_path(queue, base_dir)
    source = load_rgba(source_path)
    if source.size != canvas:
        raise ValueError(f"source canvas mismatch: {source.size} != {canvas}")
    source_alpha = source.getchannel("A")
    queue_bytes = queue_path.read_bytes()
    queue_sha256 = bytes_sha256(queue_bytes)
    source_sha256 = file_sha256(source_path)
    masks: dict[str, Image.Image] = {}
    target_paths: dict[str, Path] = {}
    target_refs: dict[str, str] = {}
    target_hashes: dict[str, str] = {}
    input_paths = referenced_artifact_paths(queue, base_dir, document_path=queue_path)
    input_paths.add(source_path)
    for asset in assets:
        layer_id = str(asset["layer_id"])
        path, reference = _target_path(asset, base_dir)
        mask = load_soft_mask(path, canvas)
        masks[layer_id] = mask
        target_paths[layer_id] = path
        target_refs[layer_id] = reference
        target_hashes[layer_id] = file_sha256(path)
        input_paths.add(path)
    actual_run_id = run_id or deterministic_run_id(
        project=queue.get("project"),
        queue_sha256=queue_sha256,
        source_sha256=source_sha256,
        target_hashes=target_hashes,
        config=config,
        layer_ids=selected_layer_ids,
    )
    if not actual_run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    artifact_scope = hashlib.sha256(
        "|".join(
            (
                actual_run_id,
                queue_sha256,
                source_sha256,
                canonical_mapping_sha256(dict(sorted(target_hashes.items()))),
                repr(config),
                canonical_mapping_sha256({"selected_layers": selected_layer_ids}),
            )
        ).encode("utf-8")
    ).hexdigest()[:12]

    layer_records: list[dict[str, Any]] = []
    payloads: dict[Path, SpooledTemporaryFile[bytes]] = {}
    output_paths: set[Path] = set()
    layer_inpaint_masks: dict[str, bytes] = {}
    layer_conflict_masks: dict[str, bytes] = {}
    for asset in selected_assets:
        layer_id = str(asset["layer_id"])
        role = str(asset.get("role", ""))
        target = masks[layer_id]
        try:
            protect, protect_details = derive_protect_mask(
                target,
                source_alpha,
                radius_px=config.protect_radius_px,
                role=role,
                min_area_px=config.fine_part_min_area_px,
            )
            foreground, contradictory = _adjacent_layers(
                asset,
                target,
                assets,
                masks,
                radius=config.edge_radius_px,
            )
            other_masks = [
                mask
                for other_id, mask in masks.items()
                if other_id != layer_id and other_id not in foreground
            ]
            edge, clipped, edge_details = derive_edge_extension_mask(
                target,
                source_alpha,
                radius_px=config.edge_radius_px,
                exclusion_masks=other_masks,
            )
            if foreground and pixel_count(edge):
                foreground_neighborhood = mask_union(
                    *(dilate(masks[value], 1) for value in foreground)
                )
                priority_edge = mask_intersection(edge, foreground_neighborhood)
                fallback_edge = edge.point(lambda value: round(value * 0.5), mode="L")
                edge = mask_union(priority_edge, fallback_edge)
                edge_details = {
                    **edge_details,
                    "method": "occluder_aware_target_dilation_ring",
                    "soft_coverage": round(
                        sum(value * count for value, count in enumerate(edge.histogram())) / 255.0,
                        6,
                    ),
                    "area_px": pixel_count(edge),
                    "derivation_reasons": [
                        *edge_details["derivation_reasons"],
                        "foreground_occluder_boundary_prioritized",
                    ],
                }
            occluder_names, occluder_masks = _front_hair_occluders(asset, assets, masks)
            inpaint, inpaint_details = derive_forehead_inpaint_mask(
                target,
                protect,
                occluder_masks,
                role=role,
                expected_region=asset.get("expected_region"),
            )
            candidates = {
                "protect": protect,
                "edge_extension": edge,
                "inpaint": inpaint,
            }
            conflicts, conflict_mask = detect_candidate_conflicts(
                target,
                candidates,
                source_alpha=source_alpha,
                max_area_ratio=config.max_candidate_area_ratio,
                min_island_area_px=config.min_island_area_px,
                layer_id=layer_id,
            )
            clip_conflict = _canvas_clip_conflict(layer_id, target, config.edge_radius_px)
            alpha_conflict = _source_alpha_clip_conflict(layer_id, clipped)
            if clip_conflict:
                conflicts.append(clip_conflict)
            if alpha_conflict:
                conflicts.append(alpha_conflict)
                conflict_mask = mask_union(conflict_mask, clipped)
            if contradictory:
                conflicts.append(
                    {
                        "type": "draw_order_contradiction",
                        "severity": "review",
                        "area_px": 0,
                        "layers": [layer_id, *contradictory],
                        "candidate_types": ["edge_extension"],
                        "reason": "adjacent layers are not in front of the target draw order",
                    }
                )

            records: dict[str, Any] = {}
            details_by_type: dict[str, Mapping[str, Any]] = {
                "protect": protect_details,
                "edge_extension": edge_details,
                "inpaint": inpaint_details,
            }
            confidence_by_type = {
                "protect": _confidence(asset, 0.92),
                "edge_extension": _confidence(asset, 0.72 if foreground else 0.62),
                "inpaint": _confidence(asset, 0.55),
            }
            adjacent_by_type = {
                "protect": [],
                "edge_extension": foreground,
                "inpaint": occluder_names,
            }
            for mask_type, candidate_mask in candidates.items():
                if candidate_mask is None:
                    records[mask_type] = {
                        "status": "unavailable",
                        "reason": inpaint_details.get("reason", "complete_shape_not_estimable"),
                        "requires_review": True,
                        "derivation_reasons": list(inpaint_details.get("derivation_reasons", [])),
                    }
                    continue
                record = _candidate_record(
                    layer_id=layer_id,
                    mask_type=mask_type,
                    mask=candidate_mask,
                    details=details_by_type[mask_type],
                    confidence=confidence_by_type[mask_type],
                    output_dir=output_dir,
                    base_dir=base_dir,
                    run_id=actual_run_id,
                    artifact_scope=artifact_scope,
                    conflict_types=_candidate_conflict_labels(conflicts, mask_type),
                    adjacent_layers=adjacent_by_type[mask_type],
                )
                records[mask_type] = record
                soft_path = resolve_inside_base(
                    base_dir,
                    str(record["soft_mask_file"]),
                    f"{layer_id} {mask_type} soft output",
                )
                binary_path = resolve_inside_base(
                    base_dir,
                    str(record["binary_mask_file"]),
                    f"{layer_id} {mask_type} binary output",
                )
                preview_path = resolve_inside_base(
                    base_dir,
                    str(record["preview_file"]),
                    f"{layer_id} {mask_type} preview output",
                )
                output_paths.update((soft_path, binary_path, preview_path))
                adjacent_mask = (
                    mask_union(*(masks[value] for value in adjacent_by_type[mask_type]))
                    if adjacent_by_type[mask_type]
                    else Image.new("L", canvas, 0)
                )
                soft_image = candidate_mask.convert("L")
                binary_image = binary(candidate_mask, config.binary_threshold)
                preview_image = build_mask_derivation_preview(
                    source,
                    target,
                    candidate_mask,
                    conflict_mask=conflict_mask,
                    adjacent_mask=adjacent_mask,
                    mask_type=mask_type,
                )
                soft_content = _png_bytes(soft_image)
                binary_content = _png_bytes(binary_image)
                preview_content = _png_bytes(preview_image)
                record["soft_mask_sha256"] = bytes_sha256(soft_content)
                record["binary_mask_sha256"] = bytes_sha256(binary_content)
                record["preview_sha256"] = bytes_sha256(preview_content)
                _store_payload(payloads, soft_path, soft_content)
                _store_payload(payloads, binary_path, binary_content)
                _store_payload(payloads, preview_path, preview_content)
            if inpaint is not None:
                layer_inpaint_masks[layer_id] = _png_bytes(inpaint)
            layer_conflict_masks[layer_id] = _png_bytes(conflict_mask)
            layer_confidence = min(
                float(record["confidence"])
                for record in records.values()
                if isinstance(record, Mapping) and "confidence" in record
            )
            layer_records.append(
                {
                    "layer_id": layer_id,
                    "role": role,
                    "side": asset.get("side"),
                    "draw_order": asset.get("draw_order"),
                    "target_mask": {
                        "path": target_refs[layer_id],
                        "sha256": target_hashes[layer_id],
                        "soft_grayscale": True,
                    },
                    "candidates": records,
                    "conflicts": conflicts,
                    "confidence": round(layer_confidence, 6),
                    "requires_review": True,
                    "status": "candidates_created",
                }
            )
        except (OSError, ValueError) as exc:
            layer_records.append(
                {
                    "layer_id": layer_id,
                    "role": role,
                    "side": asset.get("side"),
                    "draw_order": asset.get("draw_order"),
                    "target_mask": {
                        "path": target_refs[layer_id],
                        "sha256": target_hashes[layer_id],
                        "soft_grayscale": True,
                    },
                    "candidates": {},
                    "conflicts": [],
                    "confidence": 0.0,
                    "requires_review": True,
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )

    # Cross-layer inpaint conflicts are reported, never subtracted or silently resolved.
    for first_index, first in enumerate(layer_records):
        first_id = str(first["layer_id"])
        first_content = layer_inpaint_masks.get(first_id)
        if first_content is None:
            continue
        for second in layer_records[first_index + 1 :]:
            second_id = str(second["layer_id"])
            second_content = layer_inpaint_masks.get(second_id)
            if second_content is None:
                continue
            with Image.open(BytesIO(first_content)) as opened_first:
                first_inpaint = opened_first.convert("L")
            with Image.open(BytesIO(second_content)) as opened_second:
                second_inpaint = opened_second.convert("L")
            overlap = mask_intersection(first_inpaint, second_inpaint)
            area = pixel_count(overlap)
            if area:
                record = {
                    "type": "inter_layer_inpaint_overlap",
                    "severity": "review",
                    "area_px": area,
                    "layers": [first["layer_id"], second["layer_id"]],
                    "candidate_types": ["inpaint"],
                    "reason": "inpaint candidates for different layers overlap",
                }
                first["conflicts"].append(record)
                second["conflicts"].append(record)
                with Image.open(BytesIO(layer_conflict_masks[first_id])) as opened:
                    first_conflicts = opened.convert("L")
                with Image.open(BytesIO(layer_conflict_masks[second_id])) as opened:
                    second_conflicts = opened.convert("L")
                layer_conflict_masks[first_id] = _png_bytes(mask_union(first_conflicts, overlap))
                layer_conflict_masks[second_id] = _png_bytes(mask_union(second_conflicts, overlap))

    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for layer in layer_records:
        pair_role = layer.get("role")
        side = layer.get("side")
        if isinstance(pair_role, str) and side in {"L", "R"}:
            paired[pair_role][str(side)] = layer
    for pair in paired.values():
        if set(pair) != {"L", "R"}:
            continue
        left_id = str(pair["L"]["layer_id"])
        right_id = str(pair["R"]["layer_id"])
        warning = symmetry_warning(masks[left_id], masks[right_id])
        if warning:
            warning["layers"] = [left_id, right_id]
            pair["L"]["conflicts"].append(warning)
            pair["R"]["conflicts"].append(warning)

    # Re-render after cross-layer detection so every spatial conflict is visible in previews.
    for layer in layer_records:
        layer_id = str(layer["layer_id"])
        render_records = layer.get("candidates")
        if not isinstance(render_records, Mapping):
            continue
        with Image.open(BytesIO(layer_conflict_masks[layer_id])) as opened:
            conflict_mask = opened.convert("L")
        for mask_type, render_record in render_records.items():
            if not isinstance(render_record, dict) or "candidate_id" not in render_record:
                continue
            soft_path = resolve_inside_base(
                base_dir,
                str(render_record["soft_mask_file"]),
                f"{layer_id} {mask_type} soft output",
            )
            stream = payloads[soft_path]
            stream.seek(0)
            with Image.open(stream) as opened:
                candidate_mask = opened.convert("L")
            adjacent_layers = render_record.get("adjacent_layers", [])
            adjacent_mask = (
                mask_union(*(masks[str(value)] for value in adjacent_layers))
                if isinstance(adjacent_layers, list) and adjacent_layers
                else Image.new("L", canvas, 0)
            )
            preview = build_mask_derivation_preview(
                source,
                masks[layer_id],
                candidate_mask,
                conflict_mask=conflict_mask,
                adjacent_mask=adjacent_mask,
                mask_type=str(mask_type),
            )
            preview_path = resolve_inside_base(
                base_dir,
                str(render_record["preview_file"]),
                f"{layer_id} {mask_type} preview output",
            )
            preview_content = _png_bytes(preview)
            render_record["preview_sha256"] = bytes_sha256(preview_content)
            _store_payload(payloads, preview_path, preview_content)

    candidate_records = [
        candidate
        for layer in layer_records
        for candidate in (
            layer.get("candidates", {}).values()
            if isinstance(layer.get("candidates"), Mapping)
            else []
        )
        if isinstance(candidate, Mapping) and "candidate_id" in candidate
    ]
    unavailable = sum(
        1
        for layer in layer_records
        for candidate in (
            layer.get("candidates", {}).values()
            if isinstance(layer.get("candidates"), Mapping)
            else []
        )
        if isinstance(candidate, Mapping) and candidate.get("status") == "unavailable"
    )
    failed = sum(layer.get("status") == "failed" for layer in layer_records)
    segmentation_runs = sorted(
        {
            str(asset.get("segmentation_run_id"))
            for asset in selected_assets
            if isinstance(asset.get("segmentation_run_id"), str)
        }
    )
    document = {
        "schema_version": 1,
        "project": queue.get("project"),
        "run_id": actual_run_id,
        "canonical_queue": queue_ref,
        "canonical_queue_sha256": queue_sha256,
        "canonical_queue_content_sha256": canonical_mapping_sha256(queue),
        "source_image": source_ref,
        "source_image_sha256": source_sha256,
        "derived_from_segmentation": {
            "present": bool(segmentation_runs),
            "run_ids": segmentation_runs,
        },
        "input_masks": [
            {
                "layer_id": layer_id,
                "path": target_refs[layer_id],
                "sha256": target_hashes[layer_id],
                "purpose": "target_or_context",
            }
            for layer_id in sorted(target_hashes)
        ],
        "canvas": {"width": canvas[0], "height": canvas[1], "origin": [0, 0]},
        "status": "partial_failure" if failed else "completed",
        "configuration": {
            "protect_radius_px": config.protect_radius_px,
            "edge_radius_px": config.edge_radius_px,
            "fine_part_min_area_px": config.fine_part_min_area_px,
            "max_candidate_area_ratio": config.max_candidate_area_ratio,
            "min_island_area_px": config.min_island_area_px,
            "binary_threshold": config.binary_threshold,
        },
        "execution_scope": {
            "selected_layers": selected_layer_ids,
            "all_layers": layer_ids is None,
        },
        "layers": layer_records,
        "summary": {
            "layers_processed": len(layer_records),
            "candidates_created": len(candidate_records),
            "auto_rejected": sum(
                candidate.get("status") == "rejected" for candidate in candidate_records
            ),
            "review_required": sum(layer.get("requires_review") is True for layer in layer_records),
            "unavailable": unavailable,
            "failed": failed,
            "canonical_queue_modified": False,
        },
    }
    returned_payloads = payloads
    if not retain_artifacts:
        for stream in payloads.values():
            stream.close()
        returned_payloads = {}
    return document, DerivationArtifacts(
        payloads=returned_payloads,
        output_paths=frozenset(output_paths),
        input_paths=frozenset(input_paths),
    )


def load_queue(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"queue not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("queue YAML root must be a mapping")
    return data
