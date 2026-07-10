from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageChops

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import require_output_suffix, resolve_inside_base


def _canvas(data: Mapping[str, Any]) -> tuple[int, int]:
    canvas = data.get("canvas")
    if not isinstance(canvas, Mapping):
        raise ValueError("segmentation result canvas must be a mapping")
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
        raise ValueError("segmentation result canvas must contain positive width/height")
    return width, height


def _candidate_list(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("segmentation result candidates must be a non-empty list")
    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidates[{index}] must be a mapping")
        candidate_id = candidate.get("candidate_id")
        layer_id = candidate.get("layer_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidates[{index}].candidate_id must be a non-empty string")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        candidate_ids.add(candidate_id)
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError(f"candidates[{index}].layer_id must be a non-empty string")
        for score_field in ("confidence", "stability_score"):
            score = candidate.get(score_field)
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError(f"candidate {candidate_id} {score_field} must be between 0 and 1")
        bbox = candidate.get("bbox_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise ValueError(f"candidate {candidate_id} bbox_xyxy must contain four numbers")
        candidates.append(deepcopy(dict(candidate)))
    return candidates


def _source_path(data: Mapping[str, Any], base_dir: Path) -> Path:
    source = data.get("source_image")
    if not isinstance(source, Mapping):
        raise ValueError("segmentation result source_image must be a mapping")
    value = source.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("segmentation result source_image.path must be a non-empty string")
    return resolve_inside_base(base_dir, value, "segmentation source image")


def _queue_path(data: Mapping[str, Any], base_dir: Path) -> Path:
    value = data.get("asset_generation_queue")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("segmentation result asset_generation_queue must be a non-empty string")
    return resolve_inside_base(base_dir, value, "canonical asset generation queue")


def _candidate_mask_path(candidate: Mapping[str, Any], base_dir: Path) -> Path:
    value = candidate.get("binary_mask_file", candidate.get("mask_file"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate {candidate.get('candidate_id')} mask path is required")
    return resolve_inside_base(base_dir, value, "segmentation candidate mask")


def _load_binary(path: Path, canvas: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"candidate mask not found: {path}")
    with Image.open(path) as opened:
        mask = opened.convert("L")
    if mask.size != canvas:
        raise ValueError(f"mask canvas mismatch: {path}: {mask.size} != {canvas}")
    return mask.point(lambda value: 255 if value > 0 else 0, mode="L")


def _count(binary: Image.Image) -> int:
    return binary.histogram()[255]


def _normalized_bbox(
    candidate: Mapping[str, Any],
    canvas: tuple[int, int],
) -> tuple[float, float, float, float]:
    raw = candidate["bbox_xyxy"]
    assert isinstance(raw, list)
    width, height = canvas
    x1, y1, x2, y2 = (float(value) for value in raw)
    if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError(f"candidate {candidate['candidate_id']} bbox is outside canvas")
    return x1 / width, y1 / height, x2 / width, y2 / height


def _side_score(side: object, bbox: tuple[float, float, float, float]) -> tuple[float, str | None]:
    center_x = (bbox[0] + bbox[2]) / 2
    if side == "L":
        return (1.0 if center_x <= 0.5 else max(0.0, 1.0 - (center_x - 0.5) * 4), None)
    if side == "R":
        return (1.0 if center_x >= 0.5 else max(0.0, 1.0 - (0.5 - center_x) * 4), None)
    if side == "C":
        return max(0.0, 1.0 - abs(center_x - 0.5) * 3), None
    return 0.5, "ambiguous_side"


def _region_score(
    expected: object,
    bbox: tuple[float, float, float, float],
) -> tuple[float, str | None]:
    if not isinstance(expected, Mapping):
        return 0.5, None
    try:
        x_min = float(expected["x_min"])
        y_min = float(expected["y_min"])
        x_max = float(expected["x_max"])
        y_max = float(expected["y_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("expected_region must contain normalized x/y bounds") from exc
    intersection_width = max(0.0, min(bbox[2], x_max) - max(bbox[0], x_min))
    intersection_height = max(0.0, min(bbox[3], y_max) - max(bbox[1], y_min))
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    containment = intersection_width * intersection_height / bbox_area
    return (
        containment,
        None if containment >= 0.8 else "outside_expected_region",
    )


def _taxonomy_score(candidate: Mapping[str, Any]) -> float:
    semantic = str(candidate.get("semantic_prompt", "")).lower().replace("_", " ")
    role = str(candidate.get("role", "")).lower().replace("_", " ")
    layer = str(candidate.get("layer_id", "")).lower().replace("_", " ")
    role_tokens = {token for token in role.split() if token not in {"l", "r", "c"}}
    if not role_tokens:
        return 0.5
    observed = set(semantic.split()) | set(layer.split())
    return len(role_tokens & observed) / len(role_tokens)


def _initial_score(
    candidate: dict[str, Any],
    *,
    canvas: tuple[int, int],
    binary: Image.Image | None,
    source_alpha: Image.Image | None,
) -> tuple[float, list[str], dict[str, Any]]:
    confidence = float(candidate["confidence"])
    stability = float(candidate["stability_score"])
    bbox = _normalized_bbox(candidate, canvas)
    side_score, side_reason = _side_score(candidate.get("side"), bbox)
    region_score, region_reason = _region_score(candidate.get("expected_region"), bbox)
    taxonomy_score = _taxonomy_score(candidate)
    draw_order = candidate.get("draw_order")
    draw_order_score = (
        1.0
        if isinstance(draw_order, int) and not isinstance(draw_order, bool) and draw_order > 0
        else 0.0
    )
    reasons = list(candidate.get("rejection_reasons", []))
    reasons = [str(reason) for reason in reasons]
    if confidence < 0.5:
        reasons.append("low_confidence")
    if side_reason:
        reasons.append(side_reason)
    if side_score < 0.5:
        reasons.append("side_position_mismatch")
    if region_reason:
        reasons.append(region_reason)
    if draw_order_score == 0.0:
        reasons.append("invalid_draw_order")

    canvas_area = canvas[0] * canvas[1]
    recorded_area = candidate.get("area_px")
    area = int(recorded_area) if isinstance(recorded_area, int) else 0
    source_overlap = 0.5
    containment = region_score
    if binary is not None:
        area = _count(binary)
        if isinstance(recorded_area, int) and recorded_area != area:
            reasons.append("recorded_area_mismatch")
        if source_alpha is not None and area:
            intersection = ImageChops.multiply(binary, source_alpha)
            source_overlap = _count(intersection) / area
            if source_overlap < 0.95:
                reasons.append("mask_outside_source_alpha")
    area_ratio = area / canvas_area
    if area_ratio < 0.0001 or area_ratio > 0.75:
        reasons.append("abnormal_area")
    area_score = max(0.0, 1.0 - min(abs(math.log10(max(area_ratio, 1e-6)) + 1.2) / 5, 1))
    score = (
        confidence * 0.29
        + stability * 0.17
        + side_score * 0.14
        + region_score * 0.10
        + source_overlap * 0.12
        + containment * 0.06
        + area_score * 0.05
        + taxonomy_score * 0.05
        + draw_order_score * 0.02
    )
    metrics = {
        "side_score": round(side_score, 6),
        "expected_region_score": round(region_score, 6),
        "source_alpha_overlap": round(source_overlap, 6),
        "containment": round(containment, 6),
        "area_ratio": round(area_ratio, 8),
        "taxonomy_score": round(taxonomy_score, 6),
        "draw_order_score": draw_order_score,
        "conflict_overlap": 0.0,
        "symmetry_score": 1.0,
    }
    return score, list(dict.fromkeys(reasons)), metrics


def _apply_conflicts(
    candidates: list[dict[str, Any]],
    binaries: Mapping[str, Image.Image],
) -> None:
    for index, first in enumerate(candidates):
        first_id = str(first["candidate_id"])
        first_binary = binaries.get(first_id)
        if first_binary is None:
            continue
        first_area = _count(first_binary)
        for second in candidates[index + 1 :]:
            if second.get("layer_id") == first.get("layer_id"):
                continue
            second_id = str(second["candidate_id"])
            second_binary = binaries.get(second_id)
            if second_binary is None:
                continue
            denominator = min(first_area, _count(second_binary))
            if not denominator:
                continue
            overlap = _count(ImageChops.multiply(first_binary, second_binary)) / denominator
            for candidate in (first, second):
                metrics = candidate["ranking_metrics"]
                assert isinstance(metrics, dict)
                metrics["conflict_overlap"] = round(
                    max(float(metrics["conflict_overlap"]), overlap),
                    6,
                )
                if overlap > 0.65:
                    reasons = candidate["rejection_reasons"]
                    assert isinstance(reasons, list)
                    reasons.append("candidate_conflict")
                    candidate["ranking_score"] = max(
                        0.0,
                        float(candidate["ranking_score"]) - 0.12,
                    )


def _apply_symmetry(candidates: list[dict[str, Any]], canvas: tuple[int, int]) -> None:
    by_role: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        role = candidate.get("role")
        side = candidate.get("side")
        if isinstance(role, str) and side in {"L", "R"}:
            current = by_role[role].get(str(side))
            if current is None or float(candidate["ranking_score"]) > float(
                current["ranking_score"]
            ):
                by_role[role][str(side)] = candidate
    for pair in by_role.values():
        if set(pair) != {"L", "R"}:
            continue
        left = pair["L"]
        right = pair["R"]
        left_box = _normalized_bbox(left, canvas)
        right_box = _normalized_bbox(right, canvas)
        mirror_delta = abs((left_box[0] + left_box[2]) / 2 + (right_box[0] + right_box[2]) / 2 - 1)
        left_area = max(1, int(left.get("area_px", 1)))
        right_area = max(1, int(right.get("area_px", 1)))
        area_ratio = min(left_area, right_area) / max(left_area, right_area)
        symmetry = max(0.0, min(1.0, (1.0 - mirror_delta * 2) * area_ratio))
        for candidate in (left, right):
            metrics = candidate["ranking_metrics"]
            assert isinstance(metrics, dict)
            metrics["symmetry_score"] = round(symmetry, 6)
            if symmetry < 0.6:
                reasons = candidate["rejection_reasons"]
                assert isinstance(reasons, list)
                reasons.append("left_right_asymmetry")
                candidate["ranking_score"] = max(
                    0.0,
                    float(candidate["ranking_score"]) - 0.08,
                )


def rank_candidates(
    data: Mapping[str, Any],
    *,
    base_dir: Path,
    inspect_images: bool = True,
) -> dict[str, Any]:
    if data.get("status") != "completed":
        raise ValueError("only completed segmentation results can be ranked")
    canvas = _canvas(data)
    candidates = _candidate_list(data)
    source_alpha: Image.Image | None = None
    binaries: dict[str, Image.Image] = {}
    if inspect_images:
        source_path = _source_path(data, base_dir)
        if not source_path.is_file():
            raise FileNotFoundError(f"source image not found: {source_path}")
        with Image.open(source_path) as opened:
            source = opened.convert("RGBA")
        if source.size != canvas:
            raise ValueError(f"source canvas mismatch: {source.size} != {canvas}")
        source_alpha = source.getchannel("A").point(
            lambda value: 255 if value > 0 else 0,
            mode="L",
        )
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            binaries[candidate_id] = _load_binary(
                _candidate_mask_path(candidate, base_dir),
                canvas,
            )

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        score, reasons, metrics = _initial_score(
            candidate,
            canvas=canvas,
            binary=binaries.get(candidate_id),
            source_alpha=source_alpha,
        )
        candidate["ranking_score"] = round(score, 6)
        candidate["ranking_metrics"] = metrics
        candidate["rejection_reasons"] = reasons
    _apply_conflicts(candidates, binaries)
    _apply_symmetry(candidates, canvas)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate["rejection_reasons"] = list(
            dict.fromkeys(str(reason) for reason in candidate["rejection_reasons"])
        )
        candidate["requires_review"] = bool(candidate["rejection_reasons"])
        candidate["ranking_score"] = round(float(candidate["ranking_score"]), 6)
        grouped[str(candidate["layer_id"])].append(candidate)
    ranked: list[dict[str, Any]] = []
    for layer_id in sorted(grouped):
        layer_candidates = sorted(
            grouped[layer_id],
            key=lambda item: (
                -float(item["ranking_score"]),
                -float(item["confidence"]),
                str(item["candidate_id"]),
            ),
        )
        for rank, candidate in enumerate(layer_candidates, start=1):
            candidate["rank"] = rank
            ranked.append(candidate)
    result = deepcopy(dict(data))
    result["status"] = "ranked"
    result["candidates"] = ranked
    result["summary"] = {
        "candidate_count": len(ranked),
        "layer_count": len(grouped),
        "needs_review_count": sum(1 for candidate in ranked if candidate["requires_review"]),
        "automatic_assignment": False,
    }
    return result


def _atomic_write_yaml(path: Path, data: Mapping[str, Any], *, force: bool) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "ranked segmentation output")
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
        description="Rank segmentation candidates without selecting one."
    )
    parser.add_argument("segmentation_result", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        input_path = resolve_inside_base(
            base_dir,
            str(args.segmentation_result),
            "segmentation result",
        )
        output = resolve_inside_base(base_dir, str(args.output), "ranked segmentation output")
        if output == input_path:
            raise ValueError("ranked output must not overwrite the segmentation result")
        data = load_yaml_mapping(input_path)
        ranked = rank_candidates(data, base_dir=base_dir, inspect_images=args.execute)
        protected = {
            input_path.resolve(),
            _source_path(data, base_dir).resolve(),
            _queue_path(data, base_dir).resolve(),
        }
        protected.update(
            _candidate_mask_path(candidate, base_dir).resolve()
            for candidate in _candidate_list(data)
        )
        if output.resolve() in protected:
            raise ValueError("ranked output must not overwrite source or candidate masks")
        if args.execute:
            _atomic_write_yaml(output, ranked, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = {
        "status": "written" if args.execute else "planned",
        "output": str(output),
        **ranked["summary"],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
