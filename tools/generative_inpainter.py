from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from PIL import Image, ImageChops

from tools.artifact_validation import load_yaml_mapping
from tools.asset_pipeline_common import (
    atomic_save_png,
    load_binary_mask,
    load_rgba,
    load_soft_mask,
    require_output_suffix,
    resolve_inside_base,
)
from tools.asset_quality_evaluator import (
    DEFAULT_QUALITY_THRESHOLDS,
    difference_score,
    evaluate_part,
    premultiplied_difference_image,
)
from tools.backends.inpainting import create_backend
from tools.backends.inpainting.base import BackendUnavailableError, model_size
from tools.inpainting_preview import build_inpainting_preview

REQUEST_FIELDS = (
    "schema_version",
    "project",
    "run_id",
    "layer_id",
    "source_image",
    "current_part",
    "target_mask",
    "protect_mask",
    "edge_extension_mask",
    "inpaint_mask",
    "prompt",
    "negative_prompt",
    "backend",
    "backend_config",
    "candidate_count",
    "seed_policy",
    "output_dir",
)
CANDIDATE_FIELDS = (
    "candidate_id",
    "output_file",
    "preview_file",
    "output_sha256",
    "preview_sha256",
    "seed",
    "backend",
    "model_id",
    "model_revision",
    "inference_steps",
    "guidance_scale",
    "strength",
    "crop_box",
    "padding",
    "resize_from",
    "resize_to",
    "quality_metrics",
    "quality_status",
    "requires_review",
    "rejection_reasons",
)
NUMERIC_QUALITY_METRICS = (
    "white_halo_px",
    "transparent_hole_px",
    "overlap_deficit_px",
    "preserve_region_difference_score",
    "edge_extension_difference_score",
    "inpaint_region_source_difference_score",
    "inpaint_outside_difference_score",
    "edge_continuity_score",
    "boundary_color_difference_score",
    "visual_reconstruction_difference_score",
    "protect_difference_px",
    "inpaint_outside_difference_px",
    "alpha_continuity_score",
    "surrounding_palette_consistency_score",
    "visual_reconstruction_score",
)


def validate_inpainting_request(data: Mapping[str, Any]) -> list[str]:
    errors = [f"{field} is required" for field in REQUEST_FIELDS if field not in data]
    if data.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    for field in (
        "project",
        "run_id",
        "layer_id",
        "source_image",
        "current_part",
        "target_mask",
        "protect_mask",
        "edge_extension_mask",
        "inpaint_mask",
        "prompt",
        "backend",
        "output_dir",
    ):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(data.get("negative_prompt"), str):
        errors.append("negative_prompt must be a string")
    if not isinstance(data.get("backend_config"), Mapping):
        errors.append("backend_config must be a mapping")
    count = data.get("candidate_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        errors.append("candidate_count must be a positive integer")
    seed_policy = data.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        errors.append("seed_policy must be a mapping")
    else:
        mode = seed_policy.get("mode")
        if mode == "explicit_list":
            seeds = seed_policy.get("seeds")
            if not isinstance(seeds, list) or not seeds or not all(
                isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds
            ):
                errors.append("seed_policy.seeds must be a non-empty list of non-negative integers")
            elif isinstance(count, int) and len(seeds) < count:
                errors.append("seed_policy.seeds must contain at least candidate_count values")
        elif mode == "increment":
            seed = seed_policy.get("base_seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                errors.append("seed_policy.base_seed must be a non-negative integer")
        else:
            errors.append("seed_policy.mode must be explicit_list or increment")
    path_fields = (
        "source_image",
        "current_part",
        "target_mask",
        "protect_mask",
        "edge_extension_mask",
        "inpaint_mask",
    )
    for field in path_fields:
        value = data.get(field)
        if isinstance(value, str) and Path(value).suffix.lower() != ".png":
            errors.append(f"{field} must be a PNG path")
    if data.get("backend") not in {"mock", "diffusers", "flux_fill"}:
        errors.append("backend must be mock, diffusers, or flux_fill")
    return errors


def resolve_seeds(seed_policy: Mapping[str, Any], candidate_count: int) -> list[int]:
    if seed_policy.get("mode") == "explicit_list":
        seeds = seed_policy.get("seeds")
        if not isinstance(seeds, list):
            raise ValueError("seed_policy.seeds must be a list")
        if len(seeds) < candidate_count:
            raise ValueError("seed_policy.seeds must contain at least candidate_count values")
        return [int(seed) for seed in seeds[:candidate_count]]
    base_seed = seed_policy.get("base_seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("seed_policy.base_seed must be a non-negative integer")
    return [base_seed + index for index in range(candidate_count)]


def inpaint_crop_box(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError("backend_config.padding must be a non-negative integer")
    bbox = mask.convert("L").getbbox()
    if bbox is None:
        raise ValueError("inpaint_mask must contain at least one editable pixel")
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(mask.width, right + padding),
        min(mask.height, bottom + padding),
    )


def composite_generated_crop(
    baseline: Image.Image,
    generated_crop: Image.Image,
    inpaint_mask: Image.Image,
    protect_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> Image.Image:
    baseline_rgba = baseline.convert("RGBA")
    soft_inpaint = inpaint_mask.convert("L")
    protect = protect_mask.convert("L")
    if soft_inpaint.size != baseline_rgba.size or protect.size != baseline_rgba.size:
        raise ValueError("baseline and masks must use the same canvas")
    expected_crop_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    generated = generated_crop.convert("RGBA")
    if generated.size != expected_crop_size:
        raise ValueError("generated crop was not restored to its original crop size")
    generated_canvas = baseline_rgba.copy()
    generated_canvas.paste(generated, (crop_box[0], crop_box[1]))
    editable_soft_mask = ImageChops.subtract(soft_inpaint, protect)
    candidate = Image.composite(generated_canvas, baseline_rgba, editable_soft_mask)
    return Image.composite(baseline_rgba, candidate, protect)


def _difference_pixel_count(
    reference: Image.Image, candidate: Image.Image, mask: Image.Image
) -> int:
    difference = premultiplied_difference_image(reference, candidate)
    binary = mask.convert("L").point(lambda value: 255 if value else 0, mode="L")
    combined = Image.new("L", reference.size, 0)
    for channel in difference.split():
        combined = ImageChops.lighter(combined, ImageChops.multiply(channel, binary))
    return sum(combined.point(lambda value: 255 if value else 0, mode="L").histogram()[1:])


def _context_continuity_metrics(
    candidate: Image.Image,
    inpaint_mask: Image.Image,
    seam_support_mask: Image.Image,
) -> tuple[float, float]:
    rgba = candidate.convert("RGBA")
    inpaint = inpaint_mask.convert("L")
    support = seam_support_mask.convert("L")
    if rgba.size != inpaint.size or support.size != rgba.size:
        raise ValueError("candidate context masks must use the same canvas")
    bbox = inpaint.getbbox()
    if bbox is None:
        return 0.0, 0.0
    pixels: Any = rgba.load()
    inpaint_pixels: Any = inpaint.load()
    support_pixels: Any = support.load()
    alpha_differences: list[int] = []
    generated_colors: list[tuple[int, int, int]] = []
    support_points: set[tuple[int, int]] = set()
    left, top, right, bottom = bbox
    for y in range(top, bottom):
        for x in range(left, right):
            if inpaint_pixels[x, y] == 0 or pixels[x, y][3] == 0:
                continue
            generated_colors.append(pixels[x, y][:3])
            for ny in range(max(0, y - 1), min(rgba.height, y + 2)):
                for nx in range(max(0, x - 1), min(rgba.width, x + 2)):
                    if support_pixels[nx, ny] == 0:
                        continue
                    support_points.add((nx, ny))
                    alpha_differences.append(abs(pixels[x, y][3] - pixels[nx, ny][3]))
    alpha_continuity = max(alpha_differences, default=0) / 255
    support_colors = [pixels[x, y][:3] for x, y in sorted(support_points)]
    if not generated_colors or not support_colors:
        return alpha_continuity, 0.0

    def mean_color(colors: list[tuple[int, int, int]]) -> tuple[float, float, float]:
        count = len(colors)
        return (
            sum(color[0] for color in colors) / count,
            sum(color[1] for color in colors) / count,
            sum(color[2] for color in colors) / count,
        )

    generated_mean = mean_color(generated_colors)
    support_mean = mean_color(support_colors)
    palette_score = sum(
        abs(generated_mean[channel] - support_mean[channel]) for channel in range(3)
    ) / (3 * 255)
    return alpha_continuity, palette_score


def evaluate_inpainting_candidate(
    candidate: Image.Image,
    baseline: Image.Image,
    target_mask: Image.Image,
    protect_mask: Image.Image,
    edge_extension_mask: Image.Image,
    inpaint_mask: Image.Image,
    *,
    source_image: Image.Image | None = None,
    quality_thresholds: Mapping[str, int | float] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if candidate.size != baseline.size:
        return {"canvas_match": False, "origin_match": False}, ["canvas_mismatch"]
    empty_extension = Image.new("L", baseline.size, 0)
    evaluation = evaluate_part(
        candidate,
        baseline,
        target_mask,
        overlap_margin_px=0,
        protect_mask=protect_mask,
        edge_extension_mask=empty_extension,
        inpaint_mask=inpaint_mask,
        reconstructed=baseline,
        thresholds=quality_thresholds,
    )
    metrics = dict(evaluation["metrics"])
    binary_inpaint = inpaint_mask.convert("L").point(
        lambda value: 255 if value else 0, mode="L"
    )
    outside = ImageChops.invert(binary_inpaint)
    strict_outside_score = difference_score(baseline, candidate, outside)
    metrics["inpaint_outside_difference_score"] = strict_outside_score
    metrics["protect_difference_px"] = _difference_pixel_count(
        baseline, candidate, protect_mask
    )
    metrics["inpaint_outside_difference_px"] = _difference_pixel_count(
        baseline, candidate, outside
    )
    metrics["edge_extension_difference_score"] = difference_score(
        baseline, candidate, edge_extension_mask
    )
    seam_support = ImageChops.subtract(
        ImageChops.lighter(
            ImageChops.lighter(target_mask.convert("L"), protect_mask.convert("L")),
            edge_extension_mask.convert("L"),
        ),
        inpaint_mask.convert("L"),
    )
    alpha_continuity, palette_consistency = _context_continuity_metrics(
        candidate, inpaint_mask, seam_support
    )
    metrics["alpha_continuity_score"] = alpha_continuity
    metrics["surrounding_palette_consistency_score"] = palette_consistency
    source_context = (source_image or baseline).convert("RGBA")
    if source_context.size != baseline.size:
        raise ValueError("source_image and candidate must use the same canvas")
    source_overlay_reconstruction = Image.alpha_composite(candidate, source_context)
    visual_mask = ImageChops.lighter(target_mask.convert("L"), inpaint_mask.convert("L"))
    visual_reconstruction_score = difference_score(
        source_context, source_overlay_reconstruction, visual_mask
    )
    metrics["visual_reconstruction_difference_score"] = visual_reconstruction_score
    metrics["visual_reconstruction_score"] = visual_reconstruction_score
    metrics["canvas_match"] = True
    metrics["origin_match"] = True
    metrics["alpha_valid"] = candidate.mode == "RGBA"

    failed = set(evaluation["failed_checks"])
    if strict_outside_score == 0.0:
        failed.discard("inpaint_outside_difference_score")
    else:
        failed.add("inpaint_outside_difference_score")
    visual_threshold = float(
        (quality_thresholds or {}).get(
            "max_visual_reconstruction_difference_score",
            DEFAULT_QUALITY_THRESHOLDS["max_visual_reconstruction_difference_score"],
        )
    )
    if visual_reconstruction_score > visual_threshold:
        failed.add("visual_reconstruction_difference_score")
    else:
        failed.discard("visual_reconstruction_difference_score")
    # This is provenance only and must never become a rejection condition.
    failed.discard("inpaint_region_source_difference_score")
    reason_by_check = {
        "preserve_region_difference_score": "protect_region_changed",
        "inpaint_outside_difference_score": "inpaint_mask_outside_changed",
        "transparent_hole_px": "required_target_coverage_missing",
        "overlap_deficit_px": "required_target_coverage_missing",
        "white_halo_px": "white_halo",
        "edge_continuity_score": "edge_continuity_failure",
        "boundary_color_difference_score": "boundary_color_failure",
        "visual_reconstruction_difference_score": "visual_reconstruction_failure",
        "edge_extension_difference_score": "edge_extension_changed",
    }
    reasons = sorted({reason_by_check.get(check, check) for check in failed})
    if metrics["protect_difference_px"]:
        reasons.append("protect_region_changed")
    if not metrics["alpha_valid"]:
        reasons.append("alpha_invalid")
    return metrics, sorted(set(reasons))


def validate_inpainting_result(data: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "schema_version",
        "project",
        "run_id",
        "layer_id",
        "status",
        "backend",
        "inferred",
        "review_required",
        "canvas",
        "candidates",
        "summary",
    ):
        if field not in data:
            errors.append(f"{field} is required")
    if data.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if data.get("inferred") is not True or data.get("review_required") is not True:
        errors.append("inferred and review_required must be true")
    for field in ("project", "run_id", "layer_id", "backend"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    if data.get("status") != "completed":
        errors.append("status must equal completed")
    canvas = data.get("canvas")
    if not isinstance(canvas, Mapping):
        errors.append("canvas must be a mapping")
    else:
        for field in ("width", "height"):
            value = canvas.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"canvas.{field} must be a positive integer")
        if canvas.get("origin") != [0, 0]:
            errors.append("canvas.origin must equal [0, 0]")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must be a non-empty list")
        return errors
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"candidates[{index}] must be a mapping")
            continue
        for field in CANDIDATE_FIELDS:
            if field not in candidate:
                errors.append(f"candidates[{index}].{field} is required")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            errors.append(f"candidates[{index}].candidate_id must be a non-empty string")
        elif candidate_id in candidate_ids:
            errors.append(f"duplicate candidate_id: {candidate_id}")
        else:
            candidate_ids.add(candidate_id)
        for field in ("output_file", "preview_file"):
            value = candidate.get(field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or Path(value).suffix.lower() != ".png"
            ):
                errors.append(f"candidates[{index}].{field} must be a PNG path")
        for field in ("output_sha256", "preview_sha256"):
            value = candidate.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                errors.append(f"candidates[{index}].{field} must be a lowercase SHA-256")
        seed = candidate.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append(f"candidates[{index}].seed must be a non-negative integer")
        if candidate.get("backend") not in {"mock", "diffusers", "flux_fill"}:
            errors.append(f"candidates[{index}].backend is invalid")
        for field in ("model_id", "model_revision"):
            value = candidate.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"candidates[{index}].{field} must be a string or null")
        inference_steps = candidate.get("inference_steps")
        if (
            not isinstance(inference_steps, int)
            or isinstance(inference_steps, bool)
            or inference_steps <= 0
        ):
            errors.append(f"candidates[{index}].inference_steps must be positive")
        for field in ("guidance_scale", "strength"):
            value = candidate.get(field)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            ):
                errors.append(f"candidates[{index}].{field} must be non-negative or null")
        for field, length, minimum in (
            ("crop_box", 4, 0),
            ("resize_from", 2, 1),
            ("resize_to", 2, 1),
        ):
            value = candidate.get(field)
            if (
                not isinstance(value, list)
                or len(value) != length
                or not all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= minimum
                    for item in value
                )
            ):
                errors.append(
                    f"candidates[{index}].{field} must contain {length} integers >= {minimum}"
                )
        padding = candidate.get("padding")
        if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
            errors.append(f"candidates[{index}].padding must be non-negative")
        metrics = candidate.get("quality_metrics")
        if not isinstance(metrics, Mapping):
            errors.append(f"candidates[{index}].quality_metrics must be a mapping")
        else:
            for metric in NUMERIC_QUALITY_METRICS:
                value = metrics.get(metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    errors.append(
                        f"candidates[{index}].quality_metrics.{metric} must be non-negative"
                    )
            for metric in ("canvas_match", "origin_match", "alpha_valid"):
                if not isinstance(metrics.get(metric), bool):
                    errors.append(
                        f"candidates[{index}].quality_metrics.{metric} must be boolean"
                    )
        quality_status = candidate.get("quality_status")
        if quality_status not in {"pass", "fail"}:
            errors.append(f"candidates[{index}].quality_status must be pass or fail")
        if candidate.get("requires_review") is not True:
            errors.append(f"candidates[{index}].requires_review must be true")
        rejection_reasons = candidate.get("rejection_reasons")
        if not isinstance(rejection_reasons, list) or not all(
            isinstance(reason, str) and reason.strip() for reason in rejection_reasons
        ):
            errors.append(f"candidates[{index}].rejection_reasons must be a list")
        elif len(rejection_reasons) != len(set(rejection_reasons)):
            errors.append(f"candidates[{index}].rejection_reasons must be unique")
        elif (quality_status == "pass") != (rejection_reasons == []):
            errors.append(
                f"candidates[{index}].quality_status must agree with rejection_reasons"
            )
    return errors


def atomic_write_yaml(path: Path, data: Mapping[str, Any], *, force: bool = False) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "YAML output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_staged_files(
    publications: list[tuple[Path, Path]], stage_root: Path
) -> None:
    backup_dir = stage_root / "backups"
    backup_dir.mkdir()
    backups: dict[Path, Path] = {}
    for index, (_, destination) in enumerate(publications):
        if destination.exists():
            backup = backup_dir / f"{index:04d}.backup"
            shutil.copy2(destination, backup)
            backups[destination] = backup
    published: list[Path] = []
    try:
        for staged, destination in publications:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(destination)
            published.append(destination)
    except OSError:
        for destination in reversed(published):
            restore_backup = backups.get(destination)
            if restore_backup is None:
                destination.unlink(missing_ok=True)
            else:
                restore_backup.replace(destination)
        raise


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "candidate"


def _candidate_plan(
    request: Mapping[str, Any], seeds: list[int], output_dir: Path
) -> list[dict[str, Any]]:
    prefix = _safe_identifier(f"{request['run_id']}-{request['layer_id']}")
    planned: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        candidate_id = f"{prefix}-{index:02d}-{seed}"
        planned.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "output_path": output_dir / f"{candidate_id}.png",
                "preview_path": output_dir / f"{candidate_id}.preview.png",
            }
        )
    return planned


def _path_for_artifact(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate source-preserving local inpainting candidates (dry-run by default)."
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--model-id")
    parser.add_argument("--candidate-count", type=int)
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
        request_path = resolve_inside_base(base_dir, str(args.request), "request")
        output_path = resolve_inside_base(base_dir, str(args.output), "result output")
        require_output_suffix(output_path, {".yaml", ".yml"}, "result output")
        request = load_yaml_mapping(request_path)
        errors = validate_inpainting_request(request)
        if errors:
            raise ValueError("; ".join(errors))
        backend_name = args.backend or str(request["backend"])
        backend_config = dict(request["backend_config"])
        if args.model_id is not None:
            backend_config["model_id"] = args.model_id
        candidate_count = (
            args.candidate_count
            if args.candidate_count is not None
            else int(request["candidate_count"])
        )
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        seeds = resolve_seeds(request["seed_policy"], candidate_count)
        backend = create_backend(backend_name)
        status = backend.status()
        output_dir = resolve_inside_base(base_dir, str(request["output_dir"]), "output_dir")
        plans = _candidate_plan(request, seeds, output_dir)
        input_paths = {
            request_path,
            *(resolve_inside_base(base_dir, str(request[field]), field) for field in (
                "source_image",
                "current_part",
                "target_mask",
                "protect_mask",
                "edge_extension_mask",
                "inpaint_mask",
            )),
        }
        output_paths = {output_path}
        for plan in plans:
            output_paths.update((plan["output_path"], plan["preview_path"]))
        if len(output_paths) != 1 + len(plans) * 2:
            raise ValueError("candidate output collision detected")
        if input_paths & output_paths:
            raise ValueError(
                "inpainting outputs must not overwrite request, source, part, or masks"
            )
        symbolic_outputs = sorted(str(path) for path in output_paths if path.is_symlink())
        if symbolic_outputs:
            raise ValueError(f"inpainting outputs must not be symbolic links: {symbolic_outputs}")
        if not args.force:
            existing = sorted(str(path) for path in output_paths if path.exists())
            if existing:
                raise FileExistsError(f"refusing to overwrite without --force: {existing}")

        if not args.execute:
            report = {
                "status": "planned",
                "backend": status.to_dict(),
                "model_load_attempted": False,
                "result_output": str(output_path),
                "candidates": [
                    {
                        "candidate_id": plan["candidate_id"],
                        "seed": plan["seed"],
                        "output_file": str(plan["output_path"]),
                        "preview_file": str(plan["preview_path"]),
                    }
                    for plan in plans
                ],
            }
            print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
            return 0
        if not status.available:
            raise BackendUnavailableError(status.detail)

        source_path = resolve_inside_base(
            base_dir, str(request["source_image"]), "source_image"
        )
        source = load_rgba(source_path)
        current_part_path = resolve_inside_base(
            base_dir, str(request["current_part"]), "current_part"
        )
        baseline = load_rgba(current_part_path)
        if source.size != baseline.size:
            raise ValueError("source_image and current_part canvas/origin mismatch")
        target_mask = load_binary_mask(
            resolve_inside_base(base_dir, str(request["target_mask"]), "target_mask"),
            baseline.size,
        )
        protect_mask = load_binary_mask(
            resolve_inside_base(base_dir, str(request["protect_mask"]), "protect_mask"),
            baseline.size,
        )
        if difference_score(source, baseline, protect_mask) != 0.0:
            raise ValueError(
                "current_part must match source_image exactly inside protect_mask "
                "in premultiplied RGBA"
            )
        edge_extension_mask = load_binary_mask(
            resolve_inside_base(
                base_dir, str(request["edge_extension_mask"]), "edge_extension_mask"
            ),
            baseline.size,
        )
        inpaint_soft = load_soft_mask(
            resolve_inside_base(base_dir, str(request["inpaint_mask"]), "inpaint_mask"),
            baseline.size,
        )
        inpaint_binary = inpaint_soft.point(lambda value: 255 if value else 0, mode="L")
        padding = backend_config.get("padding", 32)
        if not isinstance(padding, int) or isinstance(padding, bool):
            raise ValueError("backend_config.padding must be a non-negative integer")
        crop_box = inpaint_crop_box(inpaint_binary, padding)
        target_size = model_size(backend_config, default=backend.recommended_size)
        crop_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
        resized_input = baseline.crop(crop_box).resize(target_size, Image.Resampling.LANCZOS)
        resized_mask = inpaint_soft.crop(crop_box).resize(target_size, Image.Resampling.BILINEAR)
        quality_thresholds = backend_config.get("quality_thresholds")
        if quality_thresholds is not None and not isinstance(quality_thresholds, Mapping):
            raise ValueError("backend_config.quality_thresholds must be a mapping")
        candidates: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(
            prefix=".inpainting-stage-", dir=base_dir
        ) as temporary_directory:
            stage_root = Path(temporary_directory)
            publications: list[tuple[Path, Path]] = []
            for index, plan in enumerate(plans):
                generated = backend.generate(
                    resized_input,
                    resized_mask,
                    prompt=str(request["prompt"]),
                    negative_prompt=str(request["negative_prompt"]),
                    seed=int(plan["seed"]),
                    config=backend_config,
                )
                if generated.size != target_size:
                    raise ValueError(
                        f"backend returned unexpected size: {generated.size} != {target_size}"
                    )
                restored = generated.resize(crop_size, Image.Resampling.LANCZOS)
                candidate_image = composite_generated_crop(
                    baseline,
                    restored,
                    inpaint_soft,
                    protect_mask,
                    crop_box,
                )
                metrics, rejection_reasons = evaluate_inpainting_candidate(
                    candidate_image,
                    baseline,
                    target_mask,
                    protect_mask,
                    edge_extension_mask,
                    inpaint_binary,
                    source_image=source,
                    quality_thresholds=quality_thresholds,
                )
                staged_candidate = stage_root / f"candidate-{index:04d}.png"
                staged_preview = stage_root / f"preview-{index:04d}.png"
                atomic_save_png(candidate_image, staged_candidate)
                atomic_save_png(
                    build_inpainting_preview(
                        baseline, candidate_image, inpaint_soft, crop_box=crop_box
                    ),
                    staged_preview,
                )
                publications.extend(
                    (
                        (staged_candidate, plan["output_path"]),
                        (staged_preview, plan["preview_path"]),
                    )
                )
                candidates.append(
                    {
                        "candidate_id": plan["candidate_id"],
                        "output_file": _path_for_artifact(plan["output_path"], base_dir),
                        "preview_file": _path_for_artifact(plan["preview_path"], base_dir),
                        "output_sha256": file_sha256(staged_candidate),
                        "preview_sha256": file_sha256(staged_preview),
                        "seed": plan["seed"],
                        "backend": backend.name,
                        "model_id": backend_config.get("model_id"),
                        "model_revision": backend_config.get("model_revision"),
                        "inference_steps": backend_config.get("inference_steps", 30),
                        "guidance_scale": backend_config.get("guidance_scale", 7.5),
                        "strength": (
                            None
                            if backend.name == "flux_fill"
                            else backend_config.get("strength", 1.0)
                        ),
                        "crop_box": list(crop_box),
                        "padding": padding,
                        "resize_from": list(crop_size),
                        "resize_to": list(target_size),
                        "quality_metrics": metrics,
                        "quality_status": "fail" if rejection_reasons else "pass",
                        "requires_review": True,
                        "rejection_reasons": rejection_reasons,
                    }
                )
            passed = sum(
                candidate["quality_status"] == "pass" for candidate in candidates
            )
            result: dict[str, Any] = {
                "schema_version": 1,
                "project": request["project"],
                "run_id": request["run_id"],
                "layer_id": request["layer_id"],
                "status": "completed",
                "derived_from_request": _path_for_artifact(request_path, base_dir),
                "source_image": request["source_image"],
                "current_part": request["current_part"],
                "masks": {
                    "target_mask": request["target_mask"],
                    "protect_mask": request["protect_mask"],
                    "edge_extension_mask": request["edge_extension_mask"],
                    "inpaint_mask": request["inpaint_mask"],
                    "generation_permission": "inpaint_mask_only",
                },
                "backend": backend.name,
                "backend_status": status.to_dict(),
                "backend_config": backend_config,
                "canvas": {
                    "width": baseline.width,
                    "height": baseline.height,
                    "origin": [0, 0],
                },
                "inferred": True,
                "review_required": True,
                "candidates": candidates,
                "summary": {
                    "candidate_count": len(candidates),
                    "passed_candidates": passed,
                    "failed_candidates": len(candidates) - passed,
                    "selection_created": False,
                },
            }
            result_errors = validate_inpainting_result(result)
            if result_errors:
                raise ValueError("invalid inpainting result: " + "; ".join(result_errors))
            staged_result = stage_root / "result.yaml"
            atomic_write_yaml(staged_result, result)
            publications.append((staged_result, output_path))
            _publish_staged_files(publications, stage_root)
    except (
        BackendUnavailableError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = {
        "status": "written",
        "output": str(output_path),
        "candidate_count": len(candidates),
        "passed_candidates": passed,
        "review_required": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
