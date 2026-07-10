from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from PIL import Image

from tools.artifact_validation import ArtifactIssue, load_yaml_mapping

GENERATION_METHODS = (
    "extract",
    "extract_and_edge_repair",
    "transparency_fill",
    "inpaint",
    "redraw",
)
GENERATION_PRIORITY = {method: index for index, method in enumerate(GENERATION_METHODS)}
QUALITY_STATUSES = {"pending", "pass", "warn", "fail"}
QUALITY_CHECKS = {
    "white_halo_px",
    "transparent_hole_px",
    "overlap_deficit_px",
    "source_pixel_difference",
}

ARTIFACT_PATH_FIELDS = (
    "source_file",
    "output_file",
    "target_mask",
    "protect_mask",
    "inpaint_mask",
)


def is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def resolve_inside_base(base_dir: Path, value: str, field: str) -> Path:
    raw = Path(value)
    resolved_base = base_dir.resolve()
    resolved = (raw if raw.is_absolute() else resolved_base / raw).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside base-dir") from exc
    return resolved


def load_rgba(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGBA")


def _load_grayscale_mask(path: Path, canvas: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"mask not found: {path}")
    with Image.open(path) as image:
        mask = image.convert("L")
    if mask.size != canvas:
        raise ValueError(f"mask canvas mismatch: {path}: {mask.size} != {canvas}")
    return mask


def load_soft_mask(path: Path, canvas: tuple[int, int]) -> Image.Image:
    """Load an antialiased grayscale mask without changing its coverage values."""
    return _load_grayscale_mask(path, canvas)


def load_binary_mask(
    path: Path,
    canvas: tuple[int, int],
    *,
    alpha_threshold: int = 1,
) -> Image.Image:
    """Load a mask for boolean membership checks using an explicit alpha threshold."""
    if not is_positive_int(alpha_threshold) or alpha_threshold > 255:
        raise ValueError("alpha_threshold must be an integer from 1 to 255")
    mask = _load_grayscale_mask(path, canvas)
    return mask.point(
        lambda value: 255 if value >= alpha_threshold else 0,
        mode="L",
    )


def load_mask(path: Path, canvas: tuple[int, int]) -> Image.Image:
    """Backward-compatible binary-mask alias; new code must choose soft or binary explicitly."""
    return load_binary_mask(path, canvas, alpha_threshold=1)


def import_parts(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts = data.get("parts")
    if not isinstance(parts, list):
        raise ValueError("parts must be a list")
    result: list[Mapping[str, Any]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise ValueError(f"parts[{index}] must be a mapping")
        if part.get("include_in_import") is True:
            result.append(part)
    return result


def referenced_artifact_paths(
    data: Mapping[str, Any],
    base_dir: Path,
    *,
    document_path: Path | None = None,
) -> set[Path]:
    """Resolve source, part, mask, and canonical derivative paths owned by an artifact."""
    paths: set[Path] = set()
    if document_path is not None:
        paths.add(document_path.resolve())

    source_image = data.get("source_image")
    if isinstance(source_image, Mapping):
        source_value = source_image.get("path")
        if isinstance(source_value, str) and source_value.strip():
            paths.add(resolve_inside_base(base_dir, source_value, "source_image.path"))
    elif isinstance(source_image, str) and source_image.strip():
        paths.add(resolve_inside_base(base_dir, source_image, "source_image"))

    character_spec = data.get("character_spec")
    if isinstance(character_spec, str) and character_spec.strip():
        paths.add(resolve_inside_base(base_dir, character_spec, "character_spec"))

    feedback_inputs = data.get("feedback_inputs")
    if isinstance(feedback_inputs, list):
        for index, value in enumerate(feedback_inputs):
            if isinstance(value, str) and value.strip():
                paths.add(
                    resolve_inside_base(base_dir, value, f"feedback_inputs[{index}]")
                )

    for collection_name in ("parts", "assets"):
        collection = data.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, Mapping):
                continue
            for field in ARTIFACT_PATH_FIELDS:
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    paths.add(
                        resolve_inside_base(
                            base_dir,
                            value,
                            f"{collection_name}[{index}].{field}",
                        )
                    )

    derivatives = data.get("derivatives")
    if isinstance(derivatives, Mapping):
        for name, value in derivatives.items():
            if isinstance(value, str) and value.strip():
                paths.add(resolve_inside_base(base_dir, value, f"derivatives.{name}"))
    return paths


def require_output_suffix(path: Path, allowed: set[str], field: str) -> None:
    normalized = {suffix.lower() for suffix in allowed}
    if path.suffix.lower() not in normalized:
        raise ValueError(f"{field} must use one of these suffixes: {sorted(normalized)}")


def mask_manifest_protected_paths(
    manifest: Mapping[str, Any],
    base_dir: Path,
    *,
    manifest_path: Path,
) -> set[Path]:
    """Collect every input/owned path reachable from a mask manifest and its queue."""
    paths = referenced_artifact_paths(
        manifest,
        base_dir,
        document_path=manifest_path,
    )
    derived_from = manifest.get("derived_from")
    if not isinstance(derived_from, Mapping):
        return paths
    queue_ref = derived_from.get("asset_generation_queue")
    if not isinstance(queue_ref, str) or not queue_ref.strip() or queue_ref == "<in-memory>":
        return paths
    queue_path = resolve_inside_base(base_dir, queue_ref, "asset generation queue")
    queue = load_yaml_mapping(queue_path)
    paths.update(
        referenced_artifact_paths(
            queue,
            base_dir,
            document_path=queue_path,
        )
    )
    return paths


def atomic_save_png(
    image: Image.Image,
    path: Path,
    *,
    force: bool = False,
) -> None:
    require_output_suffix(path, {".png"}, "PNG output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require_same_canvas(images: Mapping[str, Image.Image]) -> tuple[int, int]:
    sizes = {name: image.size for name, image in images.items()}
    unique = set(sizes.values())
    if len(unique) != 1:
        raise ValueError(f"canvas mismatch: {sizes}")
    return next(iter(unique))


def manifest_canvas(data: Mapping[str, Any]) -> tuple[int, int]:
    canvas = data.get("canvas")
    if not isinstance(canvas, Mapping):
        raise ValueError("manifest canvas must be a mapping")
    width = canvas.get("width")
    height = canvas.get("height")
    if not is_positive_int(width) or not is_positive_int(height):
        raise ValueError("manifest canvas width/height must be positive integers")
    assert isinstance(width, int) and isinstance(height, int)
    return width, height


def require_manifest_canvas(image: Image.Image, data: Mapping[str, Any], field: str) -> None:
    expected = manifest_canvas(data)
    if image.size != expected:
        raise ValueError(f"{field} canvas mismatch: {image.size} != {expected}")


def find_part(data: Mapping[str, Any], layer_id: str) -> Mapping[str, Any]:
    parts = data.get("parts")
    if not isinstance(parts, list):
        raise ValueError("parts must be a list")
    for part in parts:
        if isinstance(part, Mapping) and part.get("layer_id") == layer_id:
            return part
    raise ValueError(f"unknown part: {layer_id}")


def validate_dependency_dag(
    dependencies: Mapping[str, set[str]],
    *,
    path: str,
) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    known = set(dependencies)
    for layer_id, values in dependencies.items():
        for dependency in sorted(values - known):
            issues.append(
                ArtifactIssue(f"{path}.{layer_id}.dependencies", f"unknown part: {dependency}")
            )
    remaining = {
        layer_id: {dependency for dependency in values if dependency in known}
        for layer_id, values in dependencies.items()
    }
    while True:
        ready = {layer_id for layer_id, values in remaining.items() if not values}
        if not ready:
            break
        for layer_id in ready:
            remaining.pop(layer_id)
        for values in remaining.values():
            values.difference_update(ready)
    if remaining:
        issues.append(ArtifactIssue(path, f"dependency cycle detected: {sorted(remaining)}"))
    return issues


def validate_mask_manifest(data: Mapping[str, Any]) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    for key in (
        "schema_version",
        "project",
        "derived_from",
        "source_image",
        "canvas",
        "parts",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))
    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))
    if not isinstance(data.get("project"), str) or not str(data.get("project", "")).strip():
        issues.append(ArtifactIssue("project", "must be a non-empty string"))
    derived_from = data.get("derived_from")
    if not isinstance(derived_from, Mapping):
        issues.append(ArtifactIssue("derived_from", "must be a mapping"))
    else:
        queue_ref = derived_from.get("asset_generation_queue")
        if not isinstance(queue_ref, str) or not queue_ref.strip():
            issues.append(
                ArtifactIssue(
                    "derived_from.asset_generation_queue",
                    "must be a non-empty string",
                )
            )
        if not is_positive_int(derived_from.get("queue_schema_version")):
            issues.append(
                ArtifactIssue(
                    "derived_from.queue_schema_version",
                    "must be a positive integer",
                )
            )
    source = data.get("source_image")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        issues.append(ArtifactIssue("source_image.path", "must be a non-empty string"))
    canvas = data.get("canvas")
    if not isinstance(canvas, Mapping):
        issues.append(ArtifactIssue("canvas", "must be a mapping"))
    else:
        for key in ("width", "height"):
            if not is_positive_int(canvas.get(key)):
                issues.append(ArtifactIssue(f"canvas.{key}", "must be a positive integer"))

    parts = data.get("parts")
    if not isinstance(parts, list) or not parts:
        issues.append(ArtifactIssue("parts", "must be a non-empty list"))
        return issues
    layer_ids: set[str] = set()
    draw_orders: set[int] = set()
    dependency_map: dict[str, set[str]] = {}
    required_fields = (
        "layer_id",
        "target_mask",
        "protect_mask",
        "inpaint_mask",
        "source_file",
        "output_file",
        "generation_method",
        "dependencies",
        "draw_order",
        "overlap_margin_px",
        "quality_status",
        "refinement_attempts",
        "include_in_import",
    )
    for index, part in enumerate(parts):
        base = f"parts[{index}]"
        if not isinstance(part, Mapping):
            issues.append(ArtifactIssue(base, "must be a mapping"))
            continue
        for key in required_fields:
            if key not in part:
                issues.append(ArtifactIssue(f"{base}.{key}", "is required"))
        layer_id = part.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            issues.append(ArtifactIssue(f"{base}.layer_id", "must be a non-empty string"))
            continue
        if layer_id in layer_ids:
            issues.append(ArtifactIssue(f"{base}.layer_id", f"duplicate id: {layer_id}"))
        layer_ids.add(layer_id)
        for key in ("target_mask", "protect_mask", "inpaint_mask", "source_file", "output_file"):
            if not isinstance(part.get(key), str) or not str(part.get(key, "")).strip():
                issues.append(ArtifactIssue(f"{base}.{key}", "must be a non-empty string"))
        if part.get("generation_method") not in GENERATION_METHODS:
            issues.append(
                ArtifactIssue(
                    f"{base}.generation_method",
                    f"must be one of {list(GENERATION_METHODS)}",
                )
            )
        dependencies = part.get("dependencies")
        if not isinstance(dependencies, list) or not all(
            isinstance(value, str) and value.strip() for value in dependencies
        ):
            issues.append(ArtifactIssue(f"{base}.dependencies", "must be a list of strings"))
            dependency_map[layer_id] = set()
        else:
            dependency_map[layer_id] = set(dependencies)
        draw_order = part.get("draw_order")
        if not is_positive_int(draw_order):
            issues.append(ArtifactIssue(f"{base}.draw_order", "must be a positive integer"))
        elif draw_order in draw_orders:
            issues.append(ArtifactIssue(f"{base}.draw_order", f"duplicate order: {draw_order}"))
        else:
            assert isinstance(draw_order, int)
            draw_orders.add(draw_order)
        if not is_non_negative_int(part.get("overlap_margin_px")):
            issues.append(
                ArtifactIssue(f"{base}.overlap_margin_px", "must be a non-negative integer")
            )
        if part.get("quality_status") not in QUALITY_STATUSES:
            issues.append(
                ArtifactIssue(
                    f"{base}.quality_status",
                    f"must be one of {sorted(QUALITY_STATUSES)}",
                )
            )
        if not is_non_negative_int(part.get("refinement_attempts")):
            issues.append(
                ArtifactIssue(f"{base}.refinement_attempts", "must be a non-negative integer")
            )
        if not isinstance(part.get("include_in_import"), bool):
            issues.append(ArtifactIssue(f"{base}.include_in_import", "must be boolean"))
    issues.extend(validate_dependency_dag(dependency_map, path="parts"))
    return issues


def validate_asset_quality(data: Mapping[str, Any]) -> list[ArtifactIssue]:
    issues: list[ArtifactIssue] = []
    for key in (
        "schema_version",
        "project",
        "derived_from",
        "source_image",
        "reconstructed_source",
        "difference_image",
        "parts",
        "thresholds",
        "summary",
    ):
        if key not in data:
            issues.append(ArtifactIssue(key, "is required"))
    if data.get("schema_version") != 1:
        issues.append(ArtifactIssue("schema_version", "must equal 1"))
    if not isinstance(data.get("project"), str) or not str(data.get("project", "")).strip():
        issues.append(ArtifactIssue("project", "must be a non-empty string"))
    for key in ("source_image", "reconstructed_source", "difference_image"):
        if not isinstance(data.get(key), str) or not str(data.get(key, "")).strip():
            issues.append(ArtifactIssue(key, "must be a non-empty string"))
    derived_from = data.get("derived_from")
    if not isinstance(derived_from, Mapping):
        issues.append(ArtifactIssue("derived_from", "must be a mapping"))
    else:
        for key in ("mask_manifest", "asset_generation_queue"):
            if not isinstance(derived_from.get(key), str) or not str(
                derived_from.get(key, "")
            ).strip():
                issues.append(ArtifactIssue(f"derived_from.{key}", "must be a non-empty string"))
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, Mapping):
        issues.append(ArtifactIssue("thresholds", "must be a mapping"))
        max_global_difference = 0.0
    else:
        max_global_difference_value = thresholds.get("max_global_difference_score")
        if (
            not isinstance(max_global_difference_value, (int, float))
            or isinstance(max_global_difference_value, bool)
            or not 0 <= max_global_difference_value <= 1
        ):
            issues.append(
                ArtifactIssue(
                    "thresholds.max_global_difference_score",
                    "must be between 0 and 1",
                )
            )
            max_global_difference = 0.0
        else:
            max_global_difference = float(max_global_difference_value)
    parts = data.get("parts")
    if not isinstance(parts, list):
        issues.append(ArtifactIssue("parts", "must be a list"))
        return issues
    seen: set[str] = set()
    failed_count = 0
    for index, part in enumerate(parts):
        base = f"parts[{index}]"
        if not isinstance(part, Mapping):
            issues.append(ArtifactIssue(base, "must be a mapping"))
            continue
        layer_id = part.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            issues.append(ArtifactIssue(f"{base}.layer_id", "must be a non-empty string"))
        elif layer_id in seen:
            issues.append(ArtifactIssue(f"{base}.layer_id", f"duplicate id: {layer_id}"))
        else:
            seen.add(layer_id)
        status = part.get("quality_status")
        if status not in {"pass", "fail"}:
            issues.append(ArtifactIssue(f"{base}.quality_status", "must be pass or fail"))
        elif status == "fail":
            failed_count += 1
        metrics = part.get("metrics")
        if not isinstance(metrics, Mapping):
            issues.append(ArtifactIssue(f"{base}.metrics", "must be a mapping"))
        else:
            for key in (
                "white_halo_px",
                "transparent_hole_px",
                "overlap_deficit_px",
                "difference_score",
            ):
                value = metrics.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    issues.append(ArtifactIssue(f"{base}.metrics.{key}", "must be non-negative"))
        failed_checks = part.get("failed_checks")
        if not isinstance(failed_checks, list) or not all(
            isinstance(value, str) and value in QUALITY_CHECKS for value in failed_checks
        ):
            issues.append(
                ArtifactIssue(
                    f"{base}.failed_checks",
                    f"must only contain {sorted(QUALITY_CHECKS)}",
                )
            )
        elif isinstance(metrics, Mapping):
            expected_checks = {
                check
                for metric, check in (
                    ("white_halo_px", "white_halo_px"),
                    ("transparent_hole_px", "transparent_hole_px"),
                    ("overlap_deficit_px", "overlap_deficit_px"),
                    ("difference_score", "source_pixel_difference"),
                )
                if isinstance(metrics.get(metric), (int, float))
                and not isinstance(metrics.get(metric), bool)
                and metrics.get(metric, 0) > 0
            }
            actual_checks = set(failed_checks)
            if len(actual_checks) != len(failed_checks) or actual_checks != expected_checks:
                issues.append(
                    ArtifactIssue(
                        f"{base}.failed_checks",
                        f"must exactly match non-zero metrics: {sorted(expected_checks)}",
                    )
                )
            expected_status = "fail" if expected_checks else "pass"
            if status != expected_status:
                issues.append(
                    ArtifactIssue(
                        f"{base}.quality_status",
                        f"must equal {expected_status} for the recorded metrics",
                    )
                )
    summary = data.get("summary")
    if not isinstance(summary, Mapping):
        issues.append(ArtifactIssue("summary", "must be a mapping"))
    else:
        if summary.get("total_parts") != len(parts):
            issues.append(ArtifactIssue("summary.total_parts", f"must equal {len(parts)}"))
        if summary.get("failed_parts") != failed_count:
            issues.append(ArtifactIssue("summary.failed_parts", f"must equal {failed_count}"))
        difference_value = summary.get("global_difference_score")
        global_failed = isinstance(difference_value, (int, float)) and (
            not isinstance(difference_value, bool)
            and difference_value > max_global_difference
        )
        expected_result = "fail" if failed_count or global_failed else "pass"
        if summary.get("result") != expected_result:
            issues.append(ArtifactIssue("summary.result", f"must equal {expected_result}"))
        if (
            not isinstance(difference_value, (int, float))
            or isinstance(difference_value, bool)
            or not 0 <= difference_value <= 1
        ):
            issues.append(
                ArtifactIssue("summary.global_difference_score", "must be between 0 and 1")
            )
    return issues


def load_and_validate_mask_manifest(
    path: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    root = (base_dir or Path.cwd()).resolve()
    manifest_path = resolve_inside_base(root, str(path), "mask manifest")
    data = load_yaml_mapping(manifest_path)
    issues = validate_mask_manifest(data)
    if issues:
        raise ValueError("; ".join(issue.format() for issue in issues))
    derived_from = data.get("derived_from")
    if isinstance(derived_from, Mapping):
        queue_ref = derived_from.get("asset_generation_queue")
        if isinstance(queue_ref, str) and queue_ref != "<in-memory>":
            queue_path = resolve_inside_base(root, queue_ref, "asset generation queue")
            queue = load_yaml_mapping(queue_path)
            from tools.mask_candidate_generator import build_mask_manifest

            expected = build_mask_manifest(queue, queue_ref=queue_ref)
            if data != expected:
                raise ValueError("mask manifest is stale or differs from its canonical queue")
    return data


def write_yaml(path: Path, data: Mapping[str, Any], *, force: bool = False) -> None:
    require_output_suffix(path, {".yaml", ".yml"}, "YAML output")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
