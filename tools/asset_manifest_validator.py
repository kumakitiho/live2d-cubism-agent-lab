from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RIGHTS_STATUSES = {"confirmed", "needs_confirmation"}
READINESS_STATUSES = {"planned", "generated", "reviewed", "approved", "rejected"}
SIDES = {"L", "R", "C", "none"}
GENERATION_METHODS = {"extract", "mask_extract", "inpaint", "redraw"}
IMPORT_CONSTRAINTS = (
    "unique_layer_names",
    "one_drawable_per_layer",
    "rgb_8bit_srgb",
    "same_canvas_alignment",
    "no_reference_layers",
    "masks_resolved",
    "inferred_assets_reviewed",
)


@dataclass(frozen=True)
class ManifestIssue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True)
class ManifestValidationReport:
    errors: tuple[ManifestIssue, ...]
    warnings: tuple[ManifestIssue, ...]
    checks: tuple[str, ...]
    handoff_ready: bool

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "handoff_ready": self.handoff_ready,
            "errors": [issue.format() for issue in self.errors],
            "warnings": [issue.format() for issue in self.warnings],
            "checks": list(self.checks),
        }


def load_asset_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a mapping")
    return raw


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def resolve_artifact_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _is_non_empty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def has_file_signature(path: Path, signature: bytes) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(len(signature)) == signature
    except OSError:
        return False


def has_supported_image_signature(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return has_file_signature(path, b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return has_file_signature(path, b"\xff\xd8\xff")
    if suffix == ".webp":
        try:
            with path.open("rb") as stream:
                header = stream.read(12)
            return header[:4] == b"RIFF" and header[8:12] == b"WEBP"
        except OSError:
            return False
    return False


def validate_asset_manifest(
    data: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> ManifestValidationReport:
    errors: list[ManifestIssue] = []
    warnings: list[ManifestIssue] = []
    checks: list[str] = []

    for key in (
        "schema_version",
        "project",
        "source_image",
        "canvas",
        "output",
        "import_constraints",
        "parts",
    ):
        if key not in data:
            errors.append(ManifestIssue(key, "is required"))

    if data.get("schema_version") != 1:
        errors.append(ManifestIssue("schema_version", "must equal 1"))
    if not isinstance(data.get("project"), str) or not str(data.get("project", "")).strip():
        errors.append(ManifestIssue("project", "must be a non-empty string"))

    source_image = data.get("source_image")
    rights_confirmed = False
    if not isinstance(source_image, Mapping):
        errors.append(ManifestIssue("source_image", "must be a mapping"))
    else:
        source_path = source_image.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            errors.append(ManifestIssue("source_image.path", "must be a non-empty string"))
        rights_status = source_image.get("rights_status")
        if rights_status not in RIGHTS_STATUSES:
            errors.append(
                ManifestIssue(
                    "source_image.rights_status",
                    "must be confirmed or needs_confirmation",
                )
            )
        elif rights_status == "needs_confirmation":
            warnings.append(
                ManifestIssue(
                    "source_image.rights_status",
                    "rights must be confirmed before asset generation handoff",
                )
            )
        else:
            rights_confirmed = True

    canvas = data.get("canvas")
    canvas_ready = False
    if not isinstance(canvas, Mapping):
        errors.append(ManifestIssue("canvas", "must be a mapping"))
    else:
        for dimension in ("width", "height"):
            if not _is_positive_int(canvas.get(dimension)):
                errors.append(ManifestIssue(f"canvas.{dimension}", "must be a positive integer"))
        expected_values = {"color_mode": "RGBA", "bit_depth": 8, "color_profile": "sRGB"}
        for key, expected in expected_values.items():
            if canvas.get(key) != expected:
                errors.append(ManifestIssue(f"canvas.{key}", f"must equal {expected!r}"))
        canvas_ready = not any(issue.path.startswith("canvas") for issue in errors)

    output = data.get("output")
    output_ready = False
    if not isinstance(output, Mapping):
        errors.append(ManifestIssue("output", "must be a mapping"))
    else:
        model_import_psd = output.get("model_import_psd")
        layer_map = output.get("layer_map")
        if not isinstance(model_import_psd, str) or Path(model_import_psd).suffix.lower() != ".psd":
            errors.append(ManifestIssue("output.model_import_psd", "must be a .psd path"))
        if not isinstance(layer_map, str) or Path(layer_map).suffix.lower() not in {
            ".yaml",
            ".yml",
        }:
            errors.append(ManifestIssue("output.layer_map", "must be a .yaml or .yml path"))
        output_ready = not any(issue.path.startswith("output") for issue in errors)

    constraints = data.get("import_constraints")
    constraints_ready = False
    if not isinstance(constraints, Mapping):
        errors.append(ManifestIssue("import_constraints", "must be a mapping"))
    else:
        constraint_values: list[bool] = []
        for key in IMPORT_CONSTRAINTS:
            value = constraints.get(key)
            if not _is_bool(value):
                errors.append(ManifestIssue(f"import_constraints.{key}", "must be boolean"))
                constraint_values.append(False)
            elif value:
                checks.append(f"PASS: {key}")
                constraint_values.append(True)
            else:
                checks.append(f"PENDING: {key}")
                warnings.append(
                    ManifestIssue(
                        f"import_constraints.{key}",
                        "must be true before Cubism handoff",
                    )
                )
                constraint_values.append(False)
        constraints_ready = all(constraint_values)

    parts = data.get("parts")
    import_parts_ready = False
    inferred_assets_ready = False
    required_parts_included = True
    import_layer_pairs: set[tuple[str, str]] = set()
    import_source_files: list[str] = []
    if not isinstance(parts, list) or not parts:
        errors.append(ManifestIssue("parts", "must be a non-empty list"))
    else:
        layer_ids: list[str] = []
        layer_names: list[str] = []
        layer_orders: list[int] = []
        import_readiness: list[bool] = []
        inferred_readiness: list[bool] = []
        for index, part in enumerate(parts):
            base = f"parts[{index}]"
            if not isinstance(part, Mapping):
                errors.append(ManifestIssue(base, "must be a mapping"))
                continue
            required_fields = (
                "layer_id",
                "layer_name",
                "role",
                "side",
                "source_file",
                "prompt_id",
                "generation_method",
                "order",
                "required",
                "inferred",
                "review_required",
                "readiness",
                "include_in_import",
            )
            for field in required_fields:
                if field not in part:
                    errors.append(ManifestIssue(f"{base}.{field}", "is required"))

            layer_id = part.get("layer_id")
            if isinstance(layer_id, str) and layer_id.strip():
                layer_ids.append(layer_id)
            else:
                errors.append(ManifestIssue(f"{base}.layer_id", "must be a non-empty string"))

            layer_name = part.get("layer_name")
            if isinstance(layer_name, str) and layer_name.strip():
                layer_names.append(layer_name)
            else:
                errors.append(ManifestIssue(f"{base}.layer_name", "must be a non-empty string"))

            if not isinstance(part.get("role"), str) or not str(part.get("role", "")).strip():
                errors.append(ManifestIssue(f"{base}.role", "must be a non-empty string"))
            if part.get("side") not in SIDES:
                errors.append(ManifestIssue(f"{base}.side", "must be L, R, C, or none"))
            if part.get("generation_method") not in GENERATION_METHODS:
                errors.append(
                    ManifestIssue(
                        f"{base}.generation_method",
                        f"must be one of {sorted(GENERATION_METHODS)}",
                    )
                )
            order = part.get("order")
            if isinstance(order, int) and not isinstance(order, bool) and order > 0:
                layer_orders.append(order)
            else:
                errors.append(ManifestIssue(f"{base}.order", "must be a positive integer"))
            prompt_id = part.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                errors.append(ManifestIssue(f"{base}.prompt_id", "must be a non-empty string"))
            for field in ("required", "inferred", "review_required", "include_in_import"):
                if not _is_bool(part.get(field)):
                    errors.append(ManifestIssue(f"{base}.{field}", "must be boolean"))

            readiness = part.get("readiness")
            if readiness not in READINESS_STATUSES:
                errors.append(
                    ManifestIssue(
                        f"{base}.readiness",
                        f"must be one of {sorted(READINESS_STATUSES)}",
                    )
                )

            inferred = part.get("inferred") is True
            review_required = part.get("review_required") is True
            if inferred and not review_required:
                errors.append(
                    ManifestIssue(
                        f"{base}.review_required",
                        "must be true when inferred is true",
                    )
                )
            if part.get("generation_method") == "inpaint" and not inferred:
                errors.append(ManifestIssue(f"{base}.inferred", "must be true for inpaint assets"))
            if part.get("generation_method") == "redraw" and not review_required:
                errors.append(
                    ManifestIssue(
                        f"{base}.review_required",
                        "must be true for redraw assets",
                    )
                )
            if part.get("generation_method") in {"mask_extract", "inpaint"}:
                mask_file = part.get("mask_file")
                if not isinstance(mask_file, str) or Path(mask_file).suffix.lower() != ".png":
                    errors.append(
                        ManifestIssue(
                            f"{base}.mask_file",
                            "mask_extract and inpaint assets must reference a .png mask",
                        )
                    )
            if inferred:
                inferred_readiness.append(review_required and readiness == "approved")

            include_in_import = part.get("include_in_import") is True
            if include_in_import:
                source_file = part.get("source_file")
                if not isinstance(source_file, str) or Path(source_file).suffix.lower() != ".png":
                    errors.append(
                        ManifestIssue(
                            f"{base}.source_file",
                            "import parts must reference a .png source file",
                        )
                    )
                else:
                    import_source_files.append(source_file)
                if (
                    isinstance(layer_id, str)
                    and layer_id.strip()
                    and isinstance(layer_name, str)
                    and layer_name.strip()
                ):
                    import_layer_pairs.add((layer_id, layer_name))
                if part.get("role") in {"source_reference", "guide", "mask"}:
                    errors.append(
                        ManifestIssue(
                            f"{base}.role",
                            "reference, guide, and mask layers cannot be imported",
                        )
                    )
                import_readiness.append(readiness == "approved")
            elif part.get("required") is True:
                required_parts_included = False
                warnings.append(
                    ManifestIssue(base, "required part is excluded from the import PSD")
                )

        duplicate_ids = [key for key, count in Counter(layer_ids).items() if count > 1]
        duplicate_names = [key for key, count in Counter(layer_names).items() if count > 1]
        duplicate_orders = [key for key, count in Counter(layer_orders).items() if count > 1]
        for duplicate in duplicate_ids:
            errors.append(ManifestIssue("parts", f"duplicate layer id: {duplicate}"))
        for duplicate in duplicate_names:
            errors.append(ManifestIssue("parts", f"duplicate layer name: {duplicate}"))
        for duplicate_order in duplicate_orders:
            errors.append(ManifestIssue("parts", f"duplicate layer order: {duplicate_order}"))
        checks.append("PASS: unique layer ids" if not duplicate_ids else "FAIL: unique layer ids")
        checks.append(
            "PASS: unique layer names" if not duplicate_names else "FAIL: unique layer names"
        )
        checks.append(
            "PASS: unique layer order" if not duplicate_orders else "FAIL: unique layer order"
        )
        import_parts_ready = bool(import_readiness) and all(import_readiness)
        inferred_assets_ready = all(inferred_readiness)

    if required_parts_included:
        checks.append("PASS: all required parts are included")
    else:
        checks.append("FAIL: required parts are excluded")

    artifacts_ready = False
    if base_dir is None:
        warnings.append(
            ManifestIssue(
                "handoff.artifacts",
                "artifact existence was not checked because base_dir was not provided",
            )
        )
        checks.append("PENDING: source PNGs, PSD, and layer map existence")
    else:
        root = base_dir.resolve()
        artifacts_ready = True

        source_path_value = source_image.get("path") if isinstance(source_image, Mapping) else None
        source_artifact_path = (
            resolve_artifact_path(root, source_path_value)
            if isinstance(source_path_value, str)
            else None
        )
        if (
            source_artifact_path is None
            or not _is_non_empty_file(source_artifact_path)
            or not has_supported_image_signature(source_artifact_path)
        ):
            artifacts_ready = False
            warnings.append(
                ManifestIssue(
                    "source_image.path",
                    "source image must exist, be non-empty, and match its image signature",
                )
            )

        for index, source_file in enumerate(import_source_files):
            import_source_path = resolve_artifact_path(root, source_file)
            if not _is_non_empty_file(import_source_path) or not has_file_signature(
                import_source_path, b"\x89PNG\r\n\x1a\n"
            ):
                artifacts_ready = False
                warnings.append(
                    ManifestIssue(
                        f"handoff.import_sources[{index}]",
                        "import PNG must exist, be non-empty, and have a PNG signature: "
                        f"{source_file}",
                    )
                )

        psd_value = output.get("model_import_psd") if isinstance(output, Mapping) else None
        psd_path = resolve_artifact_path(root, psd_value) if isinstance(psd_value, str) else None
        if (
            psd_path is None
            or not _is_non_empty_file(psd_path)
            or not has_file_signature(psd_path, b"8BPS")
        ):
            artifacts_ready = False
            warnings.append(
                ManifestIssue(
                    "output.model_import_psd",
                    "model_import.psd must exist, be non-empty, and have a PSD signature",
                )
            )

        layer_map_value = output.get("layer_map") if isinstance(output, Mapping) else None
        layer_map_path = (
            resolve_artifact_path(root, layer_map_value)
            if isinstance(layer_map_value, str)
            else None
        )
        layer_map_data: Any = None
        if layer_map_path is None or not _is_non_empty_file(layer_map_path):
            artifacts_ready = False
            warnings.append(
                ManifestIssue(
                    "output.layer_map",
                    "layer map must exist and be non-empty before handoff",
                )
            )
        else:
            try:
                layer_map_data = yaml.safe_load(layer_map_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                artifacts_ready = False
                warnings.append(ManifestIssue("output.layer_map", f"cannot be read: {exc}"))

        if isinstance(layer_map_data, Mapping):
            if layer_map_data.get("project") != data.get("project"):
                artifacts_ready = False
                warnings.append(
                    ManifestIssue("output.layer_map.project", "must match manifest project")
                )
            map_canvas = layer_map_data.get("canvas")
            if not isinstance(map_canvas, Mapping) or not isinstance(canvas, Mapping):
                artifacts_ready = False
                warnings.append(
                    ManifestIssue("output.layer_map.canvas", "must match manifest canvas")
                )
            else:
                for dimension in ("width", "height"):
                    if map_canvas.get(dimension) != canvas.get(dimension):
                        artifacts_ready = False
                        warnings.append(
                            ManifestIssue(
                                f"output.layer_map.canvas.{dimension}",
                                "must match manifest canvas",
                            )
                        )

            map_layers = layer_map_data.get("layers")
            if not isinstance(map_layers, list):
                artifacts_ready = False
                warnings.append(ManifestIssue("output.layer_map.layers", "must be a list"))
            else:
                map_pairs: list[tuple[str, str]] = []
                malformed_rows = False
                for index, layer in enumerate(map_layers):
                    if not isinstance(layer, Mapping):
                        malformed_rows = True
                        warnings.append(
                            ManifestIssue(
                                f"output.layer_map.layers[{index}]",
                                "must be a mapping with layer_id and name",
                            )
                        )
                        continue
                    map_layer_id = layer.get("layer_id")
                    map_layer_name = layer.get("name")
                    if not isinstance(map_layer_id, str) or not isinstance(map_layer_name, str):
                        malformed_rows = True
                        warnings.append(
                            ManifestIssue(
                                f"output.layer_map.layers[{index}]",
                                "layer_id and name must be strings",
                            )
                        )
                        continue
                    map_pairs.append((map_layer_id, map_layer_name))

                map_ids = [pair[0] for pair in map_pairs]
                map_names = [pair[1] for pair in map_pairs]
                duplicate_map_ids = [key for key, count in Counter(map_ids).items() if count > 1]
                duplicate_map_names = [
                    key for key, count in Counter(map_names).items() if count > 1
                ]
                if malformed_rows or duplicate_map_ids or duplicate_map_names:
                    artifacts_ready = False
                    warnings.append(
                        ManifestIssue(
                            "output.layer_map.layers",
                            "layer IDs and names must be present and unique",
                        )
                    )
                if len(map_pairs) != len(import_layer_pairs):
                    artifacts_ready = False
                    warnings.append(
                        ManifestIssue(
                            "output.layer_map.layers",
                            "layer count must exactly match import manifest parts",
                        )
                    )
                if set(map_pairs) != import_layer_pairs:
                    artifacts_ready = False
                    warnings.append(
                        ManifestIssue(
                            "output.layer_map.layers",
                            "(layer_id, name) pairs must exactly match import manifest parts",
                        )
                    )
        elif layer_map_path is not None and layer_map_path.is_file():
            artifacts_ready = False
            warnings.append(ManifestIssue("output.layer_map", "root must be a mapping"))

        checks.append(
            "PASS: source PNGs, PSD, and layer map artifacts"
            if artifacts_ready
            else "PENDING: source PNGs, PSD, and layer map artifacts"
        )

    handoff_ready = (
        not errors
        and rights_confirmed
        and canvas_ready
        and output_ready
        and constraints_ready
        and import_parts_ready
        and inferred_assets_ready
        and required_parts_included
        and artifacts_ready
    )
    checks.append("PASS: Cubism handoff ready" if handoff_ready else "PENDING: Cubism handoff")
    return ManifestValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=tuple(checks),
        handoff_ready=handoff_ready,
    )


def format_report(report: ManifestValidationReport) -> str:
    lines = [
        f"valid: {str(report.valid).lower()}",
        f"handoff_ready: {str(report.handoff_ready).lower()}",
    ]
    lines.extend(f"ERROR: {issue.format()}" for issue in report.errors)
    lines.extend(f"WARN: {issue.format()}" for issue in report.warnings)
    lines.extend(f"CHECK: {check}" for check in report.checks)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an image-to-Live2D asset manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Resolve source, part, PSD, and layer-map paths from this directory.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    parser.add_argument(
        "--require-handoff-ready",
        action="store_true",
        help="Return a failure status unless all Cubism handoff gates pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_asset_manifest(args.manifest)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = validate_asset_manifest(manifest, base_dir=args.base_dir)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    if not report.valid:
        return 1
    if args.require_handoff_ready and not report.handoff_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
