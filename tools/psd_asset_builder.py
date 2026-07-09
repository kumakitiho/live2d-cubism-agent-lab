from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from tools.asset_manifest_validator import (
    IMPORT_CONSTRAINTS,
    has_file_signature,
    has_supported_image_signature,
    load_asset_manifest,
    resolve_artifact_path,
    validate_asset_manifest,
)


class PsdBackend(Protocol):
    """Future adapter contract for a real PSD-writing implementation."""

    def build(
        self,
        *,
        output_path: Path,
        canvas: Mapping[str, Any],
        layers: Sequence[Mapping[str, Any]],
    ) -> Path: ...


def create_psd_build_plan(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    output_psd: Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    report = validate_asset_manifest(manifest)
    if not report.valid:
        details = "; ".join(issue.format() for issue in report.errors)
        raise ValueError(f"asset manifest is invalid: {details}")

    output = manifest["output"]
    parts = manifest["parts"]
    assert isinstance(output, Mapping)
    assert isinstance(parts, list)
    configured_output = output.get("model_import_psd")
    assert isinstance(configured_output, str)
    artifact_root = (base_dir or Path.cwd()).resolve()
    target = output_psd or Path(configured_output)
    if not target.is_absolute():
        target = artifact_root / target

    import_parts = [
        part
        for part in parts
        if isinstance(part, Mapping) and part.get("include_in_import") is True
    ]
    sorted_parts = sorted(import_parts, key=lambda item: int(item["order"]))
    layers = [
        {
            "layer_id": part["layer_id"],
            "layer_name": part["layer_name"],
            "source_file": part["source_file"],
            "order": part["order"],
            "inferred": part["inferred"],
            "review_required": part["review_required"],
            "readiness": part["readiness"],
        }
        for part in sorted_parts
    ]
    missing_sources = [
        str(layer["source_file"])
        for layer in layers
        if not has_file_signature(
            resolve_artifact_path(artifact_root, str(layer["source_file"])),
            b"\x89PNG\r\n\x1a\n",
        )
    ]
    source_image = manifest["source_image"]
    constraints = manifest["import_constraints"]
    assert isinstance(source_image, Mapping)
    assert isinstance(constraints, Mapping)
    source_path_value = source_image.get("path")
    source_path = (
        resolve_artifact_path(artifact_root, source_path_value)
        if isinstance(source_path_value, str)
        else artifact_root / "<missing-source>"
    )
    source_ready = has_supported_image_signature(source_path)
    all_import_parts_approved = bool(import_parts) and all(
        part.get("readiness") == "approved" for part in import_parts
    )
    required_parts_included = all(
        part.get("required") is not True or part.get("include_in_import") is True
        for part in parts
        if isinstance(part, Mapping)
    )
    constraints_ready = all(constraints.get(key) is True for key in IMPORT_CONSTRAINTS)
    rights_confirmed = source_image.get("rights_status") == "confirmed"
    ready_to_build = (
        report.valid
        and source_ready
        and not missing_sources
        and all_import_parts_approved
        and required_parts_included
        and constraints_ready
        and rights_confirmed
    )
    build_blockers: list[str] = []
    if not source_ready:
        build_blockers.append("source image is missing, empty, or has an invalid signature")
    if missing_sources:
        build_blockers.append("one or more import PNGs are missing or have an invalid signature")
    if not all_import_parts_approved:
        build_blockers.append("all import parts must be approved")
    if not required_parts_included:
        build_blockers.append("all required parts must be included")
    if not constraints_ready:
        build_blockers.append("all import constraints must be true")
    if not rights_confirmed:
        build_blockers.append("source image rights must be confirmed")
    return {
        "schema_version": 1,
        "status": "plan_only",
        "backend": "not_configured",
        "input_manifest": manifest_path.resolve().as_posix(),
        "output_psd": target.as_posix(),
        "canvas": dict(manifest["canvas"]),
        "layer_order": "top_to_bottom",
        "layers": layers,
        "missing_source_files": missing_sources,
        "ready_to_build": ready_to_build,
        "can_build": False,
        "build_blockers": build_blockers,
        "note": "MVP stub: no PSD file was generated. Connect a PsdBackend first.",
    }


def build_psd(plan: Mapping[str, Any], backend: PsdBackend) -> Path:
    """Execute a real backend without allowing the plan-only stub to fake an artifact."""
    if plan.get("ready_to_build") is not True:
        raise RuntimeError("PSD build plan is not ready to build")
    if plan.get("missing_source_files"):
        raise RuntimeError("PSD build plan still has missing source files")
    output_path = Path(str(plan["output_psd"]))
    canvas = plan["canvas"]
    layers = plan["layers"]
    if not isinstance(canvas, Mapping) or not isinstance(layers, list):
        raise ValueError("invalid PSD build plan")
    result = backend.build(output_path=output_path, canvas=canvas, layers=layers)
    if result.resolve() != output_path.resolve():
        raise RuntimeError(
            f"PSD backend returned a different output path: expected {output_path}, got {result}"
        )
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"PSD backend did not create the expected file: {result}")
    if not has_file_signature(result, b"8BPS"):
        raise RuntimeError(f"PSD backend output does not have a PSD signature: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a plan-only PSD layer build description from an asset manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Resolve source and generated part paths from this directory.",
    )
    parser.add_argument("--output-psd", type=Path)
    parser.add_argument(
        "--write-plan",
        type=Path,
        help="Write the YAML build plan. This never writes the PSD itself.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_asset_manifest(args.manifest)
        plan = create_psd_build_plan(
            manifest,
            manifest_path=args.manifest,
            output_psd=args.output_psd,
            base_dir=args.base_dir,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2

    rendered = yaml.safe_dump(plan, allow_unicode=True, sort_keys=False)
    if args.write_plan is None:
        print("DRY-RUN: no PSD or plan file was written")
        print(rendered, end="")
        return 0

    target = args.write_plan.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"WROTE PLAN ONLY: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
