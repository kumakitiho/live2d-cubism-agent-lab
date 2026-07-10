from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_SCOPES = ("bust_up", "half_body", "full_body")
MOTION_LEVELS = ("minimal", "standard", "expressive")
SUPPORTED_SOURCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Draw order is back-to-front: larger values are composited above smaller values.
ROLE_Z_ORDER = {
    "hair_back_hidden_fill": 1000,
    "hair_back": 1100,
    "leg": 2000,
    "foot": 2100,
    "shoe": 2200,
    "hip_base": 2300,
    "waist_base": 2400,
    "torso_base": 2500,
    "arm": 2600,
    "hand": 2700,
    "clothes_lower": 2800,
    "clothes_upper": 2900,
    "neck_base": 3000,
    "ear": 3100,
    "face_hidden_fill": 3900,
    "face_base": 4000,
    "mouth_inner": 4100,
    "tongue": 4110,
    "teeth_upper": 4120,
    "mouth_base": 4200,
    "mouth_lower": 4210,
    "mouth_upper": 4220,
    "mouth_smile_line": 4230,
    "eye": 4300,
    "eye_white": 4310,
    "eye_iris": 4320,
    "eye_pupil": 4330,
    "eye_highlight": 4340,
    "eyelid_lower": 4350,
    "eyelid_upper": 4360,
    "eyelid_shadow": 4370,
    "eye_closed_line": 4380,
    "brow": 4400,
    "hair_side": 5000,
    "hair_side_tip": 5010,
    "hair_front": 5100,
    "hair_front_tip": 5110,
    "hair_accessory": 5200,
}


@dataclass(frozen=True)
class PartTemplate:
    layer_id: str
    role: str
    side: str = "C"
    generation_method: str = "extract"
    inferred: bool = False
    review_required: bool = False
    required: bool = True
    overlap_margin_px: int = 2


CORE_PARTS = (
    PartTemplate("face_base", "face_base"),
    PartTemplate(
        "face_hidden_fill",
        "face_hidden_fill",
        generation_method="inpaint",
        inferred=True,
        review_required=True,
    ),
    PartTemplate("neck_base", "neck_base"),
    PartTemplate("ear_L", "ear", "L"),
    PartTemplate("ear_R", "ear", "R"),
)

MINIMAL_MOTION_PARTS = (
    PartTemplate("eye_L", "eye", "L"),
    PartTemplate("eye_R", "eye", "R"),
    PartTemplate("brow_L", "brow", "L"),
    PartTemplate("brow_R", "brow", "R"),
    PartTemplate("mouth_base", "mouth"),
    PartTemplate("hair_front", "hair_front"),
    PartTemplate("hair_side_L", "hair_side", "L"),
    PartTemplate("hair_side_R", "hair_side", "R"),
    PartTemplate("hair_back", "hair_back"),
)

STANDARD_MOTION_PARTS = (
    PartTemplate("eye_white_L", "eye_white", "L"),
    PartTemplate("eye_white_R", "eye_white", "R"),
    PartTemplate("eye_iris_L", "eye_iris", "L"),
    PartTemplate("eye_iris_R", "eye_iris", "R"),
    PartTemplate("eye_pupil_L", "eye_pupil", "L"),
    PartTemplate("eye_pupil_R", "eye_pupil", "R"),
    PartTemplate("eye_highlight_L", "eye_highlight", "L"),
    PartTemplate("eye_highlight_R", "eye_highlight", "R"),
    PartTemplate("eyelid_upper_L", "eyelid_upper", "L"),
    PartTemplate("eyelid_upper_R", "eyelid_upper", "R"),
    PartTemplate("eyelid_lower_L", "eyelid_lower", "L"),
    PartTemplate("eyelid_lower_R", "eyelid_lower", "R"),
    PartTemplate("brow_L", "brow", "L"),
    PartTemplate("brow_R", "brow", "R"),
    PartTemplate("mouth_upper", "mouth_upper"),
    PartTemplate("mouth_lower", "mouth_lower"),
    PartTemplate(
        "mouth_inner",
        "mouth_inner",
        generation_method="inpaint",
        inferred=True,
        review_required=True,
    ),
    PartTemplate(
        "teeth_upper",
        "teeth_upper",
        generation_method="inpaint",
        inferred=True,
        review_required=True,
    ),
    PartTemplate(
        "tongue",
        "tongue",
        generation_method="inpaint",
        inferred=True,
        review_required=True,
    ),
    PartTemplate("hair_front_C", "hair_front", "C"),
    PartTemplate("hair_front_L", "hair_front", "L"),
    PartTemplate("hair_front_R", "hair_front", "R"),
    PartTemplate("hair_side_L", "hair_side", "L"),
    PartTemplate("hair_side_R", "hair_side", "R"),
    PartTemplate("hair_back", "hair_back"),
    PartTemplate(
        "hair_back_hidden_fill",
        "hair_back_hidden_fill",
        generation_method="inpaint",
        inferred=True,
        review_required=True,
    ),
)

EXPRESSIVE_EXTRA_PARTS = (
    PartTemplate(
        "eye_closed_line_L",
        "eye_closed_line",
        "L",
        generation_method="redraw",
        inferred=True,
        review_required=True,
    ),
    PartTemplate(
        "eye_closed_line_R",
        "eye_closed_line",
        "R",
        generation_method="redraw",
        inferred=True,
        review_required=True,
    ),
    PartTemplate("eyelid_shadow_L", "eyelid_shadow", "L", review_required=True),
    PartTemplate("eyelid_shadow_R", "eyelid_shadow", "R", review_required=True),
    PartTemplate(
        "mouth_smile_line",
        "mouth_smile_line",
        generation_method="redraw",
        inferred=True,
        review_required=True,
    ),
    PartTemplate("hair_front_tip_L", "hair_front_tip", "L"),
    PartTemplate("hair_front_tip_R", "hair_front_tip", "R"),
    PartTemplate("hair_side_tip_L", "hair_side_tip", "L"),
    PartTemplate("hair_side_tip_R", "hair_side_tip", "R"),
    PartTemplate("ahoge", "hair_accessory", required=False, review_required=True),
)

SCOPE_ADDITIONS: dict[str, tuple[PartTemplate, ...]] = {
    "bust_up": (
        PartTemplate("torso_base", "torso_base"),
        PartTemplate("clothes_upper", "clothes_upper"),
    ),
    "half_body": (
        PartTemplate("waist_base", "waist_base"),
        PartTemplate("arm_L", "arm", "L"),
        PartTemplate("arm_R", "arm", "R"),
        PartTemplate("hand_L", "hand", "L", review_required=True),
        PartTemplate("hand_R", "hand", "R", review_required=True),
        PartTemplate("clothes_lower", "clothes_lower"),
    ),
    "full_body": (
        PartTemplate("hip_base", "hip_base"),
        PartTemplate("leg_L", "leg", "L"),
        PartTemplate("leg_R", "leg", "R"),
        PartTemplate("foot_L", "foot", "L"),
        PartTemplate("foot_R", "foot", "R"),
        PartTemplate("shoe_L", "shoe", "L", required=False),
        PartTemplate("shoe_R", "shoe", "R", required=False),
    ),
}


def validate_source_image(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source image not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
        raise ValueError(
            f"unsupported source image extension: {resolved.suffix}; expected {allowed}"
        )
    return resolved


def _motion_parts(motion_level: str) -> tuple[PartTemplate, ...]:
    if motion_level == "minimal":
        return MINIMAL_MOTION_PARTS
    if motion_level == "standard":
        return STANDARD_MOTION_PARTS
    if motion_level == "expressive":
        return STANDARD_MOTION_PARTS + EXPRESSIVE_EXTRA_PARTS
    raise ValueError(f"unsupported motion_level: {motion_level}")


def _scope_parts(model_scope: str) -> tuple[PartTemplate, ...]:
    if model_scope not in MODEL_SCOPES:
        raise ValueError(f"unsupported model_scope: {model_scope}")
    selected: list[PartTemplate] = []
    for scope in MODEL_SCOPES:
        selected.extend(SCOPE_ADDITIONS[scope])
        if scope == model_scope:
            break
    return tuple(selected)


def _serialize_part(part: PartTemplate, index: int) -> dict[str, Any]:
    layer_id = part.layer_id
    generation_method = part.generation_method
    if generation_method == "extract" and part.overlap_margin_px > 0:
        generation_method = "extract_and_edge_repair"
    instruction = (
        f"Create only the {part.role} part ({part.side}) on a transparent background, "
        "aligned to the source canvas."
    )
    if part.inferred:
        instruction += " Infer hidden pixels conservatively and mark the result for review."
    return {
        "layer_id": layer_id,
        "layer_name": layer_id,
        "role": part.role,
        "side": part.side,
        "generation_method": generation_method,
        "inferred": part.inferred,
        "review_required": part.review_required,
        "required": part.required,
        "source_file": f"generated/parts/{layer_id}.png",
        "target_mask": f"generated/masks/{layer_id}.target.png",
        "protect_mask": f"generated/masks/{layer_id}.protect.png",
        "edge_extension_mask": f"generated/masks/{layer_id}.edge-extension.png",
        "inpaint_mask": f"generated/masks/{layer_id}.inpaint.png",
        "overlap_margin_px": part.overlap_margin_px,
        "prompt_id": f"part-{layer_id}",
        "prompt": {
            "instruction": instruction,
            "preserve": ["character identity", "line style", "palette", "lighting direction"],
            "avoid": [
                "full character regeneration",
                "canvas resize",
                "opaque background",
                "untracked accessories",
            ],
        },
        "draw_order": ROLE_Z_ORDER.get(part.role, 3500) * 100 + index,
    }


def build_material_plan(
    source_image: Path,
    *,
    model_scope: str,
    motion_level: str,
    project: str | None = None,
) -> dict[str, Any]:
    resolved_source = validate_source_image(source_image)
    if model_scope not in MODEL_SCOPES:
        raise ValueError(f"unsupported model_scope: {model_scope}")
    if motion_level not in MOTION_LEVELS:
        raise ValueError(f"unsupported motion_level: {motion_level}")

    templates = CORE_PARTS + _motion_parts(motion_level) + _scope_parts(model_scope)
    layer_ids = [part.layer_id for part in templates]
    if len(layer_ids) != len(set(layer_ids)):
        raise RuntimeError("internal taxonomy contains duplicate layer ids")

    serialized_parts = [_serialize_part(part, index) for index, part in enumerate(templates)]
    serialized_parts.sort(key=lambda part: part["draw_order"])

    return {
        "schema_version": 1,
        "project": project or resolved_source.stem,
        "source_image": {
            "path": resolved_source.as_posix(),
            "rights_status": "needs_confirmation",
        },
        "model_scope": model_scope,
        "motion_level": motion_level,
        "generation_policy": {
            "same_canvas_alignment": True,
            "transparent_parts": True,
            "mark_inferred_hidden_regions": True,
            "review_inferred_before_handoff": True,
        },
        "parts": serialized_parts,
        "deliverables": {
            "asset_manifest": "generated/asset_manifest.yaml",
            "layer_map": "generated/layer_map.yaml",
            "model_import_psd": "generated/model_import.psd",
        },
        "handoff": {
            "target_skill": "live2d-cubism-workflow",
            "status": "blocked_until_assets_are_generated_and_approved",
        },
    }


def dump_plan(plan: dict[str, Any]) -> str:
    return yaml.safe_dump(plan, allow_unicode=True, sort_keys=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic Live2D material-generation plan from one image."
    )
    parser.add_argument("source_image", type=Path)
    parser.add_argument("--model-scope", choices=MODEL_SCOPES, default="bust_up")
    parser.add_argument("--motion-level", choices=MOTION_LEVELS, default="standard")
    parser.add_argument("--project")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/asset_generation_plan.yaml"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the YAML plan. Without this flag, print a dry-run to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_material_plan(
            args.source_image,
            model_scope=args.model_scope,
            motion_level=args.motion_level,
            project=args.project,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    rendered = dump_plan(plan)
    if not args.execute:
        print("DRY-RUN: no files were written")
        print(rendered, end="")
        return 0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
