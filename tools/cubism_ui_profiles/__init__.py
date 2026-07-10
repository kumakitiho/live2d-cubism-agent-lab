from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CubismUIProfile:
    profile_id: str
    language: str
    version_patterns: tuple[str, ...]
    window_patterns: tuple[str, ...]
    language_markers: tuple[str, ...]
    dialog_patterns: dict[str, tuple[str, ...]]
    model_open_modes: dict[str, tuple[str, ...]]
    preset_labels: dict[str, tuple[str, ...]]
    alpha_edit_patterns: tuple[str, ...]
    confirm_button_patterns: tuple[str, ...]
    save_control_patterns: tuple[str, ...]


@dataclass(frozen=True)
class ProfileSelection:
    profile: CubismUIProfile | None
    candidates: tuple[str, ...]
    reasons: tuple[str, ...]
    failure_code: str | None = None


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a non-empty string array")
    return tuple(value)


def _pattern_map(value: object, field: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, patterns in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = _strings(patterns, f"{field}.{key}")
    return result


def load_profile(path: Path) -> CubismUIProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"profile must be a YAML object: {path}")
    auto_mesh = raw.get("auto_mesh")
    if not isinstance(auto_mesh, dict):
        raise ValueError(f"auto_mesh must be an object: {path}")
    alpha_edit = auto_mesh.get("alpha_edit")
    confirm_button = auto_mesh.get("confirm_button")
    if not isinstance(alpha_edit, dict) or not isinstance(confirm_button, dict):
        raise ValueError(f"auto_mesh controls must be objects: {path}")
    profile = CubismUIProfile(
        profile_id=str(raw.get("profile_id", "")),
        language=str(raw.get("language", "")),
        version_patterns=_strings(raw.get("version_patterns"), "version_patterns"),
        window_patterns=_strings(raw.get("window_patterns"), "window_patterns"),
        language_markers=_strings(raw.get("language_markers"), "language_markers"),
        dialog_patterns=_pattern_map(raw.get("dialog_patterns"), "dialog_patterns"),
        model_open_modes=_pattern_map(raw.get("model_open_modes"), "model_open_modes"),
        preset_labels=_pattern_map(auto_mesh.get("presets"), "auto_mesh.presets"),
        alpha_edit_patterns=_strings(alpha_edit.get("patterns"), "auto_mesh.alpha_edit.patterns"),
        confirm_button_patterns=_strings(
            confirm_button.get("patterns"), "auto_mesh.confirm_button.patterns"
        ),
        save_control_patterns=_strings(raw.get("save_control_patterns"), "save_control_patterns"),
    )
    if not profile.profile_id or not profile.language:
        raise ValueError(f"profile_id and language are required: {path}")
    for pattern in (
        *profile.window_patterns,
        *profile.version_patterns,
        *profile.language_markers,
        *profile.alpha_edit_patterns,
        *profile.confirm_button_patterns,
        *profile.save_control_patterns,
    ):
        re.compile(pattern)
    return profile


def load_profiles(directory: Path = PROFILE_DIR) -> tuple[CubismUIProfile, ...]:
    profiles = tuple(load_profile(path) for path in sorted(directory.glob("*.yaml")))
    identifiers = [profile.profile_id for profile in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Cubism UI profile_id values must be unique")
    return profiles


def get_profile(profile_id: str, directory: Path = PROFILE_DIR) -> CubismUIProfile:
    matches = [profile for profile in load_profiles(directory) if profile.profile_id == profile_id]
    if len(matches) != 1:
        raise ValueError(f"unknown Cubism UI profile: {profile_id}")
    return matches[0]


def select_profile(
    window_title: str,
    observed_labels: list[str] | tuple[str, ...],
    *,
    version_evidence: str | None = None,
    profiles: tuple[CubismUIProfile, ...] | None = None,
) -> ProfileSelection:
    available = profiles or load_profiles()
    evidence = [window_title, *observed_labels]
    window_candidates = tuple(
        profile
        for profile in available
        if any(
            re.search(pattern, window_title, re.IGNORECASE)
            for pattern in profile.window_patterns
        )
    )
    version_value = version_evidence or window_title
    version_candidates = tuple(
        profile
        for profile in window_candidates
        if any(
            re.search(pattern, version_value, re.IGNORECASE)
            for pattern in profile.version_patterns
        )
    )
    if window_candidates and not version_candidates:
        return ProfileSelection(
            None,
            tuple(profile.profile_id for profile in window_candidates),
            (f"no supported profile version matched: {version_value}",),
            "unsupported_version",
        )
    ranked: list[tuple[int, CubismUIProfile, str]] = []
    for profile in version_candidates:
        marker_hits = [
            marker
            for marker in profile.language_markers
            if any(re.search(marker, value, re.IGNORECASE) for value in evidence)
        ]
        if marker_hits:
            ranked.append(
                (len(marker_hits), profile, f"matched {len(marker_hits)} language marker(s)")
            )

    if not ranked:
        candidate_ids = tuple(profile.profile_id for profile in version_candidates)
        reason = (
            "window matched but no language marker was exposed"
            if candidate_ids
            else "no window/profile pattern matched"
        )
        return ProfileSelection(None, candidate_ids, (reason,), "unsupported_language")

    best_score = max(item[0] for item in ranked)
    best = [item for item in ranked if item[0] == best_score]
    if len(best) != 1:
        return ProfileSelection(
            None,
            tuple(item[1].profile_id for item in best),
            ("multiple profiles matched with the same evidence score",),
            "unsupported_language",
        )
    score, profile, reason = best[0]
    return ProfileSelection(profile, (profile.profile_id,), (f"{reason}; score={score}",), None)


def profile_to_public_dict(profile: CubismUIProfile) -> dict[str, Any]:
    return {"profile_id": profile.profile_id, "language": profile.language}
