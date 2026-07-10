from __future__ import annotations

import pytest

from tools.cubism_ui_profiles import load_profiles, select_profile


def test_english_profile_selection() -> None:
    selection = select_profile("Live2D Cubism Editor 5", ["File", "Edit", "Save"])
    assert selection.profile is not None
    assert selection.profile.profile_id == "cubism-5-en"


def test_japanese_profile_selection() -> None:
    selection = select_profile("Live2D Cubism Editor 5", ["ファイル", "編集", "保存"])
    assert selection.profile is not None
    assert selection.profile.profile_id == "cubism-5-ja"


def test_profile_mismatch_stops_without_guessing() -> None:
    selection = select_profile("Live2D Cubism Editor 5", ["Archivo", "Editar"])
    assert selection.profile is None
    assert set(selection.candidates) == {"cubism-5-en", "cubism-5-ja"}
    assert "no language marker" in selection.reasons[0]


@pytest.mark.parametrize("version", ["6", "15", "50"])
def test_unsupported_version_stops_before_language_selection(version: str) -> None:
    selection = select_profile(
        f"Live2D Cubism Editor {version}",
        ["File", "Edit"],
        version_evidence=f"CubismEditor{version}.exe Live2D Cubism Editor {version}",
    )
    assert selection.profile is None
    assert selection.failure_code == "unsupported_version"


def test_profiles_have_unique_ids() -> None:
    profiles = load_profiles()
    assert {profile.profile_id for profile in profiles} == {"cubism-5-en", "cubism-5-ja"}
