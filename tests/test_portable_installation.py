from __future__ import annotations

import os
import sys
from pathlib import Path

import setup_missalign_project
from convertMissalignment import __version__
from convertMissalignment.cli import normalise_setup_arguments


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_version_is_0_1_10() -> None:
    assert __version__ == "0.1.10"


def test_legacy_translation_condition_is_normalised() -> None:
    assert normalise_setup_arguments(["--condition", "translation"]) == [
        "--condition",
        "raw_xf_translation",
    ]
    assert normalise_setup_arguments(["--condition=translation"]) == [
        "--condition=raw_xf_translation"
    ]


def test_runtime_defaults_do_not_contain_personal_paths() -> None:
    assert setup_missalign_project.DEFAULT_ENV == ""
    forbidden = (
        "/gpfs/cssb/user/maiorcam/software/envs/missalign",
        "/gpfs/cssb/user/mjoensso",
        "/gpfs/cssb/user/hellertj",
    )
    public_runtime_files = [
        ROOT / "setup_missalign_project.py",
        ROOT / "setup_warp_project.py",
        ROOT / "scripts" / "runtime_env.py",
        ROOT / "scripts" / "setup_missalign_inputs.sh",
        ROOT / "config" / "cluster_profiles.toml",
        ROOT / "config" / "project_settings.example.toml",
    ]
    for path in public_runtime_files:
        content = path.read_text(encoding="utf-8")
        assert not any(value in content for value in forbidden), path


def test_default_environment_falls_back_to_active_prefix(monkeypatch) -> None:
    monkeypatch.delenv("MISSALIGN_ENV", raising=False)
    profile = {"missalign_environment": ""}
    selected = (
        None
        or os.environ.get("MISSALIGN_ENV")
        or str(profile.get("missalign_environment") or "").strip()
        or sys.prefix
    )
    assert selected == sys.prefix
