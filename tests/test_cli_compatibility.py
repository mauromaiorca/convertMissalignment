from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from convertMissalignment import __version__
from convertMissalignment.cli import normalise_setup_arguments
from setup_missalign_project import _normalise_conditions

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_source_version_is_0_1_8() -> None:
    assert __version__ == "0.1.8"
    assert (ROOT / "VERSION").read_text().strip() == "0.1.8"


def test_translation_condition_is_backward_compatible() -> None:
    assert normalise_setup_arguments(["--condition", "translation"]) == [
        "--condition",
        "raw_xf_translation",
    ]
    assert normalise_setup_arguments(["--condition=translation"]) == [
        "--condition=raw_xf_translation"
    ]
    assert _normalise_conditions(["translation"]) == ("raw_xf_translation",)


def test_module_version_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "convertMissalignment", "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "0.1.8" in completed.stdout


def test_historical_setup_help_accepts_translation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "convertMissalignment",
            "setup",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "translation" in completed.stdout
    assert "raw_xf_translation" in completed.stdout
