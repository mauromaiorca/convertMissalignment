from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_help_does_not_expose_extra_binning() -> None:
    commands = (
        ("setup_missalign_project.py", "--help"),
        ("setup_warp_project.py", "--help"),
        ("prepare_imod_to_warp.py", "--help"),
        ("prepare_imod_to_warp.py", "prepare", "--help"),
    )
    for script, *args in commands:
        result = _run(script, *args)
        assert result.returncode == 0, result.stderr
        assert "--extra-binning" not in result.stdout


def test_public_clis_reject_extra_binning() -> None:
    commands = (
        (
            "setup_missalign_project.py",
            "--data-dir", ".",
            "--out-dir", "out",
            "--basename", "TS",
            "--extra-binning", "2",
        ),
        (
            "setup_warp_project.py",
            "--data-dir", ".",
            "--out-dir", "out",
            "--basename", "TS",
            "--extra-binning", "2",
        ),
        (
            "prepare_imod_to_warp.py",
            "settings.toml",
            "--extra-binning", "2",
        ),
        (
            "prepare_imod_to_warp.py",
            "prepare", "settings.toml",
            "--extra-binning", "2",
        ),
    )
    for script, *args in commands:
        result = _run(script, *args)
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr
        assert "--extra-binning" in result.stderr
