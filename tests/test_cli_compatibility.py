from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from convertMissalignment import __version__
from convertMissalignment.cli import (
    COMMANDS,
    export_guide,
    find_projects,
    find_reconstruct_batches,
    inventory,
    normalise_setup_arguments,
    reconstruct,
)
from setup_missalign_project import _normalise_conditions

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path, *datasets: str) -> Path:
    """A minimal project tree carrying only what 'reconstruct' inspects."""
    project = tmp_path / "project"
    (project).mkdir()
    (project / "project_settings.toml").write_text("[project]\nbasename = 'TS'\n")
    for dataset in datasets:
        batch_dir = project / "batches" / "warp_data" / dataset
        batch_dir.mkdir(parents=True)
        (batch_dir / "reconstruct.sbatch").write_text("#!/usr/bin/env bash\ntrue\n")
    return project


def test_reconstruct_finds_the_generated_batch(tmp_path: Path) -> None:
    project = _project(tmp_path, "1.363Apx")
    found = find_reconstruct_batches(project)
    assert [b.parent.name for b in found] == ["1.363Apx"]


def test_reconstruct_prints_the_submission_command(tmp_path: Path, capsys) -> None:
    project = _project(tmp_path, "1.363Apx")
    assert reconstruct([str(project), "--print"]) == 0
    out = capsys.readouterr().out
    assert "1.363Apx" in out
    assert "sbatch" in out


def test_reconstruct_rejects_a_non_project_directory(tmp_path: Path) -> None:
    assert reconstruct([str(tmp_path), "--print"]) == 2


def test_reconstruct_requires_dataset_when_ambiguous(tmp_path: Path) -> None:
    project = _project(tmp_path, "1.363Apx", "5.452Apx")
    assert reconstruct([str(project), "--print"]) == 2
    assert reconstruct([str(project), "--dataset", "5.452Apx", "--print"]) == 0


def test_distribution_source_version_is_0_1_15() -> None:
    assert __version__ == "0.1.15"
    assert (ROOT / "VERSION").read_text().strip() == "0.1.15"


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
    assert "0.1.15" in completed.stdout


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


def test_inventory_reports_done_and_missing_steps(tmp_path: Path, capsys) -> None:
    project = _project(tmp_path, "17.6Apx")
    done = project / "warp_data" / "17.6Apx" / "reconstructions" / "17.58Apx"
    done.mkdir(parents=True)
    (done / "TS_x.mrc").write_text("volume")
    assert inventory([str(project)]) == 0
    out = capsys.readouterr().out
    assert "17.6Apx" in out
    assert "TS_x.mrc" in out                     # produced artefact is located
    assert "run_full.sbatch" in out              # missing step names its command


def test_inventory_rejects_a_non_project_directory(tmp_path: Path) -> None:
    assert inventory([str(tmp_path)]) == 2


def test_prepare_input_is_no_longer_a_public_command() -> None:
    assert "prepare-input" not in COMMANDS
    assert "input" in COMMANDS


def test_find_projects_lists_projects_inside_a_directory(tmp_path: Path) -> None:
    for name in ("projA", "projB"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "project_settings.toml").write_text("[project]\n")
    assert [p.name for p in find_projects(tmp_path)] == ["projA", "projB"]
    assert find_projects(tmp_path / "projA") == [(tmp_path / "projA").resolve()]


def test_export_without_arguments_explains_what_to_run(tmp_path: Path, capsys) -> None:
    project = tmp_path / "proj"
    (project / "batches" / "export" / "5.4Apx").mkdir(parents=True)
    (project / "project_settings.toml").write_text("[project]\n")
    import os

    previous = os.getcwd()
    os.chdir(project)
    try:
        assert export_guide([]) == 0
    finally:
        os.chdir(previous)
    out = capsys.readouterr().out
    assert "export finalize" in out
    assert "export_imod_and_reconstruct.sbatch" in out


def test_inventory_lines_stay_narrow(tmp_path: Path, capsys) -> None:
    project = _project(tmp_path, "17.6Apx")
    volume = project / "warp_data" / "17.6Apx" / "reconstructions" / "17.58Apx"
    volume.mkdir(parents=True)
    (volume / "TS_a_very_long_series_name_raw_xf_translation_17.58Apx.mrc").write_text("v")
    assert inventory([str(project)]) == 0
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("Path ") or ".mrc" in line or ".png" in line:
            continue                            # full paths are deliberately on one line
        assert len(line) <= 80, line


def test_inventory_states_before_and_after_tomograms(tmp_path: Path, capsys) -> None:
    project = _project(tmp_path, "17.6Apx")
    rec = project / "warp_data" / "17.6Apx" / "reconstructions" / "17.59Apx"
    rec.mkdir(parents=True)
    (rec / "TS_x.mrc").write_text("v")
    (rec / "manifest.json").write_text('{"purpose": "geometry validation before MissAlignment"}')
    assert inventory([str(project)]) == 0
    out = capsys.readouterr().out
    assert "TOMOGRAMS" in out
    assert "already aligned" in out   # "before" is not "unaligned"
    assert "engine: geometry validation before MissAlignment" in out
    assert "NOT PRODUCED" in out                      # the after volume is absent
    assert "compare_reconstructions.sbatch" in out    # and how to obtain it
