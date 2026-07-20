from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_settings(path: Path, project: Path) -> None:
    path.write_text(
        f'''[project]\nbasename = "TS1"\n\n[paths]\noutput_dir = "{project}"\n\n[conversion]\ninitial_conditions = ["raw_xf_affine_fixed"]\n\n[missalignment]\nrefinement_mode = "standard"\n\n[datasets]\nnative_id = "5.452Apx"\n'''
    )


def _make_dataset(project: Path, dataset_id: str, *, source_id: str | None = None,
                  accepted: bool = True, selected: bool = False) -> Path:
    dataset = project / "warp_data" / dataset_id
    root = dataset / ".warp_project"
    root.mkdir(parents=True, exist_ok=True)
    (root / "series.xml").write_text("<TiltSeries />\n")
    (root / "_converted.marker").write_text("ok\n")
    (root / "conversion_validation.json").write_text("{}\n")
    stack = root / "tiltstack" / "series" / "series.st"
    stack.parent.mkdir(parents=True)
    stack.write_bytes(b"stack")
    manifest = {
        "artifact_id": f"warp-{dataset_id}",
        "dataset_id": dataset_id,
        "pixel_size_A": float(dataset_id.removesuffix("Apx")),
        "status": "validated" if accepted else "complete",
        "preprocessing": (
            {"operation": "detector_resampling", "source_dataset_id": source_id}
            if source_id else None
        ),
        "source_artifact_id": f"warp-{source_id}" if source_id else None,
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest) + "\n")
    if accepted:
        acceptance = dataset / "reconstructions" / "acceptance.json"
        acceptance.parent.mkdir(parents=True)
        acceptance.write_text(json.dumps({"status": "accepted"}) + "\n")
    status_path = project / "project_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {
        "schema_version": 1,
        "layout_version": 8,
        "native_dataset_id": "5.452Apx",
        "datasets": {},
    }
    status["datasets"][dataset_id] = {
        "status": manifest["status"],
        "manifest": str(dataset / "manifest.json"),
    }
    if selected:
        status["selected_dataset_id"] = dataset_id
    status_path.write_text(json.dumps(status) + "\n")
    return dataset


def test_prepare_input_uses_directory_and_default_dataset(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    settings = project / "project_settings.toml"
    project.mkdir()
    _write_settings(settings, project)
    _make_dataset(project, "5.452Apx", accepted=True, selected=True)

    command = [
        sys.executable,
        str(repo / "prepare_missalignment_input.py"),
        "--directory",
        str(project),
    ]
    first = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "selected dataset: 5.452Apx" in first.stdout
    assert "input snapshots: created" in first.stdout

    for name in ("before", "smoke", "full"):
        snapshot = project / ".internal" / "workspaces" / "missalignment" / "5.452Apx" / name
        assert (snapshot / "series.xml").is_file()
        assert (snapshot / "tiltstack" / "series" / "series.st").exists()

    second = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert "input snapshots: reused" in second.stdout


def test_one_completed_preprocessed_dataset_becomes_default(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    _write_settings(project / "project_settings.toml", project)
    _make_dataset(project, "5.452Apx", accepted=True)
    _make_dataset(project, "16.356Apx", source_id="5.452Apx", accepted=True)
    # Simulate an early alpha project without selected_dataset_id. The single derived
    # dataset must still be chosen safely.
    status_path = project / "project_status.json"
    status = json.loads(status_path.read_text())
    status.pop("selected_dataset_id", None)
    status_path.write_text(json.dumps(status) + "\n")

    completed = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--directory", str(project)],
        cwd=repo, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "selected dataset: 16.356Apx" in completed.stdout


def test_dataset_path_and_list_are_accepted(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    _write_settings(project / "project_settings.toml", project)
    dataset = _make_dataset(project, "5.452Apx", accepted=True)

    listing = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--directory", str(project), "--list-datasets"],
        cwd=repo, text=True, capture_output=True,
    )
    assert listing.returncode == 0
    assert "Available datasets:" in listing.stdout
    assert "5.452Apx" in listing.stdout
    assert "imported" in listing.stdout

    selected = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--directory", str(project), "--dataset", str(dataset)],
        cwd=repo, text=True, capture_output=True,
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert "selected dataset: 5.452Apx" in selected.stdout


def _make_completed_reconstruction(project: Path, dataset_id: str) -> Path:
    attempt = (project / ".internal" / "attempts" / "reconstruction" / dataset_id /
               "warp_dataset" / "attempt_1")
    attempt.mkdir(parents=True, exist_ok=True)
    reconstruction = attempt / "reconstruction.mrc"
    reconstruction.write_bytes(b"mrc-volume")
    (attempt / "result_manifest.json").write_text(json.dumps({
        "status": "completed",
        "reconstruction": str(reconstruction),
    }) + "\n")
    (attempt.parent / "latest_success").symlink_to(attempt.name, target_is_directory=True)
    return reconstruction


def test_completed_reconstruction_is_technically_validated_automatically(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    _write_settings(project / "project_settings.toml", project)
    _make_dataset(project, "5.452Apx", accepted=False, selected=True)
    _make_completed_reconstruction(project, "5.452Apx")

    completed = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--directory", str(project)],
        cwd=repo, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "reconstruction validation: technical (automatic)" in completed.stdout
    validation_path = project / "warp_data" / "5.452Apx" / "reconstructions" / "acceptance.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "validated"
    assert validation["validation_level"] == "technical"
    assert validation["visual_inspection"] is False


def test_affine_without_reconstruction_prints_only_reconstruction_action(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    _write_settings(project / "project_settings.toml", project)
    _make_dataset(project, "5.452Apx", accepted=False, selected=True)

    completed = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--directory", str(project)],
        cwd=repo, text=True, capture_output=True,
    )
    assert completed.returncode == 2
    assert "no successful validated Warp reconstruction" in completed.stderr
    assert "reconstruct.sbatch" in completed.stderr
    assert "no separate acceptance command is required" in completed.stderr


def test_legacy_project_settings_alias_remains_accepted(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    settings = project / "project_settings.toml"
    _write_settings(settings, project)
    _make_dataset(project, "5.452Apx", accepted=True, selected=True)
    completed = subprocess.run(
        [sys.executable, str(repo / "prepare_missalignment_input.py"),
         "--project-settings", str(settings)],
        cwd=repo, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_generated_recovery_batch_is_cpu_only_and_uses_directory(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pipeline.jobs import generate_jobs
    from pipeline.runlayout import RunLayout

    layout = RunLayout.from_settings(
        out_dir=tmp_path / "project",
        basename="TS1",
        condition="raw_xf_affine_fixed",
        refinement_mode="standard",
        dataset_id="5.452Apx",
    ).create()
    settings = layout.run_dir / "project_settings.toml"
    settings.write_text("[project]\nbasename='TS1'\n")
    written = generate_jobs(
        layout,
        profile="maxwell",
        ma_command="miss-alignment --config-file full.yaml",
        smoke_command="miss-alignment --config-file smoke.yaml",
        run_script="run.sh",
        settings_path=str(settings),
        warp_staging_manifest="",
        reconstruction_config={},
    )
    batch = Path(written["missalignment/5.452Apx/prepare_input.sbatch"]).read_text()
    assert "prepare_missalignment_input.py" in batch
    assert "--directory" in batch
    assert "--project-settings" not in batch
    assert "#SBATCH --gres" not in batch
    assert "--require-cuda" not in batch
    assert "module load" not in batch
