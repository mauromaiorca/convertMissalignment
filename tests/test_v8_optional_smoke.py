from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.jobs import generate_jobs
from pipeline.runlayout import RunLayout


def _cluster() -> SimpleNamespace:
    return SimpleNamespace(
        profile="maxwell", partition="vds", cpu_partition="cpu", constraint="V100",
        gres=None, memory=None, account=None, qos=None, nodelist=None, cpus=4,
        time="01:00:00", environment="", module_init_script=None, warp_module=None,
        imod_module=None, imod_bin_dir=None, omp_num_threads=None,
        cuda_visible_devices=None,
    )


def _generate(tmp_path: Path) -> tuple[RunLayout, dict[str, str]]:
    root = tmp_path / "project"
    root.mkdir()
    settings = root / "project_settings.toml"
    settings.write_text("[project]\nbasename='TS1'\n")
    layout = RunLayout.from_settings(
        out_dir=root,
        basename="TS1",
        condition="raw_xf_affine_fixed",
        refinement_mode="standard",
        dataset_id="5.452Apx",
    ).create()
    written = generate_jobs(
        layout,
        profile="maxwell",
        ma_command="miss-alignment --config-file full.yaml",
        smoke_command="miss-alignment --config-file smoke.yaml",
        run_script="run.sh",
        settings_path=str(settings),
        cluster=_cluster(),
        reconstruction_config={},
    )
    return layout, written


def test_full_batch_does_not_require_smoke_verdict(tmp_path: Path) -> None:
    _, written = _generate(tmp_path)
    body = Path(written["missalignment/5.452Apx/run_full.sbatch"]).read_text()
    assert "run run_smoke.sbatch first" not in body
    assert "ALLOW_WITHOUT_SMOKE" not in body
    assert "proceeding directly with the full run" in body
    assert "smoke testing is recommended but optional" in body
    assert "run prepare_missalignment_input.py first" in body


def test_result_helper_records_optional_smoke_state(tmp_path: Path) -> None:
    layout, written = _generate(tmp_path)
    helper = layout.helpers_dir / "record_missalignment_result.py"
    assert helper.is_file()

    full_dir = layout.full_warp_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    final_xml = full_dir / "series.xml"
    final_xml.write_text("<TiltSeries />\n")
    result_manifest = layout.manifest("result_manifest.json")
    result_manifest.parent.mkdir(parents=True, exist_ok=True)
    result_manifest.write_text(json.dumps({"training_directory": str(full_dir)}) + "\n")
    run_manifest = layout.manifest("missalignment_run_manifest.json")
    missing_smoke = layout.results_dir / "smoke_verdict.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--project-root", str(layout.run_dir),
            "--status", "completed",
            "--result-manifest", str(result_manifest),
            "--run-manifest", str(run_manifest),
            "--smoke-verdict", str(missing_smoke),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    record = json.loads(run_manifest.read_text())
    assert record["status"] == "completed"
    assert record["smoke_performed"] is False
    assert record["smoke_result"] == {}
    assert record["final_xml"] == str(final_xml)


def test_prepare_command_prints_smoke_and_full_choices(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = project / "project_settings.toml"
    settings.write_text(
        f'''[project]\nbasename = "TS1"\n\n[paths]\noutput_dir = "{project}"\n\n'''
        '''[conversion]\ninitial_conditions = ["raw_xf_translation"]\n\n'''
        '''[missalignment]\nrefinement_mode = "standard"\n\n'''
        '''[datasets]\nnative_id = "5.452Apx"\n'''
    )
    dataset = project / "warp_data" / "5.452Apx"
    warp_root = dataset / ".warp_project"
    warp_root.mkdir(parents=True)
    (warp_root / "series.xml").write_text("<TiltSeries />\n")
    (warp_root / "_converted.marker").write_text("ok\n")
    (warp_root / "conversion_validation.json").write_text("{}\n")
    stack = warp_root / "tiltstack" / "series" / "series.st"
    stack.parent.mkdir(parents=True)
    stack.write_bytes(b"stack")
    (dataset / "manifest.json").write_text(json.dumps({
        "dataset_id": "5.452Apx",
        "pixel_size_A": 5.452,
        "status": "complete",
    }) + "\n")
    (project / "project_status.json").write_text(json.dumps({
        "schema_version": 1,
        "layout_version": 8,
        "native_dataset_id": "5.452Apx",
        "selected_dataset_id": "5.452Apx",
        "datasets": {
            "5.452Apx": {
                "status": "complete",
                "manifest": str(dataset / "manifest.json"),
            }
        },
    }) + "\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "prepare_missalignment_input.py"),
            "--directory", str(project),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "recommended safety check (optional)" in completed.stdout
    assert "run_smoke.sbatch" in completed.stdout
    assert "direct full run" in completed.stdout
    assert "run_full.sbatch" in completed.stdout
    assert "not required by run_full.sbatch" in completed.stdout
