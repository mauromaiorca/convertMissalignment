from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.jobs import generate_jobs
from pipeline.runlayout import RunLayout, format_angpix


def _cluster():
    return SimpleNamespace(
        profile="maxwell", partition="vds", cpu_partition="cpu", constraint="V100",
        gres=None, memory=None, account=None, qos=None, nodelist=None, cpus=4,
        time="01:00:00", environment="", module_init_script=None, warp_module=None,
        imod_module=None, imod_bin_dir=None, omp_num_threads=None,
        cuda_visible_devices=None,
    )


def test_layout_is_project_centred_and_dataset_specific():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "lam8_ts_004"
        layout = RunLayout.from_settings(
            out_dir=root, basename="lam8_ts_004", condition="raw_xf_affine_fixed",
            refinement_mode="standard", dataset_id="5.45Apx",
        ).create()
        assert layout.run_dir == root
        assert layout.imported_imod_dir == root / "imported_data" / "imod"
        assert layout.warp_dataset_dir == root / "warp_data" / "5.45Apx"
        assert layout.training_dir.is_symlink()
        assert layout.manifest("source_inventory.json") == root / "provenance" / "source_inventory.json"
        assert layout.manifest("result_manifest.json") == root / "missalignment" / "runs" / "5.45Apx" / "result_manifest.json"
        assert layout.batch_path("warp_data", "reconstruct.sbatch") == root / "batches" / "warp_data" / "5.45Apx" / "reconstruct.sbatch"
        assert format_angpix(10.900000) == "10.9Apx"


def test_generated_batches_are_semantic_and_shell_valid():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        root.mkdir()
        settings = root / "project_settings.toml"
        settings.write_text("[project]\nbasename='x'\n")
        layout = RunLayout.from_settings(
            out_dir=root, basename="x", condition="raw_xf_affine_fixed",
            refinement_mode="standard", dataset_id="10.9Apx",
        ).create()
        written = generate_jobs(
            layout, profile="maxwell", ma_command="miss-alignment --config full.yaml",
            smoke_command="miss-alignment --config smoke.yaml", run_script="run.sh",
            settings_path=str(settings), cluster=_cluster(), include_import=False,
            preprocess_command="echo preprocess",
            reconstruction_config={"enabled": True, "warptools": {"enabled": True}},
        )
        expected = {
            "warp_data/10.9Apx/preprocess.sbatch",
            "warp_data/10.9Apx/reconstruct.sbatch",
            "missalignment/10.9Apx/prepare_input.sbatch",
            "missalignment/10.9Apx/run_smoke.sbatch",
            "missalignment/10.9Apx/run_full.sbatch",
            "export/10.9Apx/export_imod_and_reconstruct.sbatch",
        }
        assert expected <= set(written)
        assert not any("phase" in key.lower() for key in written if key.endswith(".sbatch"))
        for key, value in written.items():
            if not key.endswith(".sbatch"):
                continue
            completed = subprocess.run(["bash", "-n", value], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr


def test_acceptance_marks_dataset_validated():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        root.mkdir()
        settings = root / "project_settings.toml"
        settings.write_text('''[project]\nbasename="x"\nlayout_version=8\n[paths]\ndata_root="%s"\noutput_dir="%s"\n[geometry]\nraw_shape_xyz=[10,10,2]\nraw_pixel_size_A=2.0\naligned_shape_xyz=[10,10,2]\naligned_pixel_size_A=2.0\ntarget_volume_shape_xyz=[10,5,10]\ntarget_pixel_size_A=2.0\ntilt_axis_angle_deg=84.5\n[conversion]\ninitial_conditions=["raw_xf_affine_fixed"]\n[conversion.condition_modes]\nraw_xf_affine_fixed="quarter-turn-affine"\n[datasets]\nnative_id="2Apx"\nselected_id="2Apx"\n[missalignment]\nrefinement_mode="standard"\nresult_backend="warp_xml"\n[ctf]\nmode="off"\n[multiresolution]\nextra_projection_binning=1\n[cluster]\nprofile="maxwell"\n[reconstruction]\nenabled=true\n[provenance]\nresolved=true\n''' % (root, root))
        layout = RunLayout.from_settings(
            out_dir=root, basename="x", condition="raw_xf_affine_fixed",
            refinement_mode="standard", dataset_id="2Apx",
        ).create()
        layout.dataset_manifest.write_text(json.dumps({
            "dataset_id": "2Apx", "pixel_size_A": 2.0, "status": "complete"
        }))
        (root / "project_status.json").write_text(json.dumps({
            "datasets": {"2Apx": {"status": "complete"}}
        }))
        attempt = layout.attempts_dir / "reconstruction" / "2Apx" / "warp_dataset" / "attempt_1"
        attempt.mkdir(parents=True)
        rec = attempt / "rec.mrc"; rec.write_bytes(b"mrc")
        (attempt / "result_manifest.json").write_text(json.dumps({
            "status": "completed", "reconstruction": str(rec)
        }))
        latest = attempt.parent / "latest_success"
        latest.symlink_to(attempt.name, target_is_directory=True)
        script = ROOT / "scripts" / "pipeline" / "accept_pre_conversion.py"
        cp = subprocess.run([
            sys.executable, str(script), "--project-settings", str(settings),
            "--dataset", "2Apx"
        ], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stdout + cp.stderr
        assert json.loads(layout.dataset_manifest.read_text())["status"] == "validated"
        assert json.loads((root / "project_status.json").read_text())["datasets"]["2Apx"]["status"] == "validated"
