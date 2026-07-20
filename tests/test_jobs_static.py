"""Focused static checks for generated Slurm jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import jobs as J
from pipeline.project_config import ClusterConfig
from pipeline.runlayout import RunLayout


def _layout(td):
    return RunLayout.from_settings(
        out_dir=Path(td), basename="64x_Vero_02",
        condition="raw_xf_affine_fixed", refinement_mode="standard",
    ).create()


def _gen(td, cluster=None):
    lay = _layout(td)
    written = J.generate_jobs(
        lay, profile="maxwell",
        ma_command=f"miss-alignment --config-file {lay.config_yaml}",
        smoke_command=f"miss-alignment --config-file {lay.run_dir}/missalignment/configs/config.smoke.yaml",
        run_script=f"{lay.run_dir}/missalignment/configs/run_missalignment.sh",
        settings_path="project_settings.toml",
        cluster=cluster or ClusterConfig(
            profile="maxwell", environment="/env", partition="vds", constraint="V100",
            cpu_partition="cssbcpu", module_init_script="/usr/share/Modules/init/bash",
            imod_module="imod/5.1.11",
            imod_bin_dir="/gpfs/cssb/software/rhel9/x86_64/imod/5.1.11/bin",
            warp_module="warp/2.0.39",
        ),
        warp_staging_manifest=str(lay.manifest("warp_staging_manifest.json")),
    )
    return lay, written


class JobStaticTests(unittest.TestCase):
    def test_conditional_gres_emission(self):
        with tempfile.TemporaryDirectory() as td:
            _, written = _gen(td, ClusterConfig(profile="maxwell", partition="vds", constraint="V100", gres=None))
            phase2 = Path(written["phase2.sbatch"]).read_text()
            self.assertIn("#SBATCH --partition=vds", phase2)
            self.assertIn("#SBATCH --constraint=V100", phase2)
            self.assertNotIn("#SBATCH --gres", phase2)

        with tempfile.TemporaryDirectory() as td:
            _, written = _gen(td, ClusterConfig(profile="maxwell", partition="vds", constraint="V100", gres="gpu:1"))
            phase2 = Path(written["phase2.sbatch"]).read_text()
            self.assertNotIn("#SBATCH --gres", phase2)

        with tempfile.TemporaryDirectory() as td:
            _, written = _gen(td, ClusterConfig(profile="other", partition="gpu", constraint=None, gres="gpu:1"))
            phase2 = Path(written["phase2.sbatch"]).read_text()
            self.assertIn("#SBATCH --gres=gpu:1", phase2)

        for empty in ("", None):
            with tempfile.TemporaryDirectory() as td:
                _, written = _gen(td, ClusterConfig(profile="maxwell", partition="vds", constraint="V100", gres=empty))
                self.assertNotIn("#SBATCH --gres", Path(written["phase2.sbatch"]).read_text())

    def test_generates_phase2_and_snapshot_reconstruction_slurm_files(self):
        with tempfile.TemporaryDirectory() as td:
            lay, written = _gen(td)
            self.assertIn("phase2.sbatch", written)
            self.assertIn("phase2a_convert_and_pre_reconstruct.sbatch", written)
            self.assertIn("phase3_pre_missalign_reconstruct.sbatch", written)
            self.assertIn("phase3_smoke_reconstruct.sbatch", written)
            self.assertIn("phase3_full_finalize_and_reconstruct.sbatch", written)
            self.assertIn("phase3_warptools_pre_vs_full_reconstruct.sbatch", written)
            self.assertNotIn("phase3_finalize.sbatch", written)
            self.assertNotIn("submit_phase2.sh", written)
            self.assertNotIn("_submit_helper.py", written)
            self.assertNotIn("00_cluster_preflight.sbatch", written)

            phase2 = Path(written["phase2.sbatch"]).read_text()
            self.assertLess(phase2.index("# ---- Phase 2 environment activation"),
                            phase2.index("DIAGNOSTIC PREAMBLE"))
            for text in (
                "set -Eeuo pipefail",
                'stage "environment/CUDA/MissAlignment preflight"',
                'stage "Warp conversion"',
                'stage "Warp-project validation"',
                'stage "MissAlignment smoke run"',
                'stage "smoke-verdict validation"',
                'stage "MissAlignment full run"',
                "run_warp_conversion.py",
                "_smoke_verdict.py",
                "run_missalignment.sh",
                "_phase2_complete.py",
                "pre_conversion/acceptance.json",
                "accept_pre_conversion.py",
            ):
                self.assertIn(text, phase2)
            self.assertIn(str(lay.training_dir), phase2)

            phase3 = Path(written["phase3_full_finalize_and_reconstruct.sbatch"]).read_text()
            self.assertNotIn("#SBATCH --partition=cpu", phase3)
            self.assertLess(phase3.index("# ---- Phase 3 IMOD environment activation"),
                            phase3.index("DIAGNOSTIC PREAMBLE"))
            self.assertIn("imod_reconstruction.py", phase3)
            self.assertIn("--project-settings project_settings.toml", phase3)
            self.assertIn("--snapshot full", phase3)
            self.assertIn("#SBATCH --partition=cssbcpu", phase3)
            self.assertIn("source /usr/share/Modules/init/bash", phase3)
            self.assertIn("module purge", phase3)
            self.assertIn("module load imod/5.1.11", phase3)
            self.assertIn("/gpfs/cssb/software/rhel9/x86_64/imod/5.1.11/bin", phase3)
            self.assertIn("command -v \"$program\"", phase3)
            self.assertNotIn("module load warp/2.0.39", phase3)
            self.assertIn("--expected-executor-sha", phase3)
            self.assertIn("--expected-settings-sha", phase3)

            pre_gate = Path(
                written["phase2a_convert_and_pre_reconstruct.sbatch"]
            ).read_text()
            self.assertIn("run_warp_conversion.py", pre_gate)
            self.assertIn("pre_conversion_reconstruction.py", pre_gate)
            self.assertNotIn('stage "MissAlignment smoke run"', pre_gate)
            self.assertIn("module load warp/2.0.39", pre_gate)
            cp = subprocess.run(
                ["bash", "-n", written["phase2a_convert_and_pre_reconstruct.sbatch"]],
                text=True,
                capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)

            warptools = Path(
                written["phase3_warptools_pre_vs_full_reconstruct.sbatch"]
            ).read_text()
            self.assertIn("#SBATCH --partition=vds", warptools)
            self.assertIn("#SBATCH --constraint=V100", warptools)
            self.assertIn("#SBATCH --mem=128G", warptools)
            self.assertIn("module load warp/2.0.39", warptools)
            self.assertIn("/env/bin/python", warptools)
            self.assertIn("warptools_reconstruction.py", warptools)
            self.assertIn("--expected-executor-sha", warptools)
            self.assertIn("--expected-settings-sha", warptools)
            self.assertIn('OUTPUT_ANGPIX="${OUTPUT_ANGPIX:-0.0}"', warptools)
            cp = subprocess.run(
                ["bash", "-n", written["phase3_warptools_pre_vs_full_reconstruct.sbatch"]],
                text=True,
                capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)

    def test_phase2_completion_helper_records_explicit_final_xml(self):
        with tempfile.TemporaryDirectory() as td:
            lay, written = _gen(td)
            final_xml = lay.run_dir / "missalignment" / "results" / "final.xml"
            final_xml.parent.mkdir(parents=True, exist_ok=True)
            final_xml.write_text("<xml/>\n")
            lay.manifest("result_manifest.json").write_text(json.dumps({
                "condition": lay.condition,
                "training_directory": str(lay.training_dir),
                "final_xml": None,
            }) + "\n")
            log = lay.run_dir / "logs" / "phase2" / "full.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"FINAL_XML={final_xml}\nFINAL_ITERATION=12\n")
            env = dict(os.environ, SLURM_JOB_ID="12345")
            cp = subprocess.run([
                sys.executable, written["_phase2_complete.py"],
                "--run-dir", str(lay.run_dir),
                "--status", "completed",
                "--command-log", str(log),
            ], text=True, capture_output=True, env=env)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            phase2 = json.loads(lay.manifest("phase2_manifest.json").read_text())
            result = json.loads(lay.manifest("result_manifest.json").read_text())
            self.assertEqual(phase2["status"], "completed")
            self.assertEqual(phase2["final_xml"], str(final_xml))
            self.assertEqual(phase2["completed_stages"],
                             ["preflight", "warp_conversion", "warp_validation",
                              "smoke", "smoke_verdict", "full_run"])
            self.assertEqual(result["final_xml"], str(final_xml))
            self.assertEqual(result["final_iteration"], 12)
            self.assertEqual(result["phase2_slurm_job_id"], "12345")

    def test_phase2_completion_helper_writes_failed_state(self):
        with tempfile.TemporaryDirectory() as td:
            lay, written = _gen(td)
            cp = subprocess.run([
                sys.executable, written["_phase2_complete.py"],
                "--run-dir", str(lay.run_dir),
                "--status", "failed",
                "--failed-line", "77",
                "--failed-command", "bad command",
            ], text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            phase2 = json.loads(lay.manifest("phase2_manifest.json").read_text())
            self.assertEqual(phase2["status"], "failed")
            self.assertEqual(phase2["failed_line"], "77")
            self.assertEqual(phase2["completed_stages"], [])


if __name__ == "__main__":
    unittest.main()
