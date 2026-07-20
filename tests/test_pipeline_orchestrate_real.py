"""Gate 5: prepare_imod_to_warp.py EXECUTES the multiresolution + CTF + affine2d
preparation on a synthetic eTomo project with real IMOD (newstack, ctfphaseflip,
tilt). Skipped (not faked) without IMOD."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from imod_affine import write_xf

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
CTFPHASEFLIP = shutil.which("ctfphaseflip")
PREPARE = ROOT / "prepare_imod_to_warp.py"
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _project(tmp: Path, basename="64x_Vero_02", src=(1024, 768), ali_dims=None, n=7, voltage=300, extra_binning=8):
    """Synthetic eTomo project. ``ali_dims`` lets the aligned stack differ in size
    from the raw stack (exercises defect #10: separate raw/aligned grids)."""
    ali_dims = ali_dims or src
    data = tmp / "data"; data.mkdir()
    yy, xx = np.mgrid[0:src[1], 0:src[0]]
    base = (np.sin(xx / 40.0) * np.cos(yy / 50.0)).astype(np.float32)
    raw = np.stack([base * (1 + 0.01 * i) for i in range(n)]).astype(np.float32)
    ayy, axx = np.mgrid[0:ali_dims[1], 0:ali_dims[0]]
    abase = (np.sin(axx / 40.0) * np.cos(ayy / 50.0)).astype(np.float32)
    ali = np.stack([abase * (1 + 0.01 * i) for i in range(n)]).astype(np.float32)
    with mrcfile.new(data / f"{basename}_ali.mrc", overwrite=True) as h:
        h.set_data(ali); h.voxel_size = 1.36
    with mrcfile.new(data / f"{basename}.mrc", overwrite=True) as h:
        h.set_data(raw); h.voxel_size = 1.36  # synthetic raw
    ang = np.linspace(-60, 60, n)
    (data / f"{basename}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{basename}.xf", np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    (data / f"{basename}.defocus").write_text(
        "\n".join("%d %d %.2f %.2f %.1f" % (i + 1, i + 1, ang[i], ang[i], 5000.0) for i in range(n)) + "\n")
    (data / "ctfcorrection.com").write_text(
        "$ctfphaseflip -StandardInput\nInputStack ali.mrc\nOutputFileName ali_ctf.mrc\n"
        f"Voltage {voltage}\nSphericalAberration 2.7\nAmplitudeContrast 0.07\nDefocusTol 200\nInterpolationWidth 20\n")
    settings = tmp / "s.toml"
    settings.write_text(f'''
[project]
basename = "{basename}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/out"
[input]
aligned_stack = "{(data / f'{basename}_ali.mrc').as_posix()}"
raw_stack = "{(data / f'{basename}.mrc').as_posix()}"
final_xf_file = "{(data / f'{basename}.xf').as_posix()}"
final_tilt_file = "{(data / f'{basename}.tlt').as_posix()}"
[geometry]
raw_dimensions_xyz = [{src[0]}, {src[1]}, 1]
raw_pixel_size_A = 1.36
tilt_count = {n}
[conversion]
initial_conditions = ["ali_identity"]
[multiresolution]
enabled = true
extra_projection_binning = {extra_binning}
thickness_source_px = 256
[ctf]
mode = "working"
defocus_file = "{(data / f'{basename}.defocus').as_posix()}"
command_file = "{(data / 'ctfcorrection.com').as_posix()}"
[missalignment]
refinement_mode = "affine2d"
''')
    return data, settings, basename


def _run(args, extra_path=None):
    env = dict(ENV)
    if extra_path:
        env["PATH"] = extra_path + ":" + env["PATH"]
    return subprocess.run([sys.executable, str(PREPARE), *args], env=env, text=True, capture_output=True)


def _fake_sbatch(tmp: Path) -> str:
    """Create a real, executable fake ``sbatch`` so --submit is exercised end to end."""
    binp = tmp / "fakebin"; binp.mkdir()
    sb = binp / "sbatch"
    sb.write_text("#!/usr/bin/env bash\necho \"Submitted batch job 424242\"\n")
    sb.chmod(0o755)
    return str(binp)


@unittest.skipUnless(HAVE and NEWSTACK and CTFPHASEFLIP, "IMOD newstack/ctfphaseflip unavailable")
class OrchestrateRealTests(unittest.TestCase):
    def test_executes_bin8_working_ctf_affine2d(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp)
            before = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            cp = _run([str(settings), "--condition", "ali_identity",
                       "--ctf-mode", "working", "--refinement-mode", "affine2d", "--generate-slurm"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            ws = tmp / "out"
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            self.assertEqual(man["extra_binning"], 8)
            self.assertEqual(man["ctf_mode"], "working")
            self.assertEqual(man["refinement_mode"], "affine2d")
            # working aligned (real newstack) and working CTF (real ctfphaseflip) exist
            self.assertIn("working_aligned_uncorrected", man["stacks"])
            self.assertIn("working_aligned_ctf", man["stacks"])
            wa = Path(man["stacks"]["working_aligned_uncorrected"]["path"])
            wc = Path(man["stacks"]["working_aligned_ctf"]["path"])
            self.assertTrue(wa.is_file() and wc.is_file())
            with mrcfile.open(wa, permissive=True) as h:
                self.assertEqual((h.data.shape[2], h.data.shape[1]), (128, 96))  # 1024/8, 768/8
            # selected stack is the CTF one; missalignment apply_ctf is false
            self.assertEqual(man["stacks"]["working_selected"]["ctf_state"], "phase_flipped")
            self.assertEqual(man["missalignment"]["apply_ctf"], False)
            # REAL MissAlignment config (from 03_run_missalignment.config_text), not a placeholder
            res = ws / "missalignment" / "results" / "ali_identity" / "affine2d"
            cfg = (res / "config.yaml").read_text()
            self.assertTrue((res / "config.yaml").is_file())
            self.assertIn("general:", cfg)
            self.assertIn("training_directory:", cfg)
            self.assertIn("iteration_settings:", cfg)
            self.assertIn("alignment: [2, 2]", cfg)   # affine2d final movement-grid stage
            self.assertIn("apply_ctf: False", cfg)     # external IMOD CTF; never double-correct
            # run script invokes the REAL executable, not an echo
            run = (res / "run_missalignment.sh").read_text()
            self.assertIn("miss-alignment", run)
            self.assertIn("--config-file", run)
            self.assertNotIn("echo 'Run on Maxwell", run)
            # internal per-condition sbatch is no longer generated; Phase 2 owns Slurm.
            self.assertFalse((res / "missalign_affine2d.sbatch").exists())
            # manifest carries the real command + complete measured geometry
            self.assertIn("miss-alignment", man["missalignment"]["command"])
            self.assertIn("source_raw", man["geometry"]["measured"])
            self.assertIn("source_aligned", man["geometry"]["measured"])
            self.assertIn("G_a", man["geometry"]["maps"])
            # working_raw + working_xf actually executed
            self.assertIn("working_raw", man["steps_run"] + man["steps_skipped"])
            self.assertIn("working_xf", man["steps_run"] + man["steps_skipped"])
            self.assertIn("working_raw", man["stacks"])
            # SOURCE DATA NOT MODIFIED
            after = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            self.assertEqual(before, after, "source data was modified")

    def test_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _, settings, _ = _project(tmp)
            a = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "off"])
            self.assertEqual(a.returncode, 0, a.stderr[-800:])
            b = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "off", "--resume"])
            self.assertEqual(b.returncode, 0, b.stderr[-800:])
            self.assertIn("steps skipped", b.stdout)
            self.assertIn("working_aligned", b.stdout.split("steps skipped")[1])  # reused

    def test_extra_binning_1_no_reduction(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _, settings, _ = _project(tmp, extra_binning=1)
            cp = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "off"])
            self.assertEqual(cp.returncode, 0, cp.stderr[-800:])
            man = json.loads((tmp / "out" / "binning_ctf_manifest.json").read_text())
            self.assertEqual(man["extra_binning"], 1)
            self.assertEqual(man["stacks"]["working_aligned_uncorrected"]["binning_state"], "source")

    def test_separate_raw_and_aligned_dims(self):
        # defect #10: raw (1024x768) and aligned (960x720) differ; working_xf must
        # use SEPARATE in/out dims and the manifest must record both grids.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp, src=(1024, 768), ali_dims=(960, 720))
            cp = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "off"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            ws = tmp / "out"
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            self.assertEqual(man["source_dims_xy"], [1024, 768])
            self.assertEqual(man["source_aligned_dims_xy"], [960, 720])
            self.assertNotEqual(man["source_dims_xy"], man["source_aligned_dims_xy"])
            meas = man["geometry"]["measured"]
            self.assertEqual(meas["source_raw"]["shape_xy"], [1024, 768])
            self.assertEqual(meas["source_aligned"]["shape_xy"], [960, 720])
            # working_xf produced with the binned raw->aligned transform
            wxf = ws / "working_aligned" / f"{bn}_raw_to_aligned_bin8.xf"
            self.assertTrue(wxf.is_file())
            self.assertEqual(sum(1 for _ in wxf.read_text().splitlines()), 7)

    def test_ctf_params_come_from_ctfcorrection_com(self):
        # defect #9: the project ctfcorrection.com (Voltage 200) is authoritative.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp, voltage=200)
            cp = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "working"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            ws = tmp / "out"
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            self.assertEqual(man["ctf_params"]["voltage_kv"], 200)
            self.assertEqual(man["ctf_param_source"]["voltage_kv"], "ctfcorrection.com")

    def test_ctf_final_both_DEFERS_to_phase3(self):
        # §1.1 fix: ctf.mode final/both must NOT generate final CTF in Phase 1
        # (final CTF depends on the post-MissAlignment refined alignment). Phase 1
        # records the deferral; working CTF (for MissAlignment) is still produced.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp, src=(512, 512))
            before = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            cp = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "both"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            ws = tmp / "out"
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            # final CTF NOT produced in Phase 1; deferral recorded
            self.assertNotIn("final_aligned_ctf", man["stacks"])
            self.assertTrue(man["final_ctf_deferred_to_phase3"])
            self.assertNotIn("final_ctf", man["steps_run"])
            self.assertTrue(any("deferred to Phase 3" in w for w in man["warnings"]))
            # working CTF (for MissAlignment) IS still produced
            self.assertIn("working_aligned_ctf", man["stacks"])
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in data.iterdir()})

    def test_working_reconstruction_runs_tilt(self):
        # defect #3: --working-reconstruction executes a real tilt reconstruction.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp, src=(512, 512))
            cp = _run([str(settings), "--condition", "ali_identity",
                       "--ctf-mode", "off", "--working-reconstruction"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            ws = tmp / "out"
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            self.assertIn("working_reconstruction", man["steps_run"] + man["steps_skipped"])
            self.assertIn("working_reconstruction", man["stacks"])
            rec = Path(man["stacks"]["working_reconstruction"]["path"])
            self.assertTrue(rec.is_file())
            self.assertEqual(man["stacks"]["working_reconstruction"]["created_by_command"], "tilt -StandardInput")

    def test_submit_invokes_real_sbatch(self):
        # defect #8: --submit must actually call sbatch (here a real fake on PATH).
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp)
            extra = _fake_sbatch(tmp)
            cp = _run([str(settings), "--condition", "ali_identity",
                       "--ctf-mode", "off", "--refinement-mode", "standard",
                       "--generate-slurm", "--submit"], extra_path=extra)
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            self.assertIn("sbatch invoked", cp.stdout)
            ws = tmp / "out"
            res = ws / "missalignment" / "results" / "ali_identity" / "standard"
            sub = json.loads((res / "submission.json").read_text())
            self.assertEqual(sub["job_id"], "424242")
            man = json.loads((ws / "binning_ctf_manifest.json").read_text())
            self.assertTrue(man["missalignment"]["submission"]["submitted"])
            self.assertIn("submission", man["steps_run"])

    def test_submit_without_sbatch_warns_not_fakes(self):
        # On a non-SLURM host (no sbatch) --submit must warn, not silently "succeed".
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, settings, bn = _project(tmp)
            cp = _run([str(settings), "--condition", "ali_identity",
                       "--ctf-mode", "off", "--generate-slurm", "--submit"])
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            self.assertIn("sbatch", cp.stdout + cp.stderr)
            man = json.loads((tmp / "out"
                              / "binning_ctf_manifest.json").read_text())
            self.assertFalse(man["missalignment"]["submission"]["submitted"])

    def test_image_based_modes_real_config(self):
        # defect #12/#13: translation/rigid/similarity are accepted real modes with
        # the proper iteration_settings (rigid/similarity require the fork).
        expect = {"translation": "alignment: global",
                  "rigid": "alignment: rigid",
                  "similarity": "alignment: similarity"}
        for mode, needle in expect.items():
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                data, settings, bn = _project(tmp)
                cp = _run([str(settings), "--condition", "ali_identity",
                           "--ctf-mode", "off", "--refinement-mode", mode])
                self.assertEqual(cp.returncode, 0, f"{mode}: " + cp.stdout[-1200:] + cp.stderr[-1200:])
                res = (tmp / "out"
                       / "missalignment" / "results" / "ali_identity" / mode)
                cfg = (res / "config.yaml").read_text()
                self.assertIn(needle, cfg, f"{mode} config missing {needle}")
                self.assertIn("miss-alignment", (res / "run_missalignment.sh").read_text())

    def test_validate_only_and_rejects_raw_plus_working_ctf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _, settings, _ = _project(tmp)
            cv = _run([str(settings), "--condition", "ali_identity", "--ctf-mode", "working", "--validate-only"])
            self.assertEqual(cv.returncode, 0, cv.stderr[-500:])
            self.assertIn("configuration OK", cv.stdout)
            # raw condition + working CTF must fail clearly
            cr = _run([str(settings), "--condition", "raw_xf_affine_fixed", "--ctf-mode", "working"])
            self.assertNotEqual(cr.returncode, 0)
            self.assertIn("ali_identity", cr.stdout + cr.stderr)


if __name__ == "__main__":
    unittest.main()
