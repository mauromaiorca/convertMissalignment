"""§9: a Warp-conversion step POPULATES the training dir before MissAlignment. Without it
the smoke/full run against an empty dir. We verify: (a) prepare emits one phase2.sbatch
that runs conversion before smoke/full; (b) the staging manifest is surfaced at the
canonical layout path; (c) run_warp_conversion.py consumes the manifest,
stages the TS dir (symlinked stack, §10) with the REAL .xf (§7), and fails CLEANLY at the
warpylib boundary (cluster-only) rather than crashing or silently producing nothing."""
from __future__ import annotations

import json
import os
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
import shutil
NEWSTACK = shutil.which("newstack")
PREPARE = ROOT / "prepare_imod_to_warp.py"
CONV = ROOT / "scripts" / "run_warp_conversion.py"
PY = sys.executable
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _project(tmp: Path, bn="64x_Vero_02", n=5):
    data = tmp / "data"; data.mkdir()
    with mrcfile.new(data / f"{bn}.mrc", overwrite=True) as h:
        h.set_data(np.random.rand(n, 64, 48).astype(np.float32)); h.voxel_size = 1.363
    with mrcfile.new(data / f"{bn}_ali.mrc", overwrite=True) as h:
        h.set_data(np.random.rand(n, 32, 24).astype(np.float32)); h.voxel_size = 2.726
    ang = np.linspace(-40, 40, n)
    (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    (data / f"{bn}.rawtlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{bn}.xf", np.stack([np.array([[1.0, 0.02], [-0.02, 1.0]])] * n),
             np.full((n, 2), 1.5))
    (data / "align.com").write_text("$tiltalign\nRotationAngle\t84.0\n")
    (data / "tilt.com").write_text("$tilt\nTHICKNESS 40\nIMAGEBINNED 1\n")
    s = tmp / "s.toml"
    s.write_text(f'''
[project]
basename = "{bn}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/out"
[conversion]
initial_conditions = ["raw_xf_affine_fixed"]
[ctf]
mode = "off"
[multiresolution]
extra_projection_binning = 1
[missalignment]
refinement_mode = "standard"
''')
    return data, s, bn


@unittest.skipUnless(HAVE and NEWSTACK, "mrcfile/newstack needed")
class JobWiringTests(unittest.TestCase):
    def test_prepare_emits_phase2_with_conversion_before_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _project(tmp)
            init = subprocess.run([PY, str(PREPARE), "init", str(s)], env=ENV, text=True, capture_output=True)
            self.assertEqual(init.returncode, 0, init.stdout + init.stderr)
            resolved = tmp / "out" / "project_settings.toml"
            prep = subprocess.run([PY, str(PREPARE), "prepare", str(resolved)],
                                  env=ENV, text=True, capture_output=True)
            self.assertEqual(prep.returncode, 0, prep.stdout[-2000:] + prep.stderr[-2000:])
            run_dirs = list((tmp / "out").glob(f"{bn}_raw_xf_affine_fixed_*"))
            self.assertTrue(run_dirs, "no run dir")
            rd = run_dirs[0]
            phase2 = rd / "jobs" / "phase2.sbatch"
            self.assertTrue(phase2.is_file(), "phase2.sbatch not generated")
            body = phase2.read_text()
            self.assertIn("run_warp_conversion.py", body)
            self.assertIn("--staging-manifest", body)
            self.assertLess(body.index("START Warp conversion"),
                            body.index("START MissAlignment smoke run"))
            self.assertLess(body.index("START smoke-verdict validation"),
                            body.index("START MissAlignment full run"))
            # staging manifest surfaced at the canonical layout path
            staging = rd / "manifests" / "warp_staging_manifest.json"
            self.assertTrue(staging.is_file(), "staging manifest not surfaced")
            man = json.loads(staging.read_text())
            self.assertFalse(man["is_identity"])                     # raw_xf_affine_fixed
            self.assertTrue(str(man["staged_xf"]).endswith(".xf"))


@unittest.skipUnless(HAVE, "mrcfile needed")
class ConversionRunnerTests(unittest.TestCase):
    def _manifest(self, tmp: Path, *, is_identity, n=5):
        bn = "s"
        stack = tmp / ("ali.mrc" if is_identity else "raw.mrc")
        with mrcfile.new(stack, overwrite=True) as h:
            h.set_data(np.random.rand(n, 32, 24).astype(np.float32)); h.voxel_size = 2.726
        tlt = tmp / "s.tlt"; tlt.write_text("\n".join("0.0" for _ in range(n)) + "\n")
        xf = tmp / "s.xf"
        write_xf(xf, np.stack([np.eye(2)] * n), np.zeros((n, 2)))
        training = tmp / "warp" / "warp_cond"
        man = {"series_name": bn, "condition": "raw_xf_affine_fixed" if not is_identity else "ali_identity",
               "warp_alignment_mode": "full-affine" if not is_identity else "identity",
               "axis_frame": "aligned", "tilt_axis_angle_deg": 84.0,
               "target_volume_shape_xyz": [24, 16, 32], "target_pixel_size_A": 2.726,
               "input_stack": str(stack), "tilt_file": str(tlt),
               "staged_xf": None if is_identity else str(xf), "is_identity": is_identity,
               "training_dir": str(training)}
        mp = tmp / "warp_staging_manifest.json"; mp.write_text(json.dumps(man))
        return mp, training, stack

    def test_stages_then_fails_cleanly_at_warpylib(self):
        # warpylib is absent locally: the runner must stage everything, then exit with the
        # cluster-only code 3 and a clear message — never a traceback, never a silent pass.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mp, training, stack = self._manifest(tmp, is_identity=False)
            cp = subprocess.run([PY, str(CONV), "--staging-manifest", str(mp)],
                                text=True, capture_output=True)
            self.assertEqual(cp.returncode, 3, cp.stdout + cp.stderr)
            self.assertIn("warpylib", cp.stdout + cp.stderr)
            # staged the TS dir with a SYMLINKED stack (§10) and the REAL .xf (§7)
            ts = training.parent / "staging" / "TS_s_raw_xf_affine_fixed"
            self.assertTrue((ts / "TS_s_raw_xf_affine_fixed.st").is_symlink())
            self.assertTrue((ts / "TS_s_raw_xf_affine_fixed.xf").is_file())
            self.assertTrue((ts / "TS_s_raw_xf_affine_fixed.rawtlt").is_file())
            # NOT converted (no warpylib) -> no marker (downstream smoke job will refuse)
            self.assertFalse((training / "_converted.marker").is_file())

    def test_missing_manifest_is_clean_error(self):
        cp = subprocess.run([PY, str(CONV), "--staging-manifest", "/no/such/manifest.json"],
                            text=True, capture_output=True)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("not found", (cp.stdout + cp.stderr).lower())


if __name__ == "__main__":
    unittest.main()
