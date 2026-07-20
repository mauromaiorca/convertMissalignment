"""Phase-3 finalize consumes the CANONICAL result contract (not latest-mtime XML),
exports constrained .xf DIRECTLY from parameters, and verify-final validates.
Requires IMOD (prepare builds the grids) + torch; skipped otherwise."""
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
    import torch  # noqa: F401
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
PREPARE = ROOT / "prepare_imod_to_warp.py"
EXPORT = ROOT / "export_warp_to_imod.py"
PY = sys.executable
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _project(tmp: Path, bn="64x_Vero_02"):
    data = tmp / "data"; data.mkdir(); n = 5
    ali = np.random.rand(n, 256, 256).astype(np.float32)
    for nm in (f"{bn}_ali.mrc", f"{bn}.mrc"):
        with mrcfile.new(data / nm, overwrite=True) as h:
            h.set_data(ali); h.voxel_size = 1.36
    ang = np.linspace(-40, 40, n)
    (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{bn}.xf", np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    s = tmp / "s.toml"
    s.write_text(f'''
[project]
basename = "{bn}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/runs"
[input]
aligned_stack = "{(data / f'{bn}_ali.mrc').as_posix()}"
raw_stack = "{(data / f'{bn}.mrc').as_posix()}"
final_xf_file = "{(data / f'{bn}.xf').as_posix()}"
final_tilt_file = "{(data / f'{bn}.tlt').as_posix()}"
[geometry]
raw_dimensions_xyz = [256, 256, 1]
raw_pixel_size_A = 1.36
[conversion]
initial_conditions = ["ali_identity"]
[multiresolution]
enabled = true
extra_projection_binning = 8
[ctf]
mode = "off"
[missalignment]
refinement_mode = "rigid"
result_backend = "constrained_json"
''')
    return data, s, bn, n


@unittest.skipUnless(HAVE and NEWSTACK, "IMOD/torch/mrcfile unavailable")
class FinalizePhase3Tests(unittest.TestCase):
    def _prepare(self, s):
        # rigid here exercises the constrained finalize math, not the 2.15 gate, so
        # explicitly allow the unavailable fork for the prepare step.
        cp = subprocess.run([PY, str(PREPARE), "prepare", str(s), "--allow-unavailable-mode", "--allow-unresolved-legacy"],
                            env=ENV, text=True, capture_output=True)
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def _write_canonical_result(self, run_dir, mode, n):
        # simulate the cluster MissAlignment producing the canonical contract
        sys.path.insert(0, str(ROOT / "scripts"))
        from alignment_models import result_contract as RC
        from alignment_models.registry import get_model
        rdir = run_dir / "missalignment" / "results" / mode
        rdir.mkdir(parents=True, exist_ok=True)
        model = get_model(mode)
        # a small per-tilt residual (tx,ty,phi)
        params = [[1.5, -1.0, 0.03 + 0.005 * i] for i in range(n)]
        RC.write_constrained_result(
            rdir, model=mode, params=params, tilt_angles=list(np.linspace(-40, 40, n)),
            param_names=tuple(model.param_names), scopes={}, gauge={}, regularization={},
            working_raw_grid=None, working_aligned_grid=None, input_hashes={}, warp_project_hash="abc",
            loss_history=[1.0, 0.01], gradient_summary={}, stage_history=[], software_versions={},
            cuda_info=None, seed=1, start_time="t0", end_time="t1", completion_status="completed")
        return rdir, params

    def test_finalize_consumes_canonical_and_exports_direct(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn, n = _project(tmp)
            self._prepare(s)
            run_dir = tmp / "runs" / f"{bn}_ali_identity_rigid"
            self._write_canonical_result(run_dir, "rigid", n)
            cp = subprocess.run([PY, str(EXPORT), "finalize", str(s), "--result", "auto"],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            ft = run_dir / "final" / "transforms"
            for name in ("working_residual.xf", "source_residual.xf",
                         "final_source_raw_to_aligned.xf", "working_raw_to_final.xf"):
                p = ft / name
                self.assertTrue(p.is_file(), name)
                self.assertEqual(len(p.read_text().splitlines()), n, name)
            fin = json.loads((run_dir / "manifests" / "finalize_manifest.json").read_text())
            self.assertEqual(fin["model"], "rigid")
            self.assertEqual(fin["n_tilts"], n)

    def test_finalize_refuses_incomplete_result(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn, n = _project(tmp)
            self._prepare(s)
            run_dir = tmp / "runs" / f"{bn}_ali_identity_rigid"
            from alignment_models import result_contract as RC
            from alignment_models.registry import get_model
            rdir = run_dir / "missalignment" / "results" / "rigid"; rdir.mkdir(parents=True, exist_ok=True)
            RC.write_constrained_result(
                rdir, model="rigid", params=[[0, 0, 0]] * n, tilt_angles=list(range(n)),
                param_names=("tx", "ty", "phi"), scopes={}, gauge={}, regularization={},
                working_raw_grid=None, working_aligned_grid=None, input_hashes={}, warp_project_hash=None,
                loss_history=[], gradient_summary={}, stage_history=[], software_versions={},
                cuda_info=None, seed=1, start_time="t0", end_time="t1", completion_status="crashed")
            cp = subprocess.run([PY, str(EXPORT), "finalize", str(s), "--result", "auto"],
                                env=ENV, text=True, capture_output=True)
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("not completed", (cp.stdout + cp.stderr))

    def test_verify_final(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn, n = _project(tmp)
            self._prepare(s)
            run_dir = tmp / "runs" / f"{bn}_ali_identity_rigid"
            self._write_canonical_result(run_dir, "rigid", n)
            subprocess.run([PY, str(EXPORT), "finalize", str(s), "--result", "auto"],
                           env=ENV, text=True, capture_output=True)
            cp = subprocess.run([PY, str(EXPORT), "verify-final", str(s)],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            val = json.loads((run_dir / "manifests" / "final_validation.json").read_text())
            self.assertTrue(val["ok"])


    def test_warp_xml_backend_refuses_mtime(self):
        # 2.16: result_backend=warp_xml must require an EXPLICIT XML, never mtime.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn, n = _project(tmp)
            self._prepare(s)
            cp = subprocess.run([PY, str(EXPORT), "finalize", str(s),
                                 "--result-backend", "warp_xml"], env=ENV, text=True, capture_output=True)
            self.assertNotEqual(cp.returncode, 0)
            out = cp.stdout + cp.stderr
            self.assertIn("explicit", out.lower())
            self.assertIn("mtime", out.lower())


class FinalizeHelpTests(unittest.TestCase):
    def test_subcommands_in_help(self):
        cp = subprocess.run([PY, str(EXPORT), "--help"], text=True, capture_output=True)
        for verb in ("finalize", "verify-final", "collect-debug"):
            self.assertIn(verb, cp.stdout, verb)


if __name__ == "__main__":
    unittest.main()
