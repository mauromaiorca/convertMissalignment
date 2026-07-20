"""End-to-end test of export_warp_to_imod.py (exact constrained-model export)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from imod_affine import forward_points_pixels, write_xf

if HAVE_TORCH:
    import alignment_models as am
    from alignment_models.serialization import write_params_json

EXPORT = ROOT / "export_warp_to_imod.py"


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class InteropExportTests(unittest.TestCase):
    def _project(self, tmp: Path, model_name="affine"):
        n = 4
        A0 = np.stack([_rot(np.deg2rad(2.0 + i)) @ np.array([[1.02, 0.02], [-0.01, 0.99]]) for i in range(n)])
        d0 = np.stack([np.array([3.0 + i, -2.0 - i]) for i in range(n)])
        xf = tmp / "series.xf"
        write_xf(xf, A0, d0)
        tlt = tmp / "series.tlt"
        tlt.write_text("\n".join(str(v) for v in np.linspace(-30, 30, n)) + "\n")
        raw = tmp / "series.st"
        raw.write_text("stack\n")  # placeholder; export does not read pixels

        model = am.get_model(model_name)
        params = model.identity_params(n) + 0.03 * torch.randn((n, model.n_params), dtype=torch.float64)
        resid = tmp / "residual.params.json"
        write_params_json(resid, model, params, tilt_angles=list(np.linspace(-30, 30, n)))

        settings = tmp / "settings.toml"
        settings.write_text(f'''
[project]
basename = "series"
[paths]
output_dir = "{tmp.as_posix()}/out"
[input]
raw_stack = "{raw.as_posix()}"
final_xf_file = "{xf.as_posix()}"
final_tilt_file = "{tlt.as_posix()}"
[geometry]
raw_dimensions_xyz = [256, 192, 1]
aligned_dimensions_xyz = [256, 192, 1]
target_volume_xyz = [256, 192, 1]
raw_pixel_size_A = 10.0
aligned_pixel_size_A = 10.0
target_pixel_size_A = 10.0
tilt_count = {n}
[conversion]
initial_conditions = ["raw_xf_affine_fixed", "ali_identity"]
[refinement]
model = "{model_name}"
[validation]
coordinate_max_tolerance_px = 0.01
''')
        return settings, resid, A0, d0, n

    def _run(self, args):
        cp = subprocess.run([sys.executable, str(EXPORT), *args], text=True, capture_output=True)
        return cp

    def test_ali_identity_export(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid, A0, d0, n = self._project(tmp)
            out = tmp / "exp"
            cp = self._run([str(settings), "--residual-params", str(resid),
                            "--condition", "ali_identity", "--out-dir", str(out)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            res_xf = out / "series_ali_identity_affine_ali_residual.xf"
            comp_xf = out / "series_ali_identity_affine_raw_to_final.xf"
            self.assertTrue(res_xf.is_file())
            self.assertTrue(comp_xf.is_file())
            report = json.loads((out / "export_report.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertLess(report["raw_ali_equivalence_max_px"], 0.01)

    def test_raw_condition_export(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid, A0, d0, n = self._project(tmp)
            out = tmp / "exp"
            cp = self._run([str(settings), "--residual-params", str(resid),
                            "--condition", "raw_xf_affine_fixed", "--out-dir", str(out)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            self.assertTrue((out / "series_raw_xf_affine_fixed_affine_raw_to_final.xf").is_file())
            self.assertFalse((out / "series_raw_xf_affine_fixed_affine_ali_residual.xf").is_file())

    def test_validate_only_and_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid, A0, d0, n = self._project(tmp)
            cpv = self._run([str(settings), "--residual-params", str(resid),
                             "--condition", "ali_identity", "--validate-only"])
            self.assertEqual(cpv.returncode, 0, cpv.stderr)
            self.assertIn("status=PASS", cpv.stdout)
            cpd = self._run([str(settings), "--residual-params", str(resid),
                             "--condition", "ali_identity", "--dry-run", "--out-dir", str(tmp / "z")])
            self.assertEqual(cpd.returncode, 0, cpd.stderr)
            self.assertFalse((tmp / "z").exists(), "dry-run must not write outputs")

    def test_identity_residual_reproduces_initial_alignment(self):
        # With no residual file, the composed raw->final must equal the original raw->ali xf.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid, A0, d0, n = self._project(tmp)
            out = tmp / "exp"
            cp = self._run([str(settings), "--condition", "ali_identity", "--out-dir", str(out)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            from imod_affine import read_xf
            cA, cd = read_xf(out / "series_ali_identity_affine_raw_to_final.xf")
            # composed (raw->final) with identity residual == original raw->ali.
            # Tolerance is the IMOD .xf text-format precision (matrix 7 decimals,
            # shift 3 decimals -- the standard newstack format); the in-memory
            # homogeneous equivalence is verified to 1e-6 in export_report.json.
            pts = np.array([[10, 10], [240, 30], [120, 170]], float)
            for i in range(n):
                a = forward_points_pixels(pts, cA[i], cd[i], (256, 192), (256, 192))
                b = forward_points_pixels(pts, A0[i], d0[i], (256, 192), (256, 192))
                self.assertLess(np.max(np.abs(a - b)), 1e-3)

    def test_export_unequal_raw_ali_dims(self):
        # Guards against raw/ali shape confusion in the export: with raw != ali
        # dims, using the wrong shape for the composed raw->final xf makes the
        # built-in raw/ali self-check diverge -> status FAIL.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            n = 4
            A0 = np.stack([_rot(np.deg2rad(2.0 + i)) @ np.array([[1.02, 0.02], [-0.01, 0.99]]) for i in range(n)])
            d0 = np.stack([np.array([3.0 + i, -2.0 - i]) for i in range(n)])
            xf = tmp / "series.xf"; write_xf(xf, A0, d0)
            (tmp / "series.tlt").write_text("\n".join(str(v) for v in np.linspace(-30, 30, n)) + "\n")
            (tmp / "series.st").write_text("stack\n")
            model = am.get_model("affine")
            params = model.identity_params(n) + 0.02 * torch.randn((n, 6), dtype=torch.float64)
            resid = tmp / "r.json"
            from alignment_models.serialization import write_params_json
            write_params_json(resid, model, params)
            settings = tmp / "s.toml"
            settings.write_text(f'''
[project]
basename = "series"
[paths]
output_dir = "{tmp.as_posix()}/out"
[input]
raw_stack = "{(tmp / 'series.st').as_posix()}"
final_xf_file = "{xf.as_posix()}"
final_tilt_file = "{(tmp / 'series.tlt').as_posix()}"
[geometry]
raw_dimensions_xyz = [256, 192, 1]
aligned_dimensions_xyz = [300, 240, 1]
target_volume_xyz = [280, 210, 1]
raw_pixel_size_A = 10.0
aligned_pixel_size_A = 10.0
target_pixel_size_A = 10.0
tilt_count = {n}
[conversion]
initial_conditions = ["ali_identity"]
[refinement]
model = "affine"
[validation]
coordinate_max_tolerance_px = 0.01
''')
            cp = self._run([str(settings), "--residual-params", str(resid),
                            "--condition", "ali_identity", "--out-dir", str(tmp / "exp")])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            report = json.loads((tmp / "exp" / "export_report.json").read_text())
            # correct code keeps the raw/ali self-check at machine precision even with unequal dims
            self.assertLess(report["raw_ali_equivalence_max_px"], 1e-6)
            self.assertEqual(report["status"], "PASS")

    def test_reconstruction_files(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid, A0, d0, n = self._project(tmp)
            out = tmp / "exp"
            cp = self._run([str(settings), "--residual-params", str(resid),
                            "--condition", "ali_identity", "--out-dir", str(out),
                            "--write-reconstruction-files"])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            script = out / "reconstruction_inputs" / "run_imod_reconstruction.sh"
            self.assertTrue(script.is_file())
            text = script.read_text()
            self.assertIn("newstack", text)
            self.assertIn("refusing to overwrite", text)  # source-safety guard


if __name__ == "__main__":
    unittest.main()
