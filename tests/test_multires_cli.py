"""Phase 7/21: multiresolution CLI integration (export --restore-to source,
rejection gate, validate --level multiresolution, config examples)."""
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
from imod_affine import read_xf, write_xf

try:
    import torch
    import alignment_models as am
    from alignment_models.serialization import write_params_json
    HAVE = True
except Exception:
    HAVE = False


def _settings(tmp, B, src_dims, xf_path):
    return f'''
[project]
basename = "mr"
[paths]
output_dir = "{tmp.as_posix()}/out"
[input]
raw_stack = "{tmp.as_posix()}/raw.st"
final_xf_file = "{xf_path.as_posix()}"
[geometry]
raw_dimensions_xyz = [{src_dims[0]}, {src_dims[1]}, 1]
raw_pixel_size_A = 1.0
aligned_pixel_size_A = 1.0
target_pixel_size_A = 1.0
tilt_count = 3
[conversion]
initial_conditions = ["ali_identity"]
[refinement]
model = "rigid"
[multiresolution]
enabled = true
extra_projection_binning = {B}
restore_target = "source"
'''


@unittest.skipUnless(HAVE, "torch unavailable")
class MultiresCliTests(unittest.TestCase):
    def _make(self, tmp, B=4, src_dims=(256, 192)):
        n = 3
        A0 = np.stack([np.eye(2) for _ in range(n)])
        d0 = np.stack([np.array([2.0 + i, -1.0 - i]) for i in range(n)])
        xf = tmp / "src.xf"; write_xf(xf, A0, d0)
        (tmp / "raw.st").write_text("x")
        # working-grid residual params (rigid)
        m = am.get_model("rigid")
        params = m.identity_params(n) + torch.tensor([[1.0, -0.5, 0.02]] * n, dtype=torch.float64)
        resid = tmp / "resid.json"; write_params_json(resid, m, params)
        settings = tmp / "s.toml"; settings.write_text(_settings(tmp, B, src_dims, xf))
        return settings, resid

    def _run(self, args):
        return subprocess.run([sys.executable, str(ROOT / "export_warp_to_imod.py"), *args],
                              text=True, capture_output=True)

    def test_restore_to_source_writes_source_xf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid = self._make(tmp, B=4, src_dims=(256, 192))
            out = tmp / "exp"
            cp = self._run([str(settings), "--residual-params", str(resid), "--condition", "ali_identity",
                            "--restore-to", "source", "--out-dir", str(out)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            xfs = list(out.glob("*final_source_raw_to_aligned.xf"))
            self.assertEqual(len(xfs), 1, "source .xf must be written")
            A, d = read_xf(xfs[0])
            self.assertEqual(len(A), 3)
            report = json.loads((out / "multiresolution_restore_report.json").read_text())
            self.assertEqual(report["factor"], 4)
            self.assertAlmostEqual(report["G_a"][0][2], 1.5, places=6)

    def test_dry_run_and_validate_only(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid = self._make(tmp)
            out = tmp / "exp"
            cpd = self._run([str(settings), "--residual-params", str(resid), "--condition", "ali_identity",
                             "--restore-to", "source", "--dry-run", "--out-dir", str(out)])
            self.assertEqual(cpd.returncode, 0, cpd.stderr)
            self.assertFalse(out.exists(), "dry-run must not write")
            cpv = self._run([str(settings), "--residual-params", str(resid), "--condition", "ali_identity",
                             "--restore-to", "source", "--validate-only", "--out-dir", str(out)])
            self.assertEqual(cpv.returncode, 0, cpv.stderr)
            self.assertIn("configuration OK", cpv.stdout)

    def test_non_divisible_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings, resid = self._make(tmp, B=4, src_dims=(250, 192))  # 250 % 4 != 0
            cp = self._run([str(settings), "--residual-params", str(resid), "--condition", "ali_identity",
                            "--restore-to", "source", "--out-dir", str(tmp / "exp")])
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("divisible", (cp.stdout + cp.stderr).lower())

    def test_validate_level_multiresolution(self):
        cp = subprocess.run([sys.executable, str(ROOT / "validate_interoperability.py"),
                             "--level", "multiresolution"], text=True, capture_output=True)
        self.assertEqual(cp.returncode, 0, cp.stdout[-2000:] + cp.stderr[-1500:])
        self.assertIn("PASS", cp.stdout)

    def test_config_examples_validate(self):
        from multiresolution import validate_request
        import tomllib
        for name in ("multiresolution_bin2", "multiresolution_bin4", "multiresolution_bin8",
                     "multiresolution_bin4_preview_xy2"):
            with (ROOT / "config" / "examples" / f"{name}.toml").open("rb") as fh:
                cfg = tomllib.load(fh)
            B = cfg["multiresolution"]["extra_projection_binning"]
            self.assertEqual(validate_request(B, (4096, 4096)), B)


if __name__ == "__main__":
    unittest.main()
