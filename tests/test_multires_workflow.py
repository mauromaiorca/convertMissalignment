"""Phase 4/7/10/20: workflow orchestration, rejection gate, .xf conversion,
command generation, Z sampling, manifest, and a real-tilt IMAGEBINNED dims check."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from imod_affine import read_xf, write_xf, xf_to_homogeneous
from multiresolution import Grid2D, MultiresError, build_plan, validate_request
from multiresolution import transfer as T
from multiresolution import workflow as W

try:
    import mrcfile
    HAVE_MRC = True
except Exception:
    HAVE_MRC = False
TILT = shutil.which("tilt")
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}


class RejectionGateTests(unittest.TestCase):
    def test_rejects_unsupported(self):
        with self.assertRaises(MultiresError):
            validate_request(3, (256, 192))            # not in {2,4,8}
        with self.assertRaises(MultiresError):
            validate_request(2.5, (256, 192))          # non-integer
        with self.assertRaises(MultiresError):
            validate_request(4, (250, 192))            # 250 not divisible by 4
        with self.assertRaises(MultiresError):
            validate_request(4, (256, 192), anisotropic=True)
        with self.assertRaises(MultiresError):
            validate_request(4, (256, 192), axis_permutation=True)
        with self.assertRaises(MultiresError):
            validate_request(4, (256, 192), reflect=True)

    def test_accepts_supported(self):
        for B in (2, 4, 8):
            self.assertEqual(validate_request(B, (256, 192)), B)


class PlanAndConversionTests(unittest.TestCase):
    def test_plan_grids_and_manifest(self):
        sr = Grid2D.axis_aligned("source_raw", (256, 192), 1.36)
        plan = build_plan(4, sr)
        self.assertEqual(plan.working_raw.shape_xy, (64, 48))
        self.assertAlmostEqual(plan.working_raw.pixel_size_xy_A[0], 1.36 * 4, places=6)
        self.assertAlmostEqual(plan.G_r[0, 2], 1.5, places=9)  # (B-1)/2
        man = plan.manifest()
        for key in ("source_raw", "source_aligned", "working_raw", "working_aligned"):
            self.assertIn("Q", man["grids"][key])
        self.assertIn("G_r_working_raw_to_source_raw", man["maps"])

    def test_xf_conversion_matches_transfer_formula(self):
        sr = Grid2D.axis_aligned("source_raw", (256, 192), 1.0)
        plan = build_plan(4, sr)
        A0 = np.stack([np.array([[np.cos(0.05), -np.sin(0.05)], [np.sin(0.05), np.cos(0.05)]]),
                       np.eye(2)])
        d0 = np.stack([np.array([3.0, -2.0]), np.array([1.0, 1.0])])
        with tempfile.TemporaryDirectory() as td:
            sxf = Path(td) / "s.xf"; wxf = Path(td) / "w.xf"
            write_xf(sxf, A0, d0)
            W.convert_source_xf_to_working(sxf, plan, wxf)
            Aw, dw = read_xf(wxf)
            for i in range(2):
                H0 = xf_to_homogeneous(A0[i], d0[i], (256, 192), (256, 192))
                expect = T.h0_working(H0, plan.G_r, plan.G_a)
                got = xf_to_homogeneous(Aw[i], dw[i], plan.working_raw.shape_xy, plan.working_aligned.shape_xy)
                self.assertTrue(np.allclose(got, expect, atol=1e-9))

    def test_working_z_sampling_is_thickness_based(self):
        # nz from physical thickness / working voxel, NOT from the binning factor
        nz, pz, ext = W.working_z_sampling(physical_thickness_A=1088.0, working_pixel_A=5.44)
        self.assertEqual(nz, 200)
        self.assertAlmostEqual(ext, 1088.0, places=3)

    def test_command_generation_and_script_syntax(self):
        sr = Grid2D.axis_aligned("source_raw", (256, 192), 1.36)
        plan = build_plan(4, sr)
        raw_cmd = W.newstack_working_raw_cmd(Path("s.st"), Path("r.st"), 4)
        self.assertIn("-shrink", raw_cmd)
        ali_cmd = W.newstack_working_aligned_onepass_cmd(Path("s.st"), Path("s.xf"), Path("a.st"), 4)
        self.assertIn("-xform", ali_cmd); self.assertIn("-shrink", ali_cmd)
        com = W.tilt_working_com(in_stack="a.st", out_rec="w.rec", tilt_file="w.tlt",
                                 working=plan.working_aligned, nz=200)
        self.assertIn("IMAGEBINNED 1", com)        # explicit working geometry -> no double scaling
        self.assertIn("FULLIMAGE 64 48", com)
        self.assertIn("THICKNESS 200", com)
        with tempfile.TemporaryDirectory() as td:
            script = W.reconstruction_run_script("w.com", "w.rec")
            p = Path(td) / "run.sh"; p.write_text(script)
            cp = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertIn("refusing to overwrite", script)


@unittest.skipUnless(TILT and HAVE_MRC, "real tilt / mrcfile unavailable")
class RealTiltImagebinnedTests(unittest.TestCase):
    def test_explicit_working_geometry_reconstruction_dims(self):
        # Minimal synthetic working aligned stack; reconstruct with explicit
        # working geometry + IMAGEBINNED 1; confirm output volume X/Z geometry.
        nx, ny, n = 64, 48, 7
        rng = np.random.default_rng(0)
        stack = rng.random((n, ny, nx)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ali = tdp / "ali.st"
            with mrcfile.new(ali, overwrite=True) as h:
                h.set_data(stack); h.voxel_size = 5.44
            tlt = tdp / "w.tlt"
            tlt.write_text("\n".join(f"{v:.2f}" for v in np.linspace(-60, 60, n)) + "\n")
            nz = 24
            rec = tdp / "w.rec"
            cp = subprocess.run(
                ["tilt", "-input", str(ali), "-output", str(rec), "-TILTFILE", str(tlt),
                 "-FULLIMAGE", f"{nx} {ny}", "-THICKNESS", str(nz), "-IMAGEBINNED", "1",
                 "-RADIAL", "0.35 0.05"],  # multi-value option = single token
                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr[-500:])
            with mrcfile.open(rec, permissive=True) as h:
                shp = h.data.shape  # (z_slices_in_Y, thickness, X) for tilt output
        # tilt writes the reconstruction as (Y, Z, X): X==nx, Z==thickness
        self.assertEqual(shp[2], nx, f"reconstruction X dim should be working nx; got {shp}")
        self.assertEqual(shp[1], nz, f"reconstruction Z (thickness) should be {nz}; got {shp}")


if __name__ == "__main__":
    unittest.main()
