"""Reconstruction library (§16): explicit inputs, source geometry, half-sets,
command-file generation, prerequisite validation. Local tilt run gated on IMOD."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reconstruction as REC
from reconstruction.model import ReconstructionError, ReconstructionRequest

NEWSTACK = shutil.which("newstack")


class ReconstructionLibTests(unittest.TestCase):
    def _files(self, td, n=5):
        d = Path(td)
        tlt = d / "s.tlt"; tlt.write_text("\n".join(f"{a:.2f}" for a in range(-40, 41, 20)[:n]) + "\n")
        xf = d / "s.xf"
        xf.write_text("\n".join("   1.0000000   0.0000000   0.0000000   1.0000000      0.000      0.000"
                                for _ in range(n)) + "\n")
        ali = d / "s_ali.mrc"; ali.write_text("fake")
        return d, tlt, xf, ali

    def test_tilt_com_uses_source_geometry(self):
        com = REC.build_tilt_com(in_stack="ali.mrc", out_rec="rec.mrc", tilt_file="s.tlt",
                                 fullimage_xy=(2048, 2048), thickness=1500)
        self.assertIn("FULLIMAGE 2048 2048", com)
        self.assertIn("THICKNESS 1500", com)
        self.assertIn("IMAGEBINNED 1", com)             # never IMAGEBINNED B
        self.assertNotIn("IMAGEBINNED 8", com)

    def test_validate_request_rejects_bad_inputs(self):
        with self.assertRaises(ReconstructionError):
            REC.validate_request(ReconstructionRequest(output_dir="/tmp/x", input_mode="aligned_stack",
                                                       tilt_file="/tmp/none.tlt"))
        with self.assertRaises(ReconstructionError):
            REC.validate_request(ReconstructionRequest(output_dir="/tmp/x", input_mode="bogus"))

    def test_prerequisite_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            d, tlt, xf, ali = self._files(td)
            req = ReconstructionRequest(output_dir=str(d / "out"), input_mode="raw_plus_xf",
                                        raw_stack=str(ali), xf_file=str(xf), tilt_file=str(tlt))
            rep = REC.validate_prerequisites(req)
            self.assertEqual(rep["xf_rows"], rep["tilt_rows"])
            # mismatch -> error
            (d / "bad.xf").write_text("1 0 0 1 0 0\n")
            req2 = ReconstructionRequest(output_dir=str(d / "o2"), input_mode="raw_plus_xf",
                                         raw_stack=str(ali), xf_file=str(d / "bad.xf"), tilt_file=str(tlt))
            with self.assertRaises(ReconstructionError):
                REC.validate_prerequisites(req2)

    def test_halfset_split_angle_and_index(self):
        angles = [-40, -20, 0, 20, 40]
        ha = REC.split_halfsets(angles, mode="angle")
        self.assertEqual(sorted(ha["even"] + ha["odd"]), [0, 1, 2, 3, 4])
        self.assertEqual(len(set(ha["even"]) & set(ha["odd"])), 0)
        hi = REC.split_halfsets(angles, mode="index")
        self.assertEqual(hi["even"], [0, 2, 4])
        self.assertEqual(hi["odd"], [1, 3])

    def test_prepare_skip_generates_files_with_halfmaps(self):
        with tempfile.TemporaryDirectory() as td:
            d, tlt, xf, ali = self._files(td)
            req = ReconstructionRequest(
                output_dir=str(d / "out"), input_mode="aligned_stack", aligned_stack=str(ali),
                tilt_file=str(tlt), execution="skip", halfmaps=True, half_split_mode="angle",
                fullimage_xy=(2048, 2048), thickness=1500, basename="s")
            res = REC.prepare_imod_reconstruction(req)
            self.assertTrue(Path(res.tilt_com).is_file())
            self.assertFalse(res.executed)
            for half in ("even", "odd"):
                self.assertTrue(Path(res.half_files[half]).is_file())
                self.assertTrue((Path(res.output_dir) / f"tilt_final_{half}.com").is_file())

    def test_prepare_slurm_generates_sbatch(self):
        with tempfile.TemporaryDirectory() as td:
            d, tlt, xf, ali = self._files(td)
            req = ReconstructionRequest(
                output_dir=str(d / "out"), input_mode="aligned_stack", aligned_stack=str(ali),
                tilt_file=str(tlt), execution="slurm", fullimage_xy=(2048, 2048),
                thickness=1500, basename="s")
            res = REC.prepare_imod_reconstruction(req)
            self.assertTrue(Path(res.sbatch).is_file())
            txt = Path(res.sbatch).read_text()
            self.assertIn("set -Eeuo pipefail", txt)
            self.assertIn("submfg", txt)

    @unittest.skipUnless(NEWSTACK, "IMOD unavailable")
    def test_local_execution_real_tilt(self):
        import numpy as np
        try:
            import mrcfile
        except Exception:
            self.skipTest("mrcfile unavailable")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); n = 5
            ali = d / "s_ali.mrc"
            with mrcfile.new(ali, overwrite=True) as h:
                h.set_data(np.random.rand(n, 128, 128).astype(np.float32)); h.voxel_size = 1.0
            tlt = d / "s.tlt"; tlt.write_text("\n".join(f"{a:.2f}" for a in np.linspace(-40, 40, n)) + "\n")
            req = ReconstructionRequest(
                output_dir=str(d / "out"), input_mode="aligned_stack", aligned_stack=str(ali),
                tilt_file=str(tlt), execution="local", fullimage_xy=(128, 128), thickness=40, basename="s")
            res = REC.prepare_imod_reconstruction(req)
            self.assertTrue(res.executed)
            self.assertTrue(Path(res.output_rec).is_file())


if __name__ == "__main__":
    unittest.main()
