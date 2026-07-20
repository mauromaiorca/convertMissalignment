"""CTF orchestration: data-state model, mode rules, command-file patching, and
REAL ctfphaseflip geometric-invariant validation. Skipped (not faked) without
ctfphaseflip/mrcfile."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pipeline import ctf as C
from pipeline.datastate import Stack, select_for_missalignment

try:
    import mrcfile
    HAVE_MRC = True
except Exception:
    HAVE_MRC = False
CTFPHASEFLIP = shutil.which("ctfphaseflip")


class DataStateTests(unittest.TestCase):
    def test_invariants(self):
        # preview/visualization cannot be allowed for missalignment or final
        with self.assertRaises(ValueError):
            Stack(role="preview", path="p.mrc", alignment_state="working_aligned",
                  ctf_state="uncorrected", binning_state="preview", intended_use="visualization_only",
                  allowed_for_missalignment=True)
        # working-binned cannot feed the final reconstruction
        with self.assertRaises(ValueError):
            Stack(role="working_aligned_ctf", path="w.mrc", alignment_state="working_aligned",
                  ctf_state="phase_flipped", binning_state="working", intended_use="missalignment_input",
                  allowed_for_final_reconstruction=True)
        # invalid literal
        with self.assertRaises(ValueError):
            Stack(role="bogus", path="x", alignment_state="raw", ctf_state="uncorrected",
                  binning_state="source", intended_use="input")

    def test_selection_is_manifest_driven(self):
        unc = Stack(role="working_aligned_uncorrected", path="u.mrc", alignment_state="working_aligned",
                    ctf_state="uncorrected", binning_state="working", intended_use="working_qc")
        ctf_stack = Stack(role="working_aligned_ctf", path="c.mrc", alignment_state="working_aligned",
                    ctf_state="phase_flipped", binning_state="working", intended_use="missalignment_input")
        self.assertEqual(select_for_missalignment(unc, ctf_stack, "working").ctf_state, "phase_flipped")
        self.assertEqual(select_for_missalignment(unc, ctf_stack, "off").ctf_state, "uncorrected")
        self.assertEqual(select_for_missalignment(unc, None, "off").path, "u.mrc")
        with self.assertRaises(ValueError):
            select_for_missalignment(unc, None, "working")  # ctf requested, none generated


class CtfModeTests(unittest.TestCase):
    def test_mode_rules(self):
        for m in ("off", "working", "final", "both"):
            C.validate_ctf_mode(m, "ali_identity", True)  # ok
        with self.assertRaises(C.CtfError):
            C.validate_ctf_mode("working", "raw_xf_affine_fixed", True)  # raw + working CTF
        with self.assertRaises(C.CtfError):
            C.validate_ctf_mode("both", "ali_identity", False)  # no aligned stack
        with self.assertRaises(C.CtfError):
            C.validate_ctf_mode("bogus", "ali_identity", True)


class CtfComPatchTests(unittest.TestCase):
    def test_copy_patch_source_unmodified(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ctfcorrection.com"
            src.write_text("$ctfphaseflip -StandardInput\nInputStack orig_ali.mrc\n"
                           "OutputFileName orig_ctf.mrc\nVoltage 300\nSphericalAberration 2.7\n"
                           "AmplitudeContrast 0.07\nDefocusTol 200\nInterpolationWidth 20\nPixelSize 0.136\n")
            before = src.read_text()
            local = Path(td) / "working_imod" / "ctf" / "ctfcorrection_working.com"
            rep = C.patch_ctf_com(src, local, {
                "InputStack": "working_ali.mrc", "OutputFileName": "working_ctf.mrc", "PixelSize": "1.088"})
            self.assertEqual(src.read_text(), before, "source must be unmodified")
            self.assertTrue(rep["source_unmodified"])
            self.assertIn("working_ali.mrc", local.read_text())
            self.assertIn("Voltage 300", local.read_text())  # preserved

    def test_rejects_transformfile_and_stripwidth(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ctfcorrection.com"
            src.write_text("$ctfphaseflip -StandardInput\nInputStack raw.mrc\nTransformFile a.xf\n")
            with self.assertRaises(C.CtfError):
                C.patch_ctf_com(src, Path(td) / "l.com", {"InputStack": "w.mrc"})  # raw-stack CTF rejected
            src2 = Path(td) / "c2.com"; src2.write_text("$ctfphaseflip\nInputStack a.mrc\n")
            with self.assertRaises(C.CtfError):
                C.patch_ctf_com(src2, Path(td) / "l2.com", {"MaximumStripWidth": "0"})


@unittest.skipUnless(CTFPHASEFLIP and HAVE_MRC, "ctfphaseflip/mrcfile unavailable")
class CtfRealImodTests(unittest.TestCase):
    def _make_aligned(self, td, n=5, nx=96, ny=96, pix_A=10.88):
        yy, xx = np.mgrid[0:ny, 0:nx]
        base = (np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype(np.float32)
        stack = np.stack([base * (1 + 0.01 * i) for i in range(n)]).astype(np.float32)
        ali = Path(td) / "ali.mrc"
        with mrcfile.new(ali, overwrite=True) as h:
            h.set_data(stack); h.voxel_size = pix_A
        ang = np.linspace(-60, 60, n)
        tlt = Path(td) / "t.tlt"; tlt.write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
        defo = Path(td) / "d.defocus"
        defo.write_text("\n".join("%d %d %.2f %.2f %.1f" % (i + 1, i + 1, ang[i], ang[i], 5000.0)
                                  for i in range(n)) + "\n")
        return ali, tlt, defo

    def test_real_ctf_geometric_invariants(self):
        with tempfile.TemporaryDirectory() as td:
            ali, tlt, defo = self._make_aligned(td)
            out = Path(td) / "ali_ctf.mrc"
            cmd = C.build_ctfphaseflip_cmd(
                input_stack=ali, output_stack=out, angle_file=tlt, defocus_file=defo,
                pixel_size_A=10.88, unbinned_pixel_A=1.36)  # working bin8 of 1.36 A
            cp = C.run_ctfphaseflip(cmd)
            self.assertEqual(cp.returncode, 0, cp.stdout[-400:] + cp.stderr[-400:])
            self.assertTrue(out.is_file())
            rep = C.validate_ctf_output(ali, out)
            self.assertTrue(rep["ok"], rep)
            self.assertTrue(rep["dims_preserved"] and rep["pixel_preserved"]
                            and rep["tilt_count_preserved"] and rep["all_finite"])
            # uncorrected and CTF stacks coexist; input not overwritten
            self.assertTrue(ali.is_file() and out.is_file())
            with mrcfile.open(ali, permissive=True) as h:
                self.assertEqual(h.data.shape[0], 5)  # input intact


if __name__ == "__main__":
    unittest.main()
