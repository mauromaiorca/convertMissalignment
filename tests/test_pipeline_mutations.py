"""Mutation/negative-control campaign for the binning + CTF + affine2d pipeline.
Proves the code/suite detects each classic error. (Composition-order, wrong-grid,
G_r-vs-G_a, and translation-only mutations are covered in test_multires_*; this
file covers the CTF/orchestration-specific ones.)"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from imod_affine import xf_to_homogeneous
from multiresolution import Grid2D, build_plan
from multiresolution import transfer as T
from pipeline import ctf as C
from pipeline.datastate import Stack, final_reconstruction_input, select_for_missalignment


class CtfMutationTests(unittest.TestCase):
    def test_double_ctf_detected(self):
        C.assert_uncorrected_input("uncorrected")  # ok
        with self.assertRaises(C.CtfError):
            C.assert_uncorrected_input("phase_flipped")  # double CTF

    def test_apply_ctf_must_be_false_for_external(self):
        # Selecting an externally CTF-corrected stack -> MissAlignment apply_ctf must be false.
        # (The orchestrator hard-codes apply_ctf=False; here we assert the data-state carries
        #  the external phase_flipped state so a true apply_ctf would be a double correction.)
        ctf = Stack(role="working_aligned_ctf", path="c.mrc", alignment_state="working_aligned",
                    ctf_state="phase_flipped", binning_state="working", intended_use="missalignment_input")
        sel = select_for_missalignment(
            Stack(role="working_aligned_uncorrected", path="u.mrc", alignment_state="working_aligned",
                  ctf_state="uncorrected", binning_state="working", intended_use="working_qc"),
            ctf, "working")
        self.assertEqual(sel.ctf_state, "phase_flipped")  # external CTF -> apply_ctf:false required

    def test_wrong_ctf_input_stack_rejected_via_mode(self):
        with self.assertRaises(C.CtfError):
            C.validate_ctf_mode("working", "raw_xf_affine_fixed", True)  # raw stack, working CTF

    def test_preview_cannot_feed_missalignment_or_final(self):
        with self.assertRaises(ValueError):
            Stack(role="preview", path="p.mrc", alignment_state="working_aligned", ctf_state="uncorrected",
                  binning_state="preview", intended_use="visualization_only", allowed_for_missalignment=True)

    def test_final_uses_source_not_working(self):
        # final_reconstruction_input never returns a working-binned stack
        final_unc = Stack(role="final_source_aligned_uncorrected", path="f.mrc", alignment_state="final_aligned",
                          ctf_state="uncorrected", binning_state="source", intended_use="final_reconstruction",
                          allowed_for_final_reconstruction=True)
        final_ctf = Stack(role="final_source_aligned_ctf", path="fc.mrc", alignment_state="final_aligned",
                          ctf_state="phase_flipped", binning_state="source", intended_use="final_reconstruction",
                          allowed_for_final_reconstruction=True)
        self.assertEqual(final_reconstruction_input(final_unc, final_ctf, "final").ctf_state, "phase_flipped")
        self.assertEqual(final_reconstruction_input(final_unc, None, "off").ctf_state, "uncorrected")


class RestoreMutationTests(unittest.TestCase):
    def setUp(self):
        # Supported scope: divisible dims -> G_r == G_a == universal bin-B map.
        sr = Grid2D.axis_aligned("sr", (256, 192), 1.0)
        sa = Grid2D.axis_aligned("sa", (256, 192), 1.0)
        self.plan = build_plan(4, sr, sa)
        a = np.deg2rad(3.0)
        A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        self.Hfw = xf_to_homogeneous(A, np.array([1.2, -0.8]), self.plan.working_raw.shape_xy,
                                     self.plan.working_aligned.shape_xy)

    def test_correct_vs_mutations_supported_scope(self):
        correct = T.restore_hfinal_working_to_source(self.Hfw, self.plan.G_a, self.plan.G_r)
        Gr, Ga = self.plan.G_r, self.plan.G_a
        muts = {
            "no_inverse": Ga @ self.Hfw @ Gr,             # wrong: forgot inverse
            "reversed": np.linalg.inv(Gr) @ self.Hfw @ Ga,  # wrong sides
            "identity_only": self.Hfw,                    # forgot the grid transfer
        }
        for name, m in muts.items():
            self.assertGreater(np.max(np.abs(correct - m)), 1e-6, f"mutation {name} not detected")

    def test_Gr_Ga_distinction_when_grids_differ(self):
        # In the supported scope G_r == G_a; the formula's use of inv(G_r) (not
        # inv(G_a)) only matters when the grids genuinely differ. Construct two
        # different bin maps manually to show the restore IS sensitive to it.
        Ga = np.array([[4.0, 0, 1.5], [0, 4.0, 1.5], [0, 0, 1]])
        Gr = np.array([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1]])  # different factor -> Gr != Ga
        correct = T.restore_hfinal_working_to_source(self.Hfw, Ga, Gr)
        swapped = Ga @ self.Hfw @ np.linalg.inv(Ga)  # wrong: inv(G_a) instead of inv(G_r)
        self.assertGreater(np.max(np.abs(correct - swapped)), 1e-6)


if __name__ == "__main__":
    unittest.main()
