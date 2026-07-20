"""Phase 1 grid-algebra invariants: transfer, restore (all 4 models), negative
controls. Pure numpy (no torch/IMOD)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from imod_affine import xf_to_homogeneous
from multiresolution import Grid2D, Grid3D, integer_binned_grid, preview_grid_from
from multiresolution import transfer as T


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _model_residual(kind, shape):
    """A working-aligned pixel-homogeneous residual representative of each model."""
    if kind == "translation":
        A, t = np.eye(2), np.array([1.5, -0.8])
    elif kind == "rigid":
        A, t = _rot(2.0), np.array([0.7, -0.4])
    elif kind == "similarity":
        A, t = np.exp(0.03) * _rot(1.5), np.array([0.5, 0.3])
    elif kind == "affine":
        A = _rot(1.0) @ np.array([[np.exp(0.02), 0.04], [0.0, np.exp(-0.01)]])
        t = np.array([0.6, -0.5])
    else:
        raise ValueError(kind)
    return xf_to_homogeneous(A, t, shape, shape)


class GridBasicsTests(unittest.TestCase):
    def test_pixel_physical_roundtrip(self):
        for shape, pix in [((256, 192), 1.36), ((257, 193), 2.0)]:
            g = Grid2D.axis_aligned("g", shape, pix)
            pts = np.array([[0, 0], [shape[0] - 1, shape[1] - 1], [10.5, 20.25]])
            back = g.physical_to_pixel(g.pixel_to_physical(pts))
            self.assertTrue(np.allclose(back, pts, atol=1e-9))

    def test_binning_offset_is_B_minus_1_over_2(self):
        src = Grid2D.axis_aligned("src", (256, 192), 1.36)
        for B in (2, 4, 8):
            G_r = integer_binned_grid(src, B).mapping_to(src)
            self.assertAlmostEqual(G_r[0, 0], B, places=9)
            self.assertAlmostEqual(G_r[1, 1], B, places=9)
            self.assertAlmostEqual(G_r[0, 2], (B - 1) / 2, places=9)
            self.assertAlmostEqual(G_r[1, 2], (B - 1) / 2, places=9)

    def test_offset_is_not_naive_translation_scaling(self):
        # The transferred translation is NOT just B*translation; the (B-1)/2 centre
        # offset must be present (proves we do not "scale translation by B").
        src = Grid2D.axis_aligned("src", (256, 192), 5.0)
        wk = integer_binned_grid(src, 4)
        G_r = wk.mapping_to(src)
        self.assertNotAlmostEqual(G_r[0, 2], 0.0, places=6)


class TransferInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.src = Grid2D.axis_aligned("src", (256, 192), 1.36)
        self.A0 = _rot(3.0) @ np.array([[1.02, 0.03], [-0.01, 0.98]])
        self.d0 = np.array([4.0, -3.0])
        self.H0_src = xf_to_homogeneous(self.A0, self.d0, self.src.shape_xy, self.src.shape_xy)

    def _grids(self, B, raw_eq_ali=True):
        wk = integer_binned_grid(self.src, B)
        G_r = wk.mapping_to(self.src)
        # aligned grid: same binning; allow a different aligned source shape
        ali_src = self.src if raw_eq_ali else Grid2D.axis_aligned("ali_src", (240, 200), 1.36)
        wk_a = integer_binned_grid(ali_src, B, name=f"wk_a{B}")
        G_a = wk_a.mapping_to(ali_src)
        # rebuild H0_src raw(src)->aligned(ali_src)
        H0 = xf_to_homogeneous(self.A0, self.d0, self.src.shape_xy, ali_src.shape_xy)
        return wk, wk_a, G_r, G_a, H0

    def test_transform_chain_invariance_all_factors(self):
        pts = np.array([[5, 7, 1], [60, 40, 1], [30, 45, 1], [0, 0, 1]], float).T
        for B in (2, 4, 8):
            for raw_eq_ali in (True, False):
                wk, wk_a, G_r, G_a, H0 = self._grids(B, raw_eq_ali)
                H0w = T.h0_working(H0, G_r, G_a)
                viaw = H0w @ pts
                vias = np.linalg.inv(G_a) @ H0 @ G_r @ pts
                self.assertTrue(np.allclose(viaw, vias, atol=1e-9))
                # inverse transfer round-trips
                H0_back = T.h0_source_from_working(H0w, G_r, G_a)
                self.assertTrue(np.allclose(H0_back, H0, atol=1e-9))

    def test_restore_invariance_all_models(self):
        src_pts = np.array([[3, 4, 1], [200, 30, 1], [120, 170, 1], [255, 191, 1]], float).T
        for B in (2, 4, 8):
            wk, wk_a, G_r, G_a, H0 = self._grids(B)
            H0w = T.h0_working(H0, G_r, G_a)
            for kind in ("translation", "rigid", "similarity", "affine"):
                dHw = _model_residual(kind, wk_a.shape_xy)
                Hf_src = T.hfinal_source(dHw, H0, G_a)
                Hf_via = T.hfinal_source_via_working(H0w, dHw, G_a, G_r)
                self.assertTrue(np.allclose(Hf_src, Hf_via, atol=1e-9), f"{kind} B={B} two-formula")
                # restore invariance: source route == working route applied to source raw points
                direct = Hf_src @ src_pts
                working_route = (G_a @ (dHw @ H0w) @ np.linalg.inv(G_r)) @ src_pts
                self.assertTrue(np.allclose(direct, working_route, atol=1e-9), f"{kind} B={B} restore")

    def test_physical_frame_residual_roundtrip(self):
        for B in (2, 4, 8):
            wk, wk_a, G_r, G_a, H0 = self._grids(B)
            dHw = _model_residual("affine", wk_a.shape_xy)
            # source aligned grid Q and working aligned grid Q
            Q_wa = wk_a.Q
            Q_sa = Grid2D.axis_aligned("ali_src", self.src.shape_xy, 1.36).Q
            dH_phys = T.deltaH_physical(dHw, Q_wa)
            dH_src_via_phys = T.deltaH_export(dH_phys, Q_sa)
            dH_src_direct = T.deltaH_source(dHw, G_a)
            self.assertTrue(np.allclose(dH_src_via_phys, dH_src_direct, atol=1e-9))


class NegativeControlTests(unittest.TestCase):
    """In-test negative controls (the full mutation campaign is separate)."""

    def setUp(self):
        self.src = Grid2D.axis_aligned("src", (256, 192), 1.36)
        self.wk = integer_binned_grid(self.src, 4)
        self.G = self.wk.mapping_to(self.src)
        self.H0 = xf_to_homogeneous(_rot(4.0), np.array([5.0, -3.0]), self.src.shape_xy, self.src.shape_xy)
        self.dHw = _model_residual("affine", self.wk.shape_xy)

    def test_composition_order_negative_control(self):
        dH_src = T.deltaH_source(self.dHw, self.G)
        correct = dH_src @ self.H0          # DeltaH @ H0
        wrong = self.H0 @ dH_src            # H0 @ DeltaH (generally wrong)
        self.assertGreater(np.max(np.abs(correct - wrong)), 1e-3)

    def test_using_Gr_instead_of_Ga_is_wrong(self):
        # G_r and G_a coincide for divisible dims (the (B-1)/2 offset is
        # shape-independent there), so the distinction only bites when raw and
        # aligned grids genuinely differ. Use a NON-divisible aligned source so
        # the measured offset differs; then conjugating by G_r instead of G_a is
        # demonstrably wrong.
        ali_src = Grid2D.axis_aligned("ali_src", (250, 190), 1.36)  # 250/4 not integer
        wk_a = integer_binned_grid(ali_src, 4)                       # floor -> (62, 47)
        G_a = wk_a.mapping_to(ali_src)
        self.assertGreater(np.max(np.abs(G_a - self.G)), 1e-6, "G_a must differ from G_r here")
        correct = T.deltaH_source(self.dHw, G_a)
        wrong = T.deltaH_source(self.dHw, self.G)  # G_r used by mistake
        self.assertGreater(np.max(np.abs(correct - wrong)), 1e-6)

    def test_naive_translation_scaling_is_wrong(self):
        # Restoring by scaling only the translation by B (ignoring the (B-1)/2
        # centre offset) disagrees with the homogeneous restore.
        dH_src = T.deltaH_source(self.dHw, self.G)
        A_w, t_w = self.dHw[:2, :2], self.dHw[:2, 2]
        naive = np.eye(3); naive[:2, :2] = A_w; naive[:2, 2] = 4.0 * t_w  # B*t only
        self.assertGreater(np.max(np.abs(dH_src - naive)), 1e-6)


if __name__ == "__main__":
    unittest.main()
