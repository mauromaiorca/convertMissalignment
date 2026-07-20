"""Phase 1/8 projection-geometry invariance: P_working = inv(G_d) @ P_source @ G_v
must reproduce an INDEPENDENT physical projection of the same physical point,
across tilts, dims, anisotropy, and pixel sizes."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from multiresolution import Grid2D, Grid3D
from multiresolution import projector as PR
from multiresolution import transfer as T


class ProjectionInvarianceTests(unittest.TestCase):
    TILTS = [-60.0, -30.0, 0.0, 25.0, 55.0]

    def _check(self, vol_s, det_s, vol_w, det_w, axis_angle=0.0, n_random=20, seed=0):
        rng = np.random.default_rng(seed)
        G_v = vol_w.mapping_to(vol_s)
        G_d = det_w.mapping_to(det_s)
        # physical points: centre, axis directions, asymmetric, boundary, random
        ext = np.array(vol_s.shape_xyz) * np.array(vol_s.voxel_size_xyz_A)
        fixed = [
            [0, 0, 0], [ext[0] * 0.3, 0, 0], [0, ext[1] * 0.3, 0], [0, 0, ext[2] * 0.3],
            [ext[0] * 0.4, -ext[1] * 0.25, ext[2] * 0.15],
            [-ext[0] * 0.49, ext[1] * 0.49, -ext[2] * 0.49],  # near boundary
        ]
        rand = (rng.uniform(-0.45, 0.45, (n_random, 3)) * ext)
        phys = np.array(fixed + rand.tolist())
        max_err = 0.0
        for theta in self.TILTS:
            P_s = PR.source_projection_matrix(vol_s, det_s, theta, axis_angle)
            P_w = T.projection_working(P_s, G_d, G_v)
            for X in phys:
                vw = vol_w.physical_to_voxel(X)
                d_via_Pw = T.project_euclidean(P_w, vw)
                d_indep = PR.project_physical_point(X, det_w, theta, axis_angle)
                # also source path must agree with independent source projection
                vs = vol_s.physical_to_voxel(X)
                d_src = T.project_euclidean(P_s, vs)
                d_src_indep = PR.project_physical_point(X, det_s, theta, axis_angle)
                max_err = max(max_err, float(np.max(np.abs(d_via_Pw - d_indep))),
                              float(np.max(np.abs(d_src - d_src_indep))))
        return max_err

    def test_isotropic_bin4_even(self):
        vol_s = Grid3D.axis_aligned("vs", (256, 192, 80), 1.36)
        det_s = Grid2D.axis_aligned("ds", (256, 192), 1.36)
        vol_w = Grid3D.axis_aligned("vw", (64, 48, 80), (5.44, 5.44, 5.44))
        det_w = Grid2D.axis_aligned("dw", (64, 48), 5.44)
        self.assertLess(self._check(vol_s, det_s, vol_w, det_w), 1e-6)

    def test_odd_dims(self):
        vol_s = Grid3D.axis_aligned("vs", (257, 193, 81), 2.0)
        det_s = Grid2D.axis_aligned("ds", (257, 193), 2.0)
        vol_w = Grid3D.axis_aligned("vw", (85, 64, 81), (6.0, 6.0, 2.0))  # bin~3 xy, z same (aniso)
        det_w = Grid2D.axis_aligned("dw", (85, 64), 6.0)
        self.assertLess(self._check(vol_s, det_s, vol_w, det_w), 1e-6)

    def test_anisotropic_preview_grid(self):
        vol_s = Grid3D.axis_aligned("vs", (256, 192, 80), 1.36)
        det_s = Grid2D.axis_aligned("ds", (256, 192), 1.36)
        # anisotropic volume (xy binned, z not) -- preview-like geometry
        vol_w = Grid3D.axis_aligned("vw", (128, 96, 80), (2.72, 2.72, 1.36))
        det_w = Grid2D.axis_aligned("dw", (128, 96), 2.72)
        self.assertTrue(vol_w.anisotropic)
        self.assertLess(self._check(vol_s, det_s, vol_w, det_w), 1e-6)

    def test_nonzero_tilt_axis_angle(self):
        vol_s = Grid3D.axis_aligned("vs", (256, 192, 80), 1.36)
        det_s = Grid2D.axis_aligned("ds", (256, 192), 1.36)
        vol_w = Grid3D.axis_aligned("vw", (64, 48, 80), (5.44, 5.44, 5.44))
        det_w = Grid2D.axis_aligned("dw", (64, 48), 5.44)
        self.assertLess(self._check(vol_s, det_s, vol_w, det_w, axis_angle=12.0), 1e-6)

    def test_different_detector_and_volume_pixel_sizes(self):
        vol_s = Grid3D.axis_aligned("vs", (256, 192, 80), (1.36, 1.36, 2.0))  # aniso source vol
        det_s = Grid2D.axis_aligned("ds", (256, 192), 1.36)
        vol_w = Grid3D.axis_aligned("vw", (64, 48, 80), (5.44, 5.44, 2.0))
        det_w = Grid2D.axis_aligned("dw", (64, 48), 5.44)
        self.assertLess(self._check(vol_s, det_s, vol_w, det_w), 1e-6)


if __name__ == "__main__":
    unittest.main()
