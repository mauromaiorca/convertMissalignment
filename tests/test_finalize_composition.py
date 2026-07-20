"""Final composition math (defect 2.17): Hfinal_source = G_a @ (DeltaH_working @
H0_working) @ inv(G_r), with H0_working = inv(G_a) @ H0_source @ G_r. The exported
final raw->aligned .xf must fold in the initial alignment H0, NOT be the bare
residual. Requires torch."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # noqa: F401
    from pipeline.finalize import _constrained_source_xf
    from imod_affine import xf_to_homogeneous, homogeneous_to_xf
    from alignment_models.registry import get_model
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "torch unavailable")
class CompositionTests(unittest.TestCase):
    def _grids(self):
        # bin-2 isotropic; raw 256x192, aligned 240x176 (distinct)
        from multiresolution import Grid2D, integer_binned_grid
        B = 2
        sr = Grid2D.axis_aligned("source_raw", (256, 192), 1.0)
        sa = Grid2D.axis_aligned("source_aligned", (240, 176), 1.0)
        wr = integer_binned_grid(sr, B); wa = integer_binned_grid(sa, B)
        G_r = wr.mapping_to(sr); G_a = wa.mapping_to(sa)
        return {"G_r": G_r.tolist(), "G_a": G_a.tolist(),
                "source_raw_shape_xy": [256, 192], "source_aligned_shape_xy": [240, 176],
                "working_aligned_shape_xy": list(wa.shape_xy)}

    def test_h0_is_composed_not_dropped(self):
        grids = self._grids()
        model = get_model("rigid")
        n = 4
        params = [[1.2, -0.8, 0.05 + 0.01 * i] for i in range(n)]
        # a NON-identity initial alignment H0_source (raw->aligned), per tilt
        raw_xy, ali_xy = (256, 192), (240, 176)
        h0_rows = []
        for i in range(n):
            A0 = np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]])
            d0 = np.array([3.0 + i, -2.0])
            h0_rows.append((A0, d0))
        # with H0 composed
        _, _, sf_h0 = _constrained_source_xf("rigid", params, list(range(n)), grids, h0_source_rows=h0_rows)
        # without H0 (residual-only route, the OLD wrong behavior)
        _, _, sf_resid = _constrained_source_xf("rigid", params, list(range(n)), grids, h0_source_rows=None)
        # the two must DIFFER (H0 is non-identity -> it changes the final transform)
        diff = max(float(np.abs(np.array(sf_h0[i][1]) - np.array(sf_resid[i][1])).max()) for i in range(n))
        self.assertGreater(diff, 0.5, "H0 composition had no effect; residual was exported as final")

    def test_composition_matches_closed_form(self):
        grids = self._grids()
        G_r = np.asarray(grids["G_r"]); G_a = np.asarray(grids["G_a"])
        raw_xy, ali_xy = (256, 192), (240, 176)
        model = get_model("rigid")
        n = 3
        params = [[0.7, -0.5, 0.04] for _ in range(n)]
        h0_rows = [(np.eye(2), np.array([2.0, -1.0])) for _ in range(n)]
        _, _, sf = _constrained_source_xf("rigid", params, list(range(n)), grids, h0_source_rows=h0_rows)
        # recompute the closed form independently
        work_xy = grids["working_aligned_shape_xy"]
        c = torch_center(work_xy)
        import torch
        DHw = model.homogeneous_physical(torch.tensor(params, dtype=torch.float64),
                                         torch.tensor(c).expand(n, 2)).detach().numpy()
        for i in range(n):
            H0_source = xf_to_homogeneous(h0_rows[i][0], h0_rows[i][1], raw_xy, ali_xy)
            H0_working = np.linalg.inv(G_a) @ H0_source @ G_r
            Hfin_w = DHw[i] @ H0_working
            Hfin_s = G_a @ Hfin_w @ np.linalg.inv(G_r)
            a_exp, d_exp = homogeneous_to_xf(Hfin_s, raw_xy, ali_xy)
            self.assertTrue(np.allclose(sf[i][0], a_exp, atol=1e-9), f"matrix tilt {i}")
            self.assertTrue(np.allclose(sf[i][1], d_exp, atol=1e-6), f"shift tilt {i}")


def torch_center(shape_xy):
    return [(shape_xy[0] - 1) / 2.0, (shape_xy[1] - 1) / 2.0]


if __name__ == "__main__":
    unittest.main()
