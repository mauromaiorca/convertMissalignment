"""Phase 6/18: restore working-grid residuals to the source grid and validate the
final source .xf with REAL newstack, for translation/rigid/similarity/affine.
Includes mutation/negative controls. Skipped (not faked) without newstack."""
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
from imod_affine import forward_points_pixels, write_xf, xf_to_homogeneous
from multiresolution import Grid2D, integer_binned_grid
from multiresolution import transfer as T
from multiresolution.restore import restore_residual_to_source

try:
    import torch
    import mrcfile
    import alignment_models as am
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}


def _centroid(img, pxy, half=7):
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


def _residual_params(model_name, n):
    m = am.get_model(model_name)
    base = {
        "translation": [60.0, -40.0],
        "rigid": [40.0, -25.0, np.deg2rad(4.0)],
        "similarity": [30.0, -20.0, np.deg2rad(3.0), 0.04],
        "affine": [25.0, -15.0, np.deg2rad(3.0), 0.04, -0.02, 0.05],
    }[model_name]
    # mild per-tilt variation
    rows = [[v * (1.0 + 0.03 * i) for v in base] for i in range(n)]
    return m, torch.tensor(rows, dtype=torch.float64)


@unittest.skipUnless(HAVE and NEWSTACK, "newstack/torch/mrcfile unavailable")
class RestoreRealImodTests(unittest.TestCase):
    def _grids(self, src_dims, B):
        sr = Grid2D.axis_aligned("source_raw", src_dims, 1.0, role="source_raw")
        sa = Grid2D.axis_aligned("source_aligned", src_dims, 1.0, role="source_aligned")
        wr = integer_binned_grid(sr, B, name="working_raw", role="working_raw")
        wa = integer_binned_grid(sa, B, name="working_aligned", role="working_aligned")
        return sr, sa, wr, wa

    def _newstack_centroids(self, src_dims, A_final, d_final, markers):
        nx, ny = src_dims
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(markers):
            img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.4 ** 2))
        pred = forward_points_pixels(markers, A_final, d_final, src_dims, src_dims)
        with tempfile.TemporaryDirectory() as td:
            inp, out, xf = Path(td) / "in.mrc", Path(td) / "o.mrc", Path(td) / "t.xf"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32)); h.voxel_size = 1.0
            xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"
                          % (A_final[0, 0], A_final[0, 1], A_final[1, 0], A_final[1, 1], d_final[0], d_final[1]))
            cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out), "-xform", str(xf), "-float", "0"],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float); o = o[0] if o.ndim == 3 else o
        return np.array([_centroid(o, pred[k]) for k in range(len(markers))]), pred

    def test_restore_all_models_vs_newstack(self):
        for src_dims in [(256, 192), (512, 384)]:
            nx, ny = src_dims
            markers = np.array([[nx * 0.3, ny * 0.34], [nx * 0.62, ny * 0.44], [nx * 0.47, ny * 0.66]])
            for B in (2, 4, 8):
                sr, sa, wr, wa = self._grids(src_dims, B)
                G_r, G_a = wr.mapping_to(sr), wa.mapping_to(sa)
                n = 3
                # source H0 (raw->aligned)
                A0 = np.stack([np.array([[np.cos(np.deg2rad(2 + i)), -np.sin(np.deg2rad(2 + i))],
                                         [np.sin(np.deg2rad(2 + i)), np.cos(np.deg2rad(2 + i))]]) for i in range(n)])
                d0 = np.stack([np.array([3.0 + i, -2.0 - i]) for i in range(n)])
                with tempfile.TemporaryDirectory() as td:
                    h0xf = Path(td) / "h0.xf"; write_xf(h0xf, A0, d0)
                    for model_name in ("translation", "rigid", "similarity", "affine"):
                        m, params = _residual_params(model_name, n)
                        info, A_final, d_final = restore_residual_to_source(
                            model=m, params=params, source_raw=sr, source_ali=sa,
                            working_raw=wr, working_ali=wa, source_h0_xf=h0xf)
                        # tilt 0 vs real newstack
                        meas, pred = self._newstack_centroids(src_dims, A_final[0], d_final[0], markers)
                        rms = float(np.sqrt(np.mean(np.sum((meas - pred) ** 2, 1))))
                        self.assertLess(rms, 0.15, f"{model_name} B={B} {src_dims}: newstack rms {rms:.4f}px")
                        # independent working-route prediction must match Hfinal_source
                        H0_source = xf_to_homogeneous(A0[0], d0[0], src_dims, src_dims)
                        dHw = np.array(info["deltaH_working"][0])
                        H0w = T.h0_working(H0_source, G_r, G_a)
                        route = T.hfinal_source_via_working(H0w, dHw, G_a, G_r)
                        Hf = np.array(info["hfinal_source"][0])
                        self.assertTrue(np.allclose(route, Hf, atol=1e-9), f"{model_name} B={B} two-route")

    def test_restore_mutations_detected(self):
        """Each restore mutation must change the result vs the correct working route."""
        src_dims = (256, 192)
        sr, sa, wr, wa = self._grids(src_dims, 4)
        G_r, G_a = wr.mapping_to(sr), wa.mapping_to(sa)
        # different source raw vs aligned dims so G_r != G_a
        sa2 = Grid2D.axis_aligned("source_aligned", (240, 200), 1.0)
        wa2 = integer_binned_grid(sa2, 4)
        G_a2 = wa2.mapping_to(sa2)
        n = 2
        m, params = _residual_params("affine", n)
        from alignment_models.serialization import homogeneous_to_xf_rows
        from alignment_models import coordinate_frames as cf
        centre = cf.physical_center_xy(wa2.shape_xy, wa2.pixel_size_xy_A[0])
        H = m.homogeneous_physical(params, np.tile(centre, (n, 1))).detach().numpy()
        Ar, dr = homogeneous_to_xf_rows(H, wa2.shape_xy, wa2.shape_xy, wa2.pixel_size_xy_A[0], wa2.pixel_size_xy_A[0])
        dHw = xf_to_homogeneous(Ar[0], dr[0], wa2.shape_xy, wa2.shape_xy)
        H0 = xf_to_homogeneous(np.array([[np.cos(0.07), -np.sin(0.07)], [np.sin(0.07), np.cos(0.07)]]),
                               np.array([4.0, -3.0]), src_dims, sa2.shape_xy)

        correct = T.deltaH_source(dHw, G_a2) @ H0           # DeltaH @ H0, conjugate by G_a
        # NOTE: in the supported scope (isotropic integer binning, shared centre)
        # G_r == G_a == [[B,0,(B-1)/2],[0,B,(B-1)/2],[0,0,1]] for ANY divisible
        # dims, so "use G_r instead of G_a" is a no-op there; that distinction is
        # only detectable with non-divisible/different grids and is covered by
        # test_multires_grids::test_using_Gr_instead_of_Ga_is_wrong.
        naive = np.eye(3); naive[:2, :2] = dHw[:2, :2]; naive[:2, 2] = 4.0 * dHw[:2, 2]
        mutations = {
            "wrong_order_H0_at_Delta": H0 @ T.deltaH_source(dHw, G_a2),
            "no_conjugation": dHw @ H0,
            "scale_translation_only": naive @ H0,
            "identity_residual_ignored": H0,  # forgetting to apply the residual
        }
        for name, mut in mutations.items():
            self.assertGreater(np.max(np.abs(correct - mut)), 1e-6, f"mutation {name} not detected")


if __name__ == "__main__":
    unittest.main()
