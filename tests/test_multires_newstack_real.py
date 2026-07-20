"""Phase 2: the Grid2D binning model vs REAL newstack (divisible scope).

Confirms the (B-1)/2 grid model reproduces real newstack reduction to <0.05 px
for dimensions divisible by B (factors 2/4/8) and decisively beats the naive
source/B model. Non-divisible dims deviate and are rejected by the CLI (Phase 7).
Skipped (not faked) when newstack/mrcfile unavailable."""
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
from multiresolution import Grid2D, integer_binned_grid

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}


def _centroid(img, pxy, half):
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


@unittest.skipUnless(HAVE and NEWSTACK, "newstack/mrcfile unavailable")
class NewstackBinningModelTests(unittest.TestCase):
    def _shrink_and_measure(self, nx, ny, B, method):
        sigma = 1.3 * B
        centres = np.array([[nx * 0.33, ny * 0.36], [nx * 0.64, ny * 0.62]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(centres):
            img += (1.0 + 0.25 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.mrc", Path(td) / "out.mrc"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32)); h.voxel_size = 1.0
            flag = ["-shrink", str(float(B))] if method == "shrink" else ["-bin", str(B)]
            cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out), *flag, "-float", "0"],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float); o = o[0] if o.ndim == 3 else o
            out_ny, out_nx = o.shape
        src = Grid2D.axis_aligned("src", (nx, ny), 1.0)
        wk = integer_binned_grid(src, B, out_shape_xy=(out_nx, out_ny))
        G_s2w = np.linalg.inv(wk.mapping_to(src))
        half = max(4, int(round(3.0 * sigma / B)))
        ge, ne = [], []
        for cx, cy in centres:
            pg = (G_s2w @ np.array([cx, cy, 1.0]))[:2]
            pn = np.array([cx / B, cy / B])
            m = _centroid(o, pg, half)
            if np.any(np.isnan(m)):
                continue
            ge.append(np.linalg.norm(m - pg)); ne.append(np.linalg.norm(m - pn))
        return float(np.sqrt(np.mean(np.square(ge)))), float(np.sqrt(np.mean(np.square(ne)))), (out_nx, out_ny)

    def test_divisible_grid_model_matches_newstack(self):
        for nx, ny in [(256, 192), (512, 384)]:
            for B in (2, 4, 8):
                for method in ("shrink", "bin"):
                    grid_rms, naive_rms, out_dims = self._shrink_and_measure(nx, ny, B, method)
                    self.assertEqual(out_dims, (nx // B, ny // B), f"{nx}x{ny}/{B} dims")
                    self.assertLess(grid_rms, 0.05, f"{nx}x{ny}/{B}/{method}: grid {grid_rms:.4f}px")
                    self.assertGreater(naive_rms, 5 * grid_rms, "naive must be clearly worse")

    def test_naive_translation_scaling_is_inadequate(self):
        # The (B-1)/2 offset is real: naive source/B is off by ~0.3-0.6 px.
        grid_rms, naive_rms, _ = self._shrink_and_measure(256, 192, 4, "shrink")
        self.assertGreater(naive_rms, 0.3)
        self.assertLess(grid_rms, 0.05)


if __name__ == "__main__":
    unittest.main()
