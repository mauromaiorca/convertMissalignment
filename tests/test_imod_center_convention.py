"""Regression tests for the IMOD image-centre convention.

Empirical finding (Phase 4, real IMOD ``newstack`` 5.1.9): IMOD applies a
``.xf`` about the geometric image centre ``(nx-1)/2, (ny-1)/2`` in 0-based
pixel coordinates (equivalently ``(nx+1)/2`` in IMOD's 1-based coordinates),
NOT ``nx/2, ny/2``.  Using ``nx/2`` introduces a systematic 0.5-pixel centre
error, attenuated by ``(I - A)`` for the transform ``A``.

The fast unit test pins the convention to the empirically verified value and
fails against the previous ``nx/2`` default.  The IMOD-backed test re-derives
the convention from the installed ``newstack`` and is skipped (not faked) when
IMOD is unavailable.
"""
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
from imod_affine import forward_points_pixels, image_center_xy

try:
    import mrcfile
    HAVE_MRCFILE = True
except Exception:
    HAVE_MRCFILE = False

NEWSTACK = shutil.which("newstack")


class ImodCentreConventionUnitTest(unittest.TestCase):
    """Fast assertion of the empirically verified IMOD centre convention."""

    def test_even_and_odd_centre_is_n_minus_1_over_2(self):
        # Even dims
        self.assertTrue(np.allclose(image_center_xy((256, 192), "imod"), [127.5, 95.5]))
        # Odd dims
        self.assertTrue(np.allclose(image_center_xy((257, 193), "imod"), [128.0, 96.0]))

    def test_pixel_center_alias_matches_imod(self):
        for shape in [(256, 192), (257, 193), (300, 240)]:
            self.assertTrue(
                np.allclose(
                    image_center_xy(shape, "imod"),
                    image_center_xy(shape, "pixel-center"),
                )
            )


def _rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _centroid(img: np.ndarray, pxy, half: int = 9) -> np.ndarray:
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


@unittest.skipUnless(NEWSTACK and HAVE_MRCFILE, "real IMOD newstack / mrcfile unavailable")
class ImodCentreConventionAgainstNewstack(unittest.TestCase):
    """Re-derive the centre convention from the installed newstack.

    Uses a large (25 deg) rotation so a 0.5-pixel centre error becomes ~0.2 px,
    far above the centroid-measurement noise floor (~0.01 px).
    """

    def _run(self, nx, ny):
        A = _rot(25.0)
        d = np.array([0.0, 0.0])
        centres = np.array([[nx * 0.34, ny * 0.30], [nx * 0.58, ny * 0.44], [nx * 0.46, ny * 0.62]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(centres):
            img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            inp, out, xf = tdp / "in.mrc", tdp / "out.mrc", tdp / "t.xf"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32))
                h.voxel_size = 1.0
            xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"
                          % (A[0, 0], A[0, 1], A[1, 0], A[1, 1], d[0], d[1]))
            env = dict(os.environ)
            env.setdefault("IMOD_DIR", "/Applications/IMOD")
            cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out),
                                 "-xform", str(xf), "-float", "0"],
                                env=env, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float)
            o = o[0] if o.ndim == 3 else o
        err_n2, err_nm1 = [], []
        for cx, cy in centres:
            ci = np.array([nx, ny], float)
            pred_n2 = (np.array([cx, cy]) - ci / 2) @ A.T + ci / 2
            pred_nm1 = (np.array([cx, cy]) - (ci - 1) / 2) @ A.T + (ci - 1) / 2
            meas = _centroid(o, pred_nm1)
            err_n2.append(np.linalg.norm(meas - pred_n2))
            err_nm1.append(np.linalg.norm(meas - pred_nm1))
        return float(np.sqrt(np.mean(np.square(err_n2)))), float(np.sqrt(np.mean(np.square(err_nm1))))

    def test_newstack_uses_n_minus_1_over_2(self):
        for nx, ny in [(256, 192), (257, 193)]:
            rms_n2, rms_nm1 = self._run(nx, ny)
            self.assertLess(rms_nm1, 0.05, f"(n-1)/2 should fit newstack for {nx}x{ny}")
            self.assertGreater(rms_n2, 0.15, f"n/2 should NOT fit newstack for {nx}x{ny}")

    def test_production_forward_map_matches_newstack(self):
        # forward_points_pixels (default 'imod') must match real newstack to <0.05px.
        nx, ny = 257, 193
        A = _rot(25.0)
        d = np.array([0.0, 0.0])
        centres = np.array([[nx * 0.34, ny * 0.30], [nx * 0.58, ny * 0.44], [nx * 0.46, ny * 0.62]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(centres):
            img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            inp, out, xf = tdp / "in.mrc", tdp / "out.mrc", tdp / "t.xf"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32))
                h.voxel_size = 1.0
            xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"
                          % (A[0, 0], A[0, 1], A[1, 0], A[1, 1], d[0], d[1]))
            env = dict(os.environ)
            env.setdefault("IMOD_DIR", "/Applications/IMOD")
            cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out),
                                 "-xform", str(xf), "-float", "0"],
                                env=env, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float)
            o = o[0] if o.ndim == 3 else o
        pred = forward_points_pixels(centres, A, d, (nx, ny), (nx, ny))
        errs = [np.linalg.norm(_centroid(o, p) - p) for p in pred]
        self.assertLess(float(np.sqrt(np.mean(np.square(errs)))), 0.05)


if __name__ == "__main__":
    unittest.main()
