"""Phase 3/5: working raw + aligned stacks (one-pass vs two-pass) and binvol
preview, validated against real IMOD. Skipped (not faked) without IMOD."""
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
from imod_affine import write_xf, xf_to_homogeneous
from multiresolution import Grid2D, Grid3D, integer_binned_grid, preview_grid_from
from multiresolution import transfer as T

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
BINVOL = shutil.which("binvol")
HEADER = shutil.which("header")
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}


def _centroid(img, pxy, half=6):
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


def _run(cmd):
    return subprocess.run(cmd, env=ENV, text=True, capture_output=True)


@unittest.skipUnless(HAVE and NEWSTACK, "newstack/mrcfile unavailable")
class WorkingAlignedTests(unittest.TestCase):
    def test_one_pass_and_two_pass_agree_with_grid(self):
        nx, ny, B = 256, 192, 4
        markers = np.array([[nx * 0.30, ny * 0.34], [nx * 0.62, ny * 0.46], [nx * 0.47, ny * 0.64]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(markers):
            img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.5 ** 2))
        A0 = np.array([[np.cos(np.deg2rad(5)), -np.sin(np.deg2rad(5))],
                       [np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5))]])
        d0 = np.array([6.0, -4.0])
        src = Grid2D.axis_aligned("src", (nx, ny), 1.0)
        wk = integer_binned_grid(src, B)
        G = wk.mapping_to(src)
        H0 = xf_to_homogeneous(A0, d0, (nx, ny), (nx, ny))
        # grid prediction: source raw marker -> working aligned
        pred_w = []
        for cx, cy in markers:
            p = inv_apply(np.linalg.inv(G) @ H0, np.array([cx, cy, 1.0]))
            pred_w.append(p[:2])
        pred_w = np.array(pred_w)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            inp = tdp / "raw.mrc"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32)); h.voxel_size = 1.0
            xf = tdp / "src.xf"; write_xf(xf, A0[None], d0[None])
            # one-pass: xform + shrink in one newstack
            out1 = tdp / "ali_onepass.mrc"
            cp = _run([NEWSTACK, "-input", str(inp), "-output", str(out1), "-xform", str(xf),
                       "-shrink", str(float(B)), "-float", "0"])
            self.assertEqual(cp.returncode, 0, cp.stderr)
            # two-pass: shrink then working xf
            wraw = tdp / "raw_bin.mrc"
            _run([NEWSTACK, "-input", str(inp), "-output", str(wraw), "-shrink", str(float(B)), "-float", "0"])
            H0w = T.h0_working(H0, G, G)
            from imod_affine import homogeneous_to_xf
            Aw, dw = homogeneous_to_xf(H0w, wk.shape_xy, wk.shape_xy)
            wxf = tdp / "work.xf"; write_xf(wxf, Aw[None], dw[None])
            out2 = tdp / "ali_twopass.mrc"
            cp = _run([NEWSTACK, "-input", str(wraw), "-output", str(out2), "-xform", str(wxf), "-float", "0"])
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out1, permissive=True) as h:
                o1 = np.asarray(h.data, float); o1 = o1[0] if o1.ndim == 3 else o1
            with mrcfile.open(out2, permissive=True) as h:
                o2 = np.asarray(h.data, float); o2 = o2[0] if o2.ndim == 3 else o2
        for label, o in (("one-pass", o1), ("two-pass", o2)):
            meas = np.array([_centroid(o, pred_w[k]) for k in range(len(markers))])
            rms = float(np.sqrt(np.mean(np.sum((meas - pred_w) ** 2, 1))))
            self.assertLess(rms, 0.2, f"{label} working aligned rms {rms:.4f}px vs grid prediction")


def inv_apply(M, x):
    return M @ x


@unittest.skipUnless(HAVE and BINVOL and HEADER, "binvol/header unavailable")
class BinvolPreviewTests(unittest.TestCase):
    def test_binvol_xy_preview_grid(self):
        nx, ny, nz = 64, 48, 40
        vol = (np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32))
        working = Grid3D.axis_aligned("working", (nx, ny, nz), 5.44)
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "rec.mrc", Path(td) / "prev.mrc"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(vol); h.voxel_size = 5.44
            cp = _run([BINVOL, "-x", "2", "-y", "2", "-z", "1", str(inp), str(out)])
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data); ovx = float(h.voxel_size.x); ovz = float(h.voxel_size.z)
                shp = (o.shape[2], o.shape[1], o.shape[0])
            # source unmodified
            with mrcfile.open(inp, permissive=True) as h:
                self.assertEqual(h.data.shape, vol.shape)
        preview = preview_grid_from(working, 2, 2, 1, out_shape_xyz=shp)
        self.assertEqual(shp, (nx // 2, ny // 2, nz))      # X/Y binned, Z unchanged
        self.assertAlmostEqual(ovx, 5.44 * 2, places=2)    # X voxel doubled
        self.assertAlmostEqual(ovz, 5.44, places=2)        # Z voxel unchanged
        self.assertTrue(preview.anisotropic, "X/Y-only preview must be marked anisotropic")
        self.assertEqual(preview.role, "visualization_only")


if __name__ == "__main__":
    unittest.main()
