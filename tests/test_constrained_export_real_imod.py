"""Exact .xf export of constrained models validated against REAL IMOD newstack.

A constrained residual (rigid / affine) is exported to a single IMOD .xf row,
applied with the real ``newstack``, and the resulting marker centroids are
compared to the model's own predicted positions. This closes the loop:
model transform -> exact .xf -> real newstack -> measured == predicted.
Skipped (not faked) when newstack/mrcfile/torch are unavailable."""
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

try:
    import torch
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False

NEWSTACK = shutil.which("newstack")

if HAVE:
    import alignment_models as am
    from alignment_models import coordinate_frames as cf
    from alignment_models.serialization import homogeneous_to_xf_rows


def _centroid(img, pxy, half=8):
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


@unittest.skipUnless(HAVE and NEWSTACK, "real newstack / mrcfile / torch unavailable")
class ConstrainedExportRealImodTests(unittest.TestCase):
    def _run(self, model_name, params_row, shape):
        nx, ny = shape
        pix = 10.0
        centres = np.array([[nx * 0.32, ny * 0.30], [nx * 0.60, ny * 0.44], [nx * 0.46, ny * 0.64]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        img = np.zeros((ny, nx), np.float32)
        for k, (cx, cy) in enumerate(centres):
            img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.2 ** 2))

        model = am.get_model(model_name)
        params = torch.tensor([params_row], dtype=torch.float64)
        center_phys = cf.physical_center_xy(shape, pix)

        # Model prediction (physical -> pixel)
        pts_phys = torch.tensor(centres * pix, dtype=torch.float64)
        pred_phys = model.apply_centered(params, pts_phys, torch.tensor(center_phys)).detach().numpy()[0]
        pred_px = pred_phys / pix

        # Exact .xf export (single tilt, ali->final, equal geometry)
        dH = model.homogeneous_physical(params, np.tile(center_phys, (1, 1))).detach().numpy()
        A_xf, d_xf = homogeneous_to_xf_rows(dH, shape, shape, pix, pix)

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            inp, out, xf = tdp / "in.mrc", tdp / "out.mrc", tdp / "t.xf"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32)); h.voxel_size = pix
            xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"
                          % (A_xf[0][0, 0], A_xf[0][0, 1], A_xf[0][1, 0], A_xf[0][1, 1], d_xf[0][0], d_xf[0][1]))
            env = dict(os.environ); env.setdefault("IMOD_DIR", "/Applications/IMOD")
            cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out),
                                 "-xform", str(xf), "-float", "0"], env=env, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float)
            o = o[0] if o.ndim == 3 else o

        errs = [np.linalg.norm(_centroid(o, pred_px[k]) - pred_px[k]) for k in range(len(centres))]
        return float(np.sqrt(np.mean(np.square(errs))))

    def test_rigid_export_matches_newstack(self):
        # tx, ty (A), phi
        for shape in [(256, 192), (257, 193)]:
            rms = self._run("rigid", [40.0, -30.0, np.deg2rad(6.0)], shape)
            self.assertLess(rms, 0.1, f"rigid export vs newstack rms {rms:.4f}px at {shape}")

    def test_affine_export_matches_newstack(self):
        # tx, ty, phi, alpha, beta, shear
        for shape in [(256, 192), (257, 193)]:
            rms = self._run("affine", [25.0, -15.0, np.deg2rad(4.0), 0.05, -0.03, 0.08], shape)
            self.assertLess(rms, 0.1, f"affine export vs newstack rms {rms:.4f}px at {shape}")

    def test_translation_export_matches_newstack(self):
        for shape in [(256, 192), (257, 193)]:
            rms = self._run("translation", [55.0, -40.0], shape)
            self.assertLess(rms, 0.1, f"translation export vs newstack rms {rms:.4f}px at {shape}")

    def test_similarity_export_matches_newstack(self):
        # tx, ty, phi, log_scale
        for shape in [(256, 192), (257, 193)]:
            rms = self._run("similarity", [20.0, -10.0, np.deg2rad(5.0), 0.06], shape)
            self.assertLess(rms, 0.1, f"similarity export vs newstack rms {rms:.4f}px at {shape}")

    def test_different_output_dims_export(self):
        """Export to a different output size (crop and pad) applied with newstack -size.

        Validates the exported .xf's input/output centre handling against real
        newstack by predicting through the exported row (forward_points_pixels)
        and comparing to measured centroids."""
        from imod_affine import forward_points_pixels
        in_shape = (256, 192)
        pix = 10.0
        for out_shape in [(300, 240), (220, 160)]:  # pad, crop
            nx, ny = in_shape
            centres = np.array([[nx * 0.34, ny * 0.30], [nx * 0.58, ny * 0.50], [nx * 0.46, ny * 0.64]])
            yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
            img = np.zeros((ny, nx), np.float32)
            for k, (cx, cy) in enumerate(centres):
                img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.2 ** 2))
            model = am.get_model("affine")
            params = torch.tensor([[18.0, -12.0, np.deg2rad(3.0), 0.04, -0.02, 0.05]], dtype=torch.float64)
            center_phys = cf.physical_center_xy(in_shape, pix)
            dH = model.homogeneous_physical(params, np.tile(center_phys, (1, 1))).detach().numpy()
            A_xf, d_xf = homogeneous_to_xf_rows(dH, in_shape, out_shape, pix, pix)
            pred = forward_points_pixels(centres, A_xf[0], d_xf[0], in_shape, out_shape)
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                inp, out, xf = tdp / "in.mrc", tdp / "out.mrc", tdp / "t.xf"
                with mrcfile.new(inp, overwrite=True) as h:
                    h.set_data(img[None].astype(np.float32)); h.voxel_size = pix
                xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"
                              % (A_xf[0][0, 0], A_xf[0][0, 1], A_xf[0][1, 0], A_xf[0][1, 1], d_xf[0][0], d_xf[0][1]))
                env = dict(os.environ); env.setdefault("IMOD_DIR", "/Applications/IMOD")
                cp = subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out), "-xform", str(xf),
                                     "-size", f"{out_shape[0]},{out_shape[1]}", "-float", "0"],
                                    env=env, text=True, capture_output=True)
                self.assertEqual(cp.returncode, 0, cp.stderr)
                with mrcfile.open(out, permissive=True) as h:
                    o = np.asarray(h.data, float)
                o = o[0] if o.ndim == 3 else o
            errs = [np.linalg.norm(_centroid(o, pred[k]) - pred[k]) for k in range(len(centres))]
            rms = float(np.sqrt(np.mean(np.square(errs))))
            self.assertLess(rms, 0.1, f"different-out-dims {out_shape} export vs newstack rms {rms:.4f}px")


if __name__ == "__main__":
    unittest.main()
