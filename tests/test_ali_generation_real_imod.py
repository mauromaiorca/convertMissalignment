"""Phase 6 - automatic .ali generation validated against REAL IMOD newstack.

This exercises ``generate_aligned_stack.py`` with the installed ``newstack``
(no stub) and verifies that:

* the generated ``.ali`` has the correct tilt count, dimensions and pixel size;
* marker centroids in the ``.ali`` match the production forward map
  ``forward_points_pixels`` (corrected (n-1)/2 IMOD centre convention), i.e. the
  generated stack is geometrically what the conversion math assumes;
* the params JSON is rewired into a usable ``ali_identity`` condition.

Skipped (not faked) when ``newstack`` / ``mrcfile`` are unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from imod_affine import forward_points_pixels, write_xf

try:
    import mrcfile
    HAVE_MRCFILE = True
except Exception:
    HAVE_MRCFILE = False

NEWSTACK = shutil.which("newstack")
GEN = ROOT / "scripts" / "generate_aligned_stack.py"


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


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


@unittest.skipUnless(NEWSTACK and HAVE_MRCFILE, "real IMOD newstack / mrcfile unavailable")
class AliGenerationRealImodTests(unittest.TestCase):
    def _build_project(self, tmp: Path, nx=256, ny=192):
        series = "ali_real"
        etomo = tmp / "etomo"
        etomo.mkdir()
        n = 4
        centres = np.array([[nx * 0.30, ny * 0.34], [nx * 0.63, ny * 0.42], [nx * 0.45, ny * 0.66]])
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        stack = np.zeros((n, ny, nx), np.float32)
        for t in range(n):
            img = np.zeros((ny, nx), np.float64)
            for k, (cx, cy) in enumerate(centres):
                img += (1.0 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.5 ** 2))
            stack[t] = img
        raw = etomo / f"{series}.st"
        with mrcfile.new(raw, overwrite=True) as h:
            h.set_data(stack)
            h.voxel_size = 10.0
        # Per-tilt distinct transforms (rotation + translation).
        mats, shifts = [], []
        for t in range(n):
            mats.append(_rot(2.5 * (t + 1)))
            shifts.append(np.array([1.5 * t, -1.0 * t]))
        xf = etomo / f"{series}.xf"
        write_xf(xf, np.stack(mats), np.stack(shifts))
        tlt = etomo / f"{series}.tlt"
        tlt.write_text("\n".join(str(v) for v in np.linspace(-30, 30, n)) + "\n")
        params = {
            "series_name": series,
            "imod_dir": str(etomo),
            "files": {
                "raw_stack": str(raw),
                "final_xf": str(xf),
                "final_tilt": str(tlt),
            },
            "geometry": {
                "raw_pixel_size_A": 10.0,
                "target_output_pixel_size_A": 10.0,
                "target_volume_shape_xyz": [nx, ny, 8],
            },
            "conditions": {},
        }
        params_path = tmp / "params.json"
        params_path.write_text(json.dumps(params, indent=2))
        return series, raw, xf, params_path, np.stack(mats), np.stack(shifts), centres, (nx, ny), n

    def test_generates_geometrically_correct_ali(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            series, raw, xf, params_path, mats, shifts, centres, (nx, ny), n = self._build_project(tmp)
            out = tmp / "generated_inputs" / f"{series}_aligned.ali"
            env = dict(os.environ)
            env.setdefault("IMOD_DIR", "/Applications/IMOD")
            cp = subprocess.run(
                [sys.executable, str(GEN), "--params", str(params_path),
                 "--output", str(out), "--module-mode", "never", "--no-use-newst-com", "--overwrite"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, f"{cp.stdout}\n{cp.stderr}")
            self.assertTrue(out.is_file())

            with mrcfile.open(out, permissive=True) as h:
                ali = np.asarray(h.data, float)
                pixel = float(h.voxel_size.x)
            self.assertEqual(ali.shape[0], n, "tilt count preserved")
            self.assertEqual((ali.shape[2], ali.shape[1]), (nx, ny), "X,Y preserved")
            self.assertAlmostEqual(pixel, 10.0, places=3)

            # Each section's markers must land where the corrected forward map says.
            max_err = 0.0
            for t in range(n):
                pred = forward_points_pixels(centres, mats[t], shifts[t], (nx, ny), (nx, ny))
                for p in pred:
                    meas = _centroid(ali[t], p)
                    self.assertFalse(np.any(np.isnan(meas)))
                    max_err = max(max_err, float(np.linalg.norm(meas - p)))
            self.assertLess(max_err, 0.12, f"ali markers vs forward map max err {max_err:.4f}px")

            # Params JSON rewired into a usable ali_identity condition.
            params = json.loads(params_path.read_text())
            cond = params["conditions"]["ali_identity"]
            self.assertEqual(cond["xf_file"], "IDENTITY")
            self.assertEqual(cond["alignment_mode"], "identity")
            self.assertEqual(cond["axis_frame"], "aligned")
            self.assertEqual(Path(cond["stack"]).resolve(), out.resolve())
            self.assertEqual(Path(cond["source_xf_file"]).resolve(), xf.resolve())
            self.assertEqual(params["counts"]["aligned_stack_tilts"], n)

    def test_newst_com_size_to_output_is_preserved(self):
        """A newst.com with SizeToOutput must drive the output dimensions."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            series, raw, xf, params_path, mats, shifts, centres, (nx, ny), n = self._build_project(tmp)
            out_nx, out_ny = 300, 240
            newst_com = tmp / "etomo" / "newst.com"
            newst_com.write_text(
                "$newstack -StandardInput\n"
                "InputFile placeholder.st\n"
                "OutputFile placeholder.ali\n"
                "TransformFile placeholder.xf\n"
                f"SizeToOutput {out_nx},{out_ny}\n"
                "ModeToOutput 2\n"
                "$b3dcopy -p placeholder.ali placeholder.ali\n"
            )
            out = tmp / "generated_inputs" / f"{series}_aligned.ali"
            env = dict(os.environ)
            env.setdefault("IMOD_DIR", "/Applications/IMOD")
            cp = subprocess.run(
                [sys.executable, str(GEN), "--params", str(params_path),
                 "--output", str(out), "--module-mode", "never",
                 "--newst-com", str(newst_com), "--overwrite"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, f"{cp.stdout}\n{cp.stderr}")
            with mrcfile.open(out, permissive=True) as h:
                ali = np.asarray(h.data, float)
            self.assertEqual((ali.shape[2], ali.shape[1]), (out_nx, out_ny),
                             "SizeToOutput from newst.com must set .ali X,Y")


if __name__ == "__main__":
    unittest.main()
