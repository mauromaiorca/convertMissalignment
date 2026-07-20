"""affine2d Warp-result restore to source + final source-resolution chain
(aligned stack, final CTF, reconstruction) with real IMOD. Skipped without IMOD."""
from __future__ import annotations

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
from imod_affine import forward_points_pixels, read_xf, write_xf, xf_to_homogeneous
from multiresolution import Grid2D, build_plan
from multiresolution import transfer as T

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack"); TILT = shutil.which("tilt"); CTF = shutil.which("ctfphaseflip")
EXPORT = ROOT / "export_warp_to_imod.py"
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _centroid(img, pxy, half=7):
    ny, nx = img.shape
    x0, y0 = int(round(pxy[0])), int(round(pxy[1]))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1); ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = np.clip(img[ya:yb, xa:xb].astype(np.float64), 0, None)
    if sub.sum() <= 1e-9:
        return np.array([np.nan, np.nan])
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return np.array([(sub * xs).sum() / sub.sum(), (sub * ys).sum() / sub.sum()])


def _rot(d):
    a = np.deg2rad(d); return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@unittest.skipUnless(HAVE and NEWSTACK, "newstack/mrcfile unavailable")
class Affine2dRestoreTests(unittest.TestCase):
    def test_affine2d_warp_restore_vs_newstack(self):
        src_dims = (512, 384); B = 4
        sr = Grid2D.axis_aligned("source_raw", src_dims, 1.0)
        sa = Grid2D.axis_aligned("source_aligned", src_dims, 1.0)
        plan = build_plan(B, sr, sa)
        n = 3
        # simulate an affine2d Warp result: working raw->final affine .xf per tilt
        Aw = np.stack([_rot(np.deg2rad(2 + i)) @ np.array([[1.02, 0.03], [-0.01, 0.99]]) for i in range(n)])
        dw = np.stack([np.array([1.5 + i, -1.0 - i]) for i in range(n)])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            wxf = tmp / "working_raw_to_final.xf"; write_xf(wxf, Aw, dw)
            settings = tmp / "s.toml"
            settings.write_text(f'''
[project]
basename = "mr"
[paths]
output_dir = "{tmp.as_posix()}/out"
[geometry]
raw_dimensions_xyz = [{src_dims[0]},{src_dims[1]},1]
raw_pixel_size_A = 1.0
aligned_dimensions_xyz = [{src_dims[0]},{src_dims[1]},1]
aligned_pixel_size_A = 1.0
[multiresolution]
enabled = true
extra_projection_binning = {B}
[refinement]
model = "rigid"
''')
            cp = subprocess.run([sys.executable, str(EXPORT), str(settings), "--condition", "ali_identity",
                                 "--result-type", "affine2d-warp", "--working-xf", str(wxf), "--out-dir", str(tmp / "exp")],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            sxf = list((tmp / "exp").glob("*final_source_raw_to_aligned.xf"))[0]
            As, ds = read_xf(sxf)
            # verify against real newstack: source .xf maps source markers where the working route predicts
            markers = np.array([[src_dims[0] * 0.3, src_dims[1] * 0.34], [src_dims[0] * 0.6, src_dims[1] * 0.5]])
            yy, xx = np.mgrid[0:src_dims[1], 0:src_dims[0]].astype(np.float64)
            img = np.zeros(src_dims[::-1], np.float32)
            for k, (cx, cy) in enumerate(markers):
                img += (1 + 0.2 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.4 ** 2))
            pred = forward_points_pixels(markers, As[0], ds[0], src_dims, src_dims)
            inp, out, xf = tmp / "in.mrc", tmp / "o.mrc", tmp / "t.xf"
            with mrcfile.new(inp, overwrite=True) as h:
                h.set_data(img[None].astype(np.float32)); h.voxel_size = 1.0
            xf.write_text("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n" % (As[0][0,0],As[0][0,1],As[0][1,0],As[0][1,1],ds[0][0],ds[0][1]))
            subprocess.run([NEWSTACK, "-input", str(inp), "-output", str(out), "-xform", str(xf), "-float", "0"],
                           env=ENV, text=True, capture_output=True, check=True)
            with mrcfile.open(out, permissive=True) as h:
                o = np.asarray(h.data, float); o = o[0] if o.ndim == 3 else o
            rms = float(np.sqrt(np.mean([np.sum((_centroid(o, pred[k]) - pred[k]) ** 2) for k in range(len(markers))])))
            self.assertLess(rms, 0.15, f"affine2d restore vs newstack rms {rms:.4f}px")
            # independent check: source == G_a @ Hfinal_working @ inv(G_r)
            Hfw = xf_to_homogeneous(Aw[0], dw[0], plan.working_raw.shape_xy, plan.working_aligned.shape_xy)
            Hfs = T.restore_hfinal_working_to_source(Hfw, plan.G_a, plan.G_r)
            from imod_affine import homogeneous_to_xf
            a, d = homogeneous_to_xf(Hfs, src_dims, src_dims)
            self.assertTrue(np.allclose(a, As[0], atol=1e-6) and np.allclose(d, ds[0], atol=1e-3))

    def test_restore_uses_separate_aligned_grid(self):
        # defect #10: raw (512x384) != aligned (480x352). The restore must measure
        # the aligned stack header and (a) record SEPARATE raw/aligned grids in the
        # report and (b) convert to .xf using the ALIGNED output dims. For equal-pixel
        # isotropic binning the *centered* .xf coefficients are themselves invariant
        # ((B-1)/2 is shape-independent), so the demonstrable effect is correct
        # provenance + output-dim targeting -- which is exactly what is verified.
        raw_dims = (512, 384); ali_dims = (480, 352); B = 4
        sr = Grid2D.axis_aligned("source_raw", raw_dims, 1.0)
        sa = Grid2D.axis_aligned("source_aligned", ali_dims, 1.0)
        plan = build_plan(B, sr, sa)
        n = 3
        Aw = np.stack([_rot(np.deg2rad(2 + i)) @ np.array([[1.02, 0.03], [-0.01, 0.99]]) for i in range(n)])
        dw = np.stack([np.array([1.5 + i, -1.0 - i]) for i in range(n)])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            wxf = tmp / "working_raw_to_final.xf"; write_xf(wxf, Aw, dw)
            ali_mrc = tmp / "mr_ali.mrc"
            with mrcfile.new(ali_mrc, overwrite=True) as h:  # real aligned header to be measured
                h.set_data(np.zeros((n, ali_dims[1], ali_dims[0]), np.float32)); h.voxel_size = 1.0
            settings = tmp / "s.toml"
            settings.write_text(f'''
[project]
basename = "mr"
[paths]
output_dir = "{tmp.as_posix()}/out"
[input]
aligned_stack = "{ali_mrc.as_posix()}"
[geometry]
raw_dimensions_xyz = [{raw_dims[0]},{raw_dims[1]},1]
raw_pixel_size_A = 1.0
[multiresolution]
enabled = true
extra_projection_binning = {B}
[refinement]
model = "rigid"
''')
            cp = subprocess.run([sys.executable, str(EXPORT), str(settings), "--condition", "ali_identity",
                                 "--result-type", "affine2d-warp", "--working-xf", str(wxf), "--out-dir", str(tmp / "exp")],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            import json as _json
            report = _json.loads((tmp / "exp" / "affine2d_restore_report.json").read_text())
            self.assertEqual(report["source_raw_dims"], list(raw_dims))
            self.assertEqual(report["source_aligned_dims"], list(ali_dims))  # MEASURED, not raw
            self.assertNotEqual(report["source_raw_dims"], report["source_aligned_dims"])
            sxf = list((tmp / "exp").glob("*final_source_raw_to_aligned.xf"))[0]
            As, ds = read_xf(sxf)
            # the exported source xf must equal the separate-grid transfer (in=raw, out=aligned)
            Hfw = xf_to_homogeneous(Aw[0], dw[0], plan.working_raw.shape_xy, plan.working_aligned.shape_xy)
            Hfs = T.restore_hfinal_working_to_source(Hfw, plan.G_a, plan.G_r)
            from imod_affine import homogeneous_to_xf
            a, d = homogeneous_to_xf(Hfs, raw_dims, ali_dims)
            self.assertTrue(np.allclose(a, As[0], atol=1e-6), f"matrix mismatch {a} vs {As[0]}")
            self.assertTrue(np.allclose(d, ds[0], atol=1e-3), f"shift mismatch {d} vs {ds[0]}")
            # the output-dim choice is material to the .xf representation: converting the
            # SAME source transform with the WRONG (raw) output dims yields a different
            # centered .xf -- so measuring the aligned dims is required for correctness.
            _, d_rawout = homogeneous_to_xf(Hfs, raw_dims, raw_dims)
            self.assertFalse(np.allclose(d, d_rawout, atol=1e-3),
                             "aligned vs raw output dims gave identical .xf; aligned-grid choice "
                             "would be irrelevant (it is not)")


@unittest.skipUnless(HAVE and NEWSTACK and TILT, "newstack/tilt unavailable")
class FinalSourceChainTests(unittest.TestCase):
    def test_final_aligned_from_source_raw_then_reconstruction(self):
        nx, ny, n = 256, 192, 7
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            raw = np.random.default_rng(0).normal(0, 1, (n, ny, nx)).astype(np.float32)
            src_raw = tmp / "raw.st"
            with mrcfile.new(src_raw, overwrite=True) as h:
                h.set_data(raw); h.voxel_size = 1.36
            # final source raw->aligned .xf (small rotations+shifts)
            A = np.stack([_rot(np.deg2rad(1 + 0.5 * i)) for i in range(n)]); d = np.stack([np.array([2.0, -1.0])] * n)
            fxf = tmp / "final.xf"; write_xf(fxf, A, d)
            # final aligned stack FROM SOURCE RAW (not working), never overwriting source
            final_ali = tmp / "final_source_aligned_uncorrected.mrc"
            cp = subprocess.run([NEWSTACK, "-input", str(src_raw), "-output", str(final_ali),
                                 "-xform", str(fxf), "-mode", "2"], env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertTrue(src_raw.is_file())  # source not overwritten
            with mrcfile.open(final_ali, permissive=True) as h:
                self.assertEqual(h.data.shape, (n, ny, nx))
                self.assertEqual(round(float(h.voxel_size.x), 2), 1.36)  # SOURCE pixel, not working
            # final reconstruction at SOURCE geometry (tilt)
            tlt = tmp / "f.tlt"; tlt.write_text("\n".join(f"{a:.2f}" for a in np.linspace(-60, 60, n)) + "\n")
            rec = tmp / "final_source.rec"
            cp = subprocess.run([TILT, "-input", str(final_ali), "-output", str(rec), "-TILTFILE", str(tlt),
                                 "-THICKNESS", "40", "-IMAGEBINNED", "1", "-RADIAL", "0.35 0.05"],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr[-400:])
            with mrcfile.open(rec, permissive=True) as h:
                self.assertEqual(h.data.shape[2], nx)  # source width, not working


@unittest.skipUnless(HAVE and CTF and NEWSTACK, "ctfphaseflip/newstack unavailable")
class FinalCtfTests(unittest.TestCase):
    def test_final_ctf_uses_source_pixel(self):
        from pipeline import ctf as C
        nx, ny, n = 96, 96, 5
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            yy, xx = np.mgrid[0:ny, 0:nx]
            base = (np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype(np.float32)
            final_ali = tmp / "final_ali.mrc"
            with mrcfile.new(final_ali, overwrite=True) as h:
                h.set_data(np.stack([base * (1 + 0.01 * i) for i in range(n)]).astype(np.float32)); h.voxel_size = 1.36
            ang = np.linspace(-60, 60, n)
            (tmp / "t.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
            (tmp / "d.defocus").write_text("\n".join("%d %d %.2f %.2f %.1f" % (i+1,i+1,ang[i],ang[i],5000.0) for i in range(n)) + "\n")
            out = tmp / "final_ctf.mrc"
            cmd = C.build_ctfphaseflip_cmd(input_stack=final_ali, output_stack=out, angle_file=tmp / "t.tlt",
                                           defocus_file=tmp / "d.defocus", pixel_size_A=1.36, unbinned_pixel_A=1.36)
            cp = C.run_ctfphaseflip(cmd)
            self.assertEqual(cp.returncode, 0, cp.stdout[-300:])
            rep = C.validate_ctf_output(final_ali, out)
            self.assertTrue(rep["ok"], rep)
            self.assertEqual(round(rep["pixel_A"], 2), 1.36)  # final CTF at SOURCE pixel


if __name__ == "__main__":
    unittest.main()
