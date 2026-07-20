"""Measured MRC geometry layer: real headers, separate raw/aligned grids,
failure modes. Skipped without mrcfile."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    np = None
    HAVE_NUMPY = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pipeline.geometry import (GeometryError, assert_or_override, measure_mrc_grid,
                               measure_source_and_working)

try:
    import mrcfile
    HAVE = HAVE_NUMPY
except Exception:
    HAVE = False


def _mrc(path, nx, ny, n=5, pix=1.36):
    with mrcfile.new(path, overwrite=True) as h:
        h.set_data(np.zeros((n, ny, nx), dtype=np.float32)); h.voxel_size = pix


class HeaderOnlyGeometryTests(unittest.TestCase):
    def test_measure_mrc_grid_uses_header_only_open(self):
        calls = []

        class FakeHandle:
            header = types.SimpleNamespace(
                nx=33, ny=22, nz=7, mode=2,
                origin=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )
            voxel_size = types.SimpleNamespace(x=1.5, y=1.5)

            @property
            def data(self):
                raise AssertionError("data should not be read for header-only geometry")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_open(path, *, permissive=False, header_only=False):
            calls.append({"permissive": permissive, "header_only": header_only})
            return FakeHandle()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "stack.mrc"
            p.write_bytes(b"header placeholder")
            fake_mrcfile = types.SimpleNamespace(open=fake_open)
            with mock.patch.dict(sys.modules, {"mrcfile": fake_mrcfile}):
                measured = measure_mrc_grid(p, role="source_raw")

        self.assertEqual(calls, [{"permissive": True, "header_only": True}])
        self.assertEqual(measured.shape_xy, (33, 22))
        self.assertEqual(measured.n_sections, 7)


@unittest.skipUnless(HAVE, "mrcfile unavailable")
class GeometryLayerTests(unittest.TestCase):
    def test_measures_real_header_and_centre(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.mrc"; _mrc(p, 256, 192, 7, 1.36)
            m = measure_mrc_grid(p, role="source_raw")
            self.assertEqual(m.shape_xy, (256, 192))
            self.assertEqual(m.n_sections, 7)
            self.assertAlmostEqual(m.pixel_size_xy_A[0], 1.36, places=3)
            self.assertEqual(m.center_xy_px, ((256 - 1) / 2, (192 - 1) / 2))  # (n-1)/2
            self.assertTrue(m.sample_all_finite)

    def test_separate_raw_and_aligned_grids(self):
        with tempfile.TemporaryDirectory() as td:
            sr = Path(td) / "raw.mrc"; sa = Path(td) / "ali.mrc"
            _mrc(sr, 1024, 1024, 7, 1.36)   # raw
            _mrc(sa, 960, 928, 7, 1.36)     # aligned cropped (DIFFERENT dims)
            wr = Path(td) / "wraw.mrc"; wa = Path(td) / "wali.mrc"
            _mrc(wr, 128, 128, 7, 10.88)    # bin8 raw
            _mrc(wa, 120, 116, 7, 10.88)    # bin8 aligned
            res = measure_source_and_working(source_raw=sr, source_aligned=sa,
                                             working_raw=wr, working_aligned=wa)
            # The KEY defect-#10 fix: raw and aligned SHAPES are measured
            # separately and genuinely differ (must be used as separate in/out
            # dims in .xf conversion). For same-factor isotropic binning the
            # G maps coincide ((B-1)/2 is shape-independent) -- that is correct.
            self.assertEqual(res["measured"]["source_raw"].shape_xy, (1024, 1024))
            self.assertEqual(res["measured"]["source_aligned"].shape_xy, (960, 928))
            self.assertNotEqual(res["measured"]["source_raw"].shape_xy,
                                res["measured"]["source_aligned"].shape_xy)
            self.assertIn("G_r", res["maps"]); self.assertIn("G_a", res["maps"])
            self.assertEqual(set(res["Q"]), {"source_raw", "source_aligned", "working_raw", "working_aligned"})

    def test_invalid_pixel_size_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "z.mrc"
            with mrcfile.new(p, overwrite=True) as h:
                h.set_data(np.zeros((3, 8, 8), dtype=np.float32))  # voxel_size left 0
            with self.assertRaises(GeometryError):
                measure_mrc_grid(p, role="source_raw")

    def test_corrupt_header_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.mrc"; p.write_bytes(b"not an mrc file")
            with self.assertRaises(GeometryError):
                measure_mrc_grid(p, role="source_raw")

    def test_missing_file_fails(self):
        with self.assertRaises(GeometryError):
            measure_mrc_grid(Path("/nonexistent/x.mrc"), role="source_raw")

    def test_override_mismatch_fails_without_force(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.mrc"; _mrc(p, 256, 192, 5, 1.36)
            m = measure_mrc_grid(p, role="source_raw")
            with self.assertRaises(GeometryError):
                assert_or_override(m, expected_shape_xy=(512, 384))  # config disagrees
            disc = assert_or_override(m, expected_shape_xy=(512, 384), force=True)
            self.assertTrue(disc)  # discrepancy recorded under force


if __name__ == "__main__":
    unittest.main()
