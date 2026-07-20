"""Phase 7 - adversarial and failure-path tests.

These verify that malformed input fails loudly with actionable errors rather
than silently producing wrong geometry, and that the non-affine export
safeguard rejects fields an IMOD ``.xf`` cannot represent.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from imod_affine import (
    fit_affine,
    read_xf,
    residual_statistics,
    write_xf,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class ReadXfFailureTests(unittest.TestCase):
    def test_singular_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td) / "s.xf",
                       "   0.0000000   0.0000000   0.0000000   0.0000000   0.000   0.000\n")
            with self.assertRaises(ValueError) as ctx:
                read_xf(p)
            self.assertIn("singular", str(ctx.exception).lower())

    def test_near_singular_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            # determinant ~ 1e-12
            p = _write(Path(td) / "ns.xf",
                       "   1.0000000   0.0000000   0.0000000   0.0000000000001   0.000   0.000\n")
            with self.assertRaises(ValueError):
                read_xf(p)

    def test_nan_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td) / "n.xf",
                       "   1.0000000   0.0000000   0.0000000   nan   0.000   0.000\n")
            with self.assertRaises(ValueError):
                read_xf(p)

    def test_inf_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td) / "i.xf",
                       "   1.0000000   0.0000000   0.0000000   inf   0.000   0.000\n")
            with self.assertRaises(ValueError):
                read_xf(p)

    def test_too_few_columns_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td) / "c.xf", "1.0 0.0 0.0 1.0 0.0\n")  # 5 cols
            with self.assertRaises(ValueError):
                read_xf(p)

    def test_valid_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.xf"
            A = np.array([[[1.0, 0.01], [-0.02, 0.99]], [[1.0, 0.0], [0.0, 1.0]]])
            d = np.array([[1.5, -2.5], [0.0, 0.0]])
            write_xf(p, A, d)
            A2, d2 = read_xf(p)
            self.assertTrue(np.allclose(A2, A, atol=1e-6))
            self.assertTrue(np.allclose(d2, d, atol=1e-3))


class FitAffineGuardTests(unittest.TestCase):
    def test_too_few_points_rejected(self):
        with self.assertRaises(ValueError):
            fit_affine(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            fit_affine(np.zeros((5, 2)), np.zeros((4, 2)))


class NonAffineExportSafeguardTests(unittest.TestCase):
    """Replicates the exporter's decision: ok iff rms<=rms_tol and max<=max_tol."""

    RMS_TOL = 0.10
    MAX_TOL = 0.25

    def _grid(self):
        xs = np.linspace(-1000, 1000, 17)
        ys = np.linspace(-800, 800, 13)
        return np.array([(x, y) for y in ys for x in xs], float)

    def test_affine_field_is_accepted(self):
        pts = self._grid()
        A = np.array([[1.01, 0.02], [-0.015, 0.99]])
        d = np.array([3.0, -2.0])
        sampled = pts @ A.T + d  # perfectly affine
        _, _, res = fit_affine(pts, sampled)
        stats = residual_statistics(res)
        ok = stats["rms"] <= self.RMS_TOL and stats["max"] <= self.MAX_TOL
        self.assertTrue(ok)
        self.assertLess(stats["max"], 1e-6)

    def test_nonaffine_field_is_rejected(self):
        pts = self._grid()
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([0.0, 0.0])
        # Add a quadratic (non-affine) warp well above tolerance.
        warp = np.column_stack([
            1e-5 * pts[:, 0] ** 2,
            1e-5 * pts[:, 1] ** 2,
        ])
        sampled = pts @ A.T + d + warp
        _, _, res = fit_affine(pts, sampled)
        stats = residual_statistics(res)
        ok = stats["rms"] <= self.RMS_TOL and stats["max"] <= self.MAX_TOL
        self.assertFalse(ok, f"non-affine field should be rejected; stats={stats}")
        self.assertGreater(stats["max"], self.MAX_TOL)


if __name__ == "__main__":
    unittest.main()
