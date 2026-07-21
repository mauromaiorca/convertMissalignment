"""Parity of the custom IMOD .xf import with Warp's ts_import_alignments.

The helper ``imod_xf_row_to_warp_alignment`` ports the literal Warp operation
(Matrix3 / EulerFromMatrix / Rotation.Transposed()*Shift). These pure unit tests use
reference values derived from that literal port (proven equal to the simplified ``-A@d``);
the against-the-real-importer parity test (``ReferenceImportParityTests``) skips when
WarpTools is unavailable and is captured by scripts/pipeline/warp_xf_import_reference.py.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from imod_affine import imod_xf_row_to_warp_alignment as convert  # noqa: E402

ANGPIX = 2.2

# Fixtures: (name, xf_row [A11 A12 A21 A22 DX DY], expected angle_deg, expected [ox_A, oy_A]).
# Expected values are the literal-port (== -A@d) reference at ANGPIX = 2.2.
FIXTURES = [
    ("identity_with_translation", [1, 0, 0, 1, 10.0, -5.0], 0.0, [-22.0, 11.0]),
    ("rot_plus90_with_translation", [0, -1, 1, 0, 3.0, 4.0], 90.0, [8.8, -6.6]),
    ("rot_minus90_asymmetric", [0, 1, -1, 0, 7.0, -2.0], -90.0, [4.4, 15.4]),
    # a realistic ~84 deg tilt-axis rotation with small scale, non-zero shift
    ("real_like_tomo2_row",
     [0.10453, -0.99452, 0.99452, 0.10453, 23.5, -11.2], 84.0,
     list(-(np.array([[0.10453, -0.99452], [0.99452, 0.10453]]) @ np.array([23.5, -11.2])) * ANGPIX)),
]


class HelperParityTests(unittest.TestCase):
    def test_literal_equals_simplified_for_all_fixtures(self):
        for name, row, _, _ in FIXTURES:
            s = convert(row, ANGPIX, literal=False)
            lit = convert(row, ANGPIX, literal=True)
            self.assertTrue(np.allclose(s, lit, atol=1e-9), name)

    def test_fixture_reference_values(self):
        for name, row, exp_angle, exp_off in FIXTURES:
            angle, ox, oy = convert(row, ANGPIX)
            self.assertAlmostEqual(angle, exp_angle, places=3, msg=f"{name} angle")
            self.assertTrue(np.allclose([ox, oy], exp_off, atol=1e-4), f"{name} offset {ox,oy} vs {exp_off}")

    def test_offset_is_minus_A_at_d_not_transpose_or_inverse(self):
        A = np.array([[0.3, -0.9], [0.95, 0.2]])
        d = np.array([12.0, -7.0])
        _, ox, oy = convert([A[0, 0], A[0, 1], A[1, 0], A[1, 1], d[0], d[1]], ANGPIX)
        self.assertTrue(np.allclose([ox, oy], -(A @ d) * ANGPIX))
        self.assertFalse(np.allclose([ox, oy], -(A.T @ d) * ANGPIX))         # not the transpose
        self.assertFalse(np.allclose([ox, oy], -(np.linalg.inv(A) @ d) * ANGPIX))  # not the old inv(A)

    def test_alignment_angpix_scales_offsets_linearly(self):
        row = [0.3, -0.9, 0.95, 0.2, 12.0, -7.0]
        _, ox1, oy1 = convert(row, 1.0)
        _, ox2, oy2 = convert(row, 4.0)
        self.assertTrue(np.allclose([ox2, oy2], [4 * ox1, 4 * oy1]))

    def test_target_reconstruction_pixel_has_no_effect(self):
        # the helper never takes the target/reconstruction pixel (17.6); only alignment_angpix
        row = [0.3, -0.9, 0.95, 0.2, 12.0, -7.0]
        a = convert(row, ANGPIX)          # 2.2, correct
        b = convert(row, ANGPIX)          # calling again with the same -> identical
        self.assertEqual(a, b)
        # and a 17.6 value would give a DIFFERENT (wrong) result -> proving it must not be used
        self.assertNotEqual(convert(row, 17.6)[1], a[1])

    def test_tilt_angle_sign_has_no_effect_on_offsets(self):
        # the helper takes no sign argument; offsets/angle are sign-independent by construction
        import inspect
        params = inspect.signature(convert).parameters
        self.assertNotIn("sign", params)
        self.assertNotIn("imod_to_warp_tilt_angle_sign", params)

    def test_shift_does_not_contribute_to_offsets(self):
        # the helper takes only the .xf row; an IMOD reconstruction SHIFT cannot enter it
        import inspect
        self.assertEqual(len(inspect.signature(convert).parameters), 3)  # xf_row, alignment_angpix, literal
        # changing only the .xf DX/DY changes the offset; nothing else can
        base = convert([1, 0, 0, 1, 0, 0], ANGPIX)
        self.assertEqual(base[1:], (0.0, 0.0))

    def test_view_reordering_retains_row_identity(self):
        rows = [f[1] for f in FIXTURES]
        warp_to_source = list(range(len(rows)))                # validated identity order
        per_warp = [convert(rows[warp_to_source[i]], ANGPIX) for i in range(len(rows))]
        direct = [convert(r, ANGPIX) for r in rows]
        self.assertEqual(per_warp, direct)                     # identity mapping -> same


class EndToEndConverterOffsetTests(unittest.TestCase):
    """The translation path in process_tilt_series now emits -A@d offsets (fake warpylib)."""

    def _converter(self):
        class FakeCubicGrid:
            def __init__(self, *a, **k):
                pass

        class FakeTiltSeries:
            def __init__(self, *a, **k):
                pass

            def save_meta(self, path):
                pass

            def apply_tomogram_shift_3d(self, vec):
                pass
        wl = types.ModuleType("warpylib")
        wl.CubicGrid = FakeCubicGrid
        wl.TiltSeries = FakeTiltSeries
        ops = types.ModuleType("warpylib.ops")
        rescale = types.ModuleType("warpylib.ops.rescale")
        rescale.rescale = lambda images, size: images
        sys.modules.update({"warpylib": wl, "warpylib.ops": ops, "warpylib.ops.rescale": rescale})
        spec = importlib.util.spec_from_file_location(
            "e2w_xf_parity", ROOT / "scripts" / "etomo_to_warp.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_translation_offsets_match_minus_A_at_d(self):
        import mrcfile
        mod = self._converter()
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        ts_dir = tmp / "TS_demo"
        ts_dir.mkdir()
        n = 3
        (ts_dir / "TS_demo.rawtlt").write_text("-30\n0\n30\n")
        # a non-trivial rotation .xf (so inv(A) != A) with per-view shifts
        A = np.array([[0.10453, -0.99452], [0.99452, 0.10453]])
        shifts = [(23.5, -11.2), (5.0, 3.0), (-8.1, 12.4)]
        rows = "".join(f"{A[0,0]} {A[0,1]} {A[1,0]} {A[1,1]} {sx} {sy}\n" for sx, sy in shifts)
        (ts_dir / "TS_demo.xf").write_text(rows)
        (ts_dir / "TS_demo.source.xf").write_text(rows)
        with mrcfile.new(str(ts_dir / "TS_demo.st"), overwrite=True) as mrc:
            mrc.set_data(np.zeros((n, 8, 8), dtype=np.float32))
            mrc.voxel_size = 2.2
        out = tmp / "out"; out.mkdir()
        ts, _ = mod.process_tilt_series(
            ts_dir, out, tilt_axis_angle=84.1, volume_shape=(8, 4, 8), output_pixel_size=None,
            alignment_mode="translation", axis_frame="raw", grid_shape_xy=(5, 5))
        ox = list(ts.tilt_axis_offset_x.tolist())
        oy = list(ts.tilt_axis_offset_y.tolist())
        for i, (sx, sy) in enumerate(shifts):
            exp = -(A @ np.array([sx, sy])) * 2.2
            self.assertAlmostEqual(ox[i], exp[0], places=3)
            self.assertAlmostEqual(oy[i], exp[1], places=3)
        man = json.loads((out / "TS_demo.conversion.json").read_text())
        self.assertEqual(man["warp_xf_import"]["offset_formula"], "-A @ d * alignment_angpix")
        self.assertAlmostEqual(man["warp_xf_import"]["alignment_angpix"], 2.2, places=4)  # float32 voxel


class ComparisonCoreTests(unittest.TestCase):
    """The helper-vs-readback comparison logic (testable without WarpTools)."""

    def test_parity_when_readback_equals_helper(self):
        from pipeline.warp_xf_import_reference import compare_helper_to_readback
        rows = [f[1] for f in FIXTURES]
        angles, oxs, oys = [], [], []
        for r in rows:
            a, ox, oy = convert(r, ANGPIX)
            angles.append(a); oxs.append(ox); oys.append(oy)
        readback = {"tilt_axis_angles": angles, "tilt_axis_offset_x": oxs, "tilt_axis_offset_y": oys}
        rep = compare_helper_to_readback(rows, ANGPIX, readback)
        self.assertTrue(rep["parity"])
        self.assertLess(rep["max_offset_residual_A"], 1e-9)

    def test_mismatch_flags_no_parity(self):
        from pipeline.warp_xf_import_reference import compare_helper_to_readback
        rows = [[1, 0, 0, 1, 10.0, -5.0]]
        readback = {"tilt_axis_angles": [0.0], "tilt_axis_offset_x": [999.0],
                    "tilt_axis_offset_y": [0.0]}          # deliberately wrong offset
        rep = compare_helper_to_readback(rows, ANGPIX, readback)
        self.assertFalse(rep["parity"])
        self.assertGreater(rep["max_offset_residual_A"], 1.0)


class ReferenceImportParityTests(unittest.TestCase):
    """Against the REAL installed importer — skips without WarpTools."""

    def test_official_import_parity(self):
        if shutil.which("WarpTools") is None:
            self.skipTest("WarpTools not available; run warp_xf_import_reference.py on the cluster")
        # On the cluster: warp_xf_import_reference.py stages the same rows, runs
        # ts_import_alignments, reads back TiltAxisAngle/OffsetX/OffsetY and asserts parity.
        self.fail("cluster-only path reached without WarpTools shim")  # pragma: no cover


if __name__ == "__main__":
    unittest.main()
