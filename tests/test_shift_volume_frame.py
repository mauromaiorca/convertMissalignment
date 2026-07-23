"""IMOD reconstruction SHIFT into the Warp volume frame (signed orientation matrix).

The tilt-angle sign was changed to -1 without applying the corresponding signed 3-D frame
transform to SHIFT. These tests pin the fix: build the SHIFT in native IMOD-MRC order
[X, Z(thickness), 0] and transform ONCE with IMOD_MRC_TO_WARP (det +1 for sign -1); the shape
permutation (0,2,1) is untouched; SHIFT is represented exactly once; the .xf offsets, view
order and angle sign are unchanged. Pure numpy — no warpylib/WarpTools/IMOD.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.volume_frames import (  # noqa: E402
    BASE_AXIS_PERMUTATION, IMOD_MRC_TO_WARP, imod_mrc_to_warp_orientation,
    volume_frame_manifest, warp_to_imod_mrc_orientation,
)
from geometry.imod_positioning import (  # noqa: E402
    imod_reconstruction_shift_to_warp, warp_shift_to_imod_reconstruction,
)


class OrientationMatrixTests(unittest.TestCase):
    def test_shape_permutation_unchanged(self):
        self.assertEqual(tuple(BASE_AXIS_PERMUTATION), (0, 2, 1))

    def test_minus_one_matrix_and_determinant(self):
        m = imod_mrc_to_warp_orientation(-1)
        self.assertTrue(np.array_equal(m, IMOD_MRC_TO_WARP))
        self.assertTrue(np.array_equal(m, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)))
        self.assertEqual(int(round(np.linalg.det(m))), 1)                 # +1, hand-preserving

    def test_axis_mapping(self):
        m = imod_mrc_to_warp_orientation(-1)
        # Warp X = IMOD-MRC X ; Warp Y = IMOD-MRC Z ; Warp Z = -IMOD-MRC Y
        self.assertTrue(np.allclose(m @ np.array([1.0, 0, 0]), [1, 0, 0]))
        self.assertTrue(np.allclose(m @ np.array([0, 1.0, 0]), [0, 0, -1]))
        self.assertTrue(np.allclose(m @ np.array([0, 0, 1.0]), [0, 1, 0]))

    def test_plus_one_uses_a_different_signed_policy(self):
        m_minus = imod_mrc_to_warp_orientation(-1)
        m_plus = imod_mrc_to_warp_orientation(1)
        self.assertFalse(np.array_equal(m_minus, m_plus))                 # not reused blindly
        self.assertEqual(int(round(np.linalg.det(m_plus))), -1)           # +1 -> handedness flipped

    def test_inverse_is_transpose(self):
        for s in (-1, 1):
            self.assertTrue(np.array_equal(
                warp_to_imod_mrc_orientation(s), imod_mrc_to_warp_orientation(s).T))

    def test_invalid_sign_rejected(self):
        for bad in (0, 2, -2, 0.5):
            with self.assertRaises(ValueError):
                imod_mrc_to_warp_orientation(bad)


class ShiftTransformTests(unittest.TestCase):
    def test_imod_mrc_vector_is_X_Z_0(self):
        m = imod_reconstruction_shift_to_warp(3.0, -8.1, 2.2, tilt_angle_sign=-1)
        # native IMOD-MRC order: [sx_A, sz_A, 0], NOT [sx, 0, sz]
        self.assertTrue(np.allclose(m["imod_shift_vector_A"], [3.0 * 2.2, -8.1 * 2.2, 0.0]))

    def test_project_shift_z_minus_8_1_becomes_warp_z_plus_17_82(self):
        m = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.2, tilt_angle_sign=-1)
        self.assertTrue(np.allclose(m["imod_shift_vector_A"], [0.0, -17.82, 0.0]))
        self.assertTrue(np.allclose(m["warp_object_shift_A"], [0.0, 0.0, 17.82]))
        self.assertEqual(m["orientation_determinant"], 1)

    def test_warp_z_displacement_in_pixels_at_target(self):
        m = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.2, tilt_angle_sign=-1)
        warp_z_A = m["warp_object_shift_A"][2]
        self.assertAlmostEqual(warp_z_A / 17.6, 1.0125, places=4)          # target pixel used ONLY here

    def test_target_pixel_size_does_not_affect_physical_shift(self):
        # the physical shift is converted with the UNBINNED IMOD pixel (2.2), never 17.6
        a = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.2, tilt_angle_sign=-1)
        # passing 17.6 as the unbinned pixel would give a different (wrong) vector -> proof it
        # must be the unbinned 2.2, not the target
        b = imod_reconstruction_shift_to_warp(0.0, -8.1, 17.6, tilt_angle_sign=-1)
        self.assertFalse(np.allclose(a["warp_object_shift_A"], b["warp_object_shift_A"]))

    def test_round_trip_restores_shift_x_z(self):
        for sx, sz in [(0.0, -8.1), (3.5, -8.1), (-2.2, 4.4), (1.0, 0.0)]:
            fwd = imod_reconstruction_shift_to_warp(sx, sz, 2.2, tilt_angle_sign=-1)
            inv = warp_shift_to_imod_reconstruction(fwd["warp_object_shift_A"], 2.2, tilt_angle_sign=-1)
            self.assertAlmostEqual(inv["shift_x_unbinned_px"], sx, places=6)
            self.assertAlmostEqual(inv["shift_z_unbinned_px"], sz, places=6)   # from MRC comp 1

    def test_shift_z_comes_from_imod_mrc_component_1(self):
        inv = warp_shift_to_imod_reconstruction([0.0, 0.0, 17.82], 2.2, tilt_angle_sign=-1)
        # component 1 of IMOD-MRC (thickness) carries SHIFT Z, not component 2
        self.assertAlmostEqual(inv["imod_shift_vector_A"][1], -17.82, places=3)
        self.assertAlmostEqual(inv["shift_z_unbinned_px"], -8.1, places=4)

    def test_plus_one_sign_gives_a_different_shift(self):
        minus = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.2, tilt_angle_sign=-1)
        plus = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.2, tilt_angle_sign=1)
        self.assertFalse(np.allclose(minus["warp_object_shift_A"], plus["warp_object_shift_A"]))


class ManifestTests(unittest.TestCase):
    def test_volume_frame_manifest_records_orientation(self):
        vf = volume_frame_manifest([720, 150, 720], quarter_turn_k=0, tilt_angle_sign=-1)
        self.assertEqual(vf["orientation_matrix_imod_mrc_to_warp"],
                         [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.assertEqual(vf["orientation_determinant"], 1)
        self.assertEqual(vf["handedness_effect"], "preserved")
        self.assertEqual(vf["shape_permutation"], [0, 2, 1])
        self.assertEqual(vf["base_axis_permutation_imod_mrc_to_warp"], [0, 2, 1])  # unchanged

    def test_plus_one_manifest_handedness_flipped(self):
        vf = volume_frame_manifest([720, 150, 720], quarter_turn_k=0, tilt_angle_sign=1)
        self.assertEqual(vf["orientation_determinant"], -1)
        self.assertEqual(vf["handedness_effect"], "flipped")

    def test_positioning_contract_bumped_to_v2(self):
        from geometry.imod_positioning import POSITIONING_CONTRACT_VERSION
        self.assertEqual(POSITIONING_CONTRACT_VERSION, 2)


class CacheInvalidationTests(unittest.TestCase):
    def test_contract_v2_changes_positioning_hash(self):
        # A positioning with a SHIFT hashes differently under v2 (all v1 markers become stale).
        from geometry.imod_positioning import ImodPositioning
        p = ImodPositioning(shift_z_unbinned_px=-8.1, unbinned_pixel_size_A=2.2,
                            imod_to_warp_tilt_angle_sign=-1)
        payload = p.positioning_hash()
        self.assertTrue(isinstance(payload, str) and len(payload) == 64)

    def test_reconstruction_contract_includes_positioning_hash(self):
        from pipeline.reconstruction_tiling import (
            ReconstructionTiling, reconstruction_contract_hash)
        tiling = ReconstructionTiling(subvolume_size=64, subvolume_padding=6)
        base = dict(tiling=tiling, output_angpix=17.6, normalize=False, warptools_version="2.0.39")
        self.assertNotEqual(
            reconstruction_contract_hash(**base, positioning_hash="a"),
            reconstruction_contract_hash(**base, positioning_hash="b"))


class ConverterEndToEndTests(unittest.TestCase):
    """SHIFT applied exactly once, .xf offsets unchanged, via the fake-warp converter."""

    def _converter(self):
        class FakeCubicGrid:
            def __init__(self, *a, **k):
                pass

        class FakeTiltSeries:
            def __init__(self, *a, **k):
                self.shift_calls = []

            def save_meta(self, path):
                pass

            def apply_tomogram_shift_3d(self, vec):
                self.shift_calls.append([float(x) for x in vec.tolist()])
        wl = types.ModuleType("warpylib")
        wl.CubicGrid = FakeCubicGrid
        wl.TiltSeries = FakeTiltSeries
        ops = types.ModuleType("warpylib.ops")
        rescale = types.ModuleType("warpylib.ops.rescale")
        rescale.rescale = lambda images, size: images
        sys.modules.update({"warpylib": wl, "warpylib.ops": ops, "warpylib.ops.rescale": rescale})
        spec = importlib.util.spec_from_file_location(
            "e2w_shift_frame", ROOT / "scripts" / "etomo_to_warp.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_shift_applied_once_and_xf_offsets_unchanged(self):
        import mrcfile
        from geometry.imod_positioning import ImodPositioning
        from imod_affine import inverse_physical_map
        mod = self._converter()
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        ts_dir = tmp / "TS_demo"; ts_dir.mkdir()
        n = 3
        (ts_dir / "TS_demo.rawtlt").write_text("-30\n0\n30\n")
        A = np.array([[0.10453, -0.99452], [0.99452, 0.10453]])
        shifts = [(23.5, -11.2), (5.0, 3.0), (-8.1, 12.4)]
        rows = "".join(f"{A[0,0]} {A[0,1]} {A[1,0]} {A[1,1]} {sx} {sy}\n" for sx, sy in shifts)
        (ts_dir / "TS_demo.xf").write_text(rows)
        (ts_dir / "TS_demo.source.xf").write_text(rows)
        with mrcfile.new(str(ts_dir / "TS_demo.st"), overwrite=True) as mrc:
            mrc.set_data(np.zeros((n, 8, 8), dtype=np.float32))
            mrc.voxel_size = 2.2
        out = tmp / "out"; out.mkdir()
        pos = ImodPositioning(shift_x_unbinned_px=0.0, shift_z_unbinned_px=-8.1,
                              unbinned_pixel_size_A=2.2, imod_to_warp_tilt_angle_sign=-1,
                              present_fields=("SHIFT",))
        ts, _ = mod.process_tilt_series(
            ts_dir, out, tilt_axis_angle=84.1, volume_shape=(8, 4, 8), output_pixel_size=None,
            alignment_mode="translation", axis_frame="raw", grid_shape_xy=(5, 5),
            positioning=pos, imod_to_warp_tilt_angle_sign=-1)
        # SHIFT: exactly one global apply_tomogram_shift_3d call, Warp [0,0,+17.82]
        self.assertEqual(len(ts.shift_calls), 1)
        self.assertTrue(np.allclose(ts.shift_calls[0], [0.0, 0.0, 17.82], atol=1e-2))
        # .xf per-view offsets UNCHANGED (still the -inv(A)@d inverse-map, SHIFT not added)
        for i, (sx, sy) in enumerate(shifts):
            _, exp = inverse_physical_map(A, np.array([sx, sy]), 2.2, 2.2)
            self.assertAlmostEqual(ts.tilt_axis_offset_x.tolist()[i], exp[0], places=3)
            self.assertAlmostEqual(ts.tilt_axis_offset_y.tolist()[i], exp[1], places=3)
        # manifest: orientation + shift vectors recorded; view order + angle sign unchanged
        man = json.loads((out / "TS_demo.conversion.json").read_text())
        vf = man["volume_frame"]
        self.assertEqual(vf["orientation_determinant"], 1)
        self.assertEqual(vf["shape_permutation"], [0, 2, 1])
        self.assertTrue(np.allclose(vf["warp_shift_vector_A"], [0.0, 0.0, 17.82], atol=1e-2))
        self.assertTrue(np.allclose(vf["imod_shift_vector_A"], [0.0, -17.82, 0.0], atol=1e-2))
        self.assertEqual(man["tilt_view_order"]["mapping"], "identity")
        self.assertEqual(man["tilt_angle_convention"]["imod_to_warp_sign"], -1)


if __name__ == "__main__":
    unittest.main()
