"""Converter application of IMOD positioning, against a fake warpylib TiltSeries.

Validates the wiring required by the mandatory Warp representation policy: OFFSET ->
level_angle_y (applied once, raw angles untouched), XAXISTILT -> level_angle_x with the
selected sign, SHIFT -> apply_tomogram_shift_3d in Angstrom. Runs without real warpylib.
The geometric correctness of the sign/Euler convention is confirmed separately by the
cluster-side validate_warp_positioning.py against the installed warpylib.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.imod_positioning import ImodPositioning  # noqa: E402


class FakeTiltSeriesWithShift:
    def __init__(self):
        self.angles = torch.tensor([-60.0, 0.0, 60.0], dtype=torch.float32)
        self.shift_calls = []

    def apply_tomogram_shift_3d(self, vec):
        self.shift_calls.append([float(x) for x in vec.tolist()])


class FakeTiltSeriesNoShift:
    def __init__(self):
        self.angles = torch.tensor([-60.0, 0.0, 60.0], dtype=torch.float32)


def _load_converter():
    warpylib = types.ModuleType("warpylib")
    warpylib.CubicGrid = object
    warpylib.TiltSeries = FakeTiltSeriesWithShift
    ops = types.ModuleType("warpylib.ops")
    rescale_mod = types.ModuleType("warpylib.ops.rescale")
    rescale_mod.rescale = lambda images, size: images
    sys.modules["warpylib"] = warpylib
    sys.modules["warpylib.ops"] = ops
    sys.modules["warpylib.ops.rescale"] = rescale_mod
    spec = importlib.util.spec_from_file_location(
        "etomo_to_warp_pos_test", ROOT / "scripts" / "etomo_to_warp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConverterPositioningTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_converter()
        self.pos = ImodPositioning(
            tilt_angle_offset_deg=-11.5, x_axis_tilt_deg=1.82,
            shift_x_unbinned_px=0.0, shift_z_unbinned_px=-8.1,
            unbinned_pixel_size_A=2.0, thickness_unbinned_px=1200,
            present_fields=("THICKNESS", "OFFSET", "XAXISTILT", "SHIFT"))

    def test_level_angles_and_shift_applied_and_recorded(self):
        ts = FakeTiltSeriesWithShift()
        raw_before = ts.angles.clone()
        applied = self.mod.apply_imod_positioning(ts, self.pos, level_angle_x_sign=-1)
        # OFFSET is baked into ts.angles by process_tilt_series -> here LevelAngleY = 0 (once).
        # apply_imod_positioning does not touch ts.angles.
        self.assertEqual(ts.level_angle_y, 0.0)
        self.assertEqual(applied["offset_representation"], "baked_into_angles")
        self.assertEqual(applied["imod_offset_deg"], -11.5)
        self.assertEqual(applied["imod_to_warp_tilt_angle_sign"], -1)
        self.assertTrue(torch.equal(ts.angles, raw_before))
        # XAXISTILT -> level_angle_x = sign * 1.82
        self.assertAlmostEqual(ts.level_angle_x, -1.82, places=6)
        self.assertEqual(applied["level_angle_x_sign"], -1)
        self.assertFalse(applied["level_angle_x_sign_validated"])   # only the cluster script validates
        # SHIFT -> apply_tomogram_shift_3d via the signed IMOD-MRC->Warp frame transform.
        # pixel 2.0, shift_z=-8.1, sign -1: IMOD-MRC [0,-16.2,0] -> Warp [0,0,+16.2].
        self.assertEqual(len(ts.shift_calls), 1)
        for got, exp in zip(ts.shift_calls[0], [0.0, 0.0, 16.2]):
            self.assertAlmostEqual(got, exp, places=3)
        self.assertEqual(applied["shift_representation"], "apply_tomogram_shift_3d")
        self.assertAlmostEqual(applied["warp_object_shift_A"][2], 16.2, places=3)
        self.assertEqual(applied["imod_shift_vector_A"][1], -16.2)   # native MRC: SHIFT Z in comp 1
        self.assertEqual(applied["orientation_determinant"], 1)
        self.assertIn("positioning_hash", applied)

    def test_positive_sign_flips_level_angle_x(self):
        ts = FakeTiltSeriesWithShift()
        self.mod.apply_imod_positioning(ts, self.pos, level_angle_x_sign=1)
        self.assertAlmostEqual(ts.level_angle_x, 1.82, places=6)

    def test_shift_requires_apply_tomogram_shift_3d(self):
        ts = FakeTiltSeriesNoShift()
        with self.assertRaises(ValueError) as cm:
            self.mod.apply_imod_positioning(ts, self.pos)
        self.assertIn("apply_tomogram_shift_3d", str(cm.exception))

    def test_zero_positioning_sets_zero_level_angles_and_no_shift(self):
        ts = FakeTiltSeriesWithShift()
        zero = ImodPositioning()   # all absent/zero
        applied = self.mod.apply_imod_positioning(ts, zero)
        self.assertEqual(ts.level_angle_y, 0.0)
        self.assertEqual(ts.level_angle_x, 0.0)
        self.assertEqual(ts.shift_calls, [])                       # no shift applied
        self.assertEqual(applied["shift_representation"], "none")


if __name__ == "__main__":
    unittest.main()
