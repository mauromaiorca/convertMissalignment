"""IMOD tilt.com positioning: parsing, physical scaling, sign/axis oracle, hashing.

Runs without warpylib/torch. The projection oracle establishes the IMOD-side
conventions numerically; the Warp round-trip (LevelAngleX sign against the installed
warpylib Euler) is a separate, warpylib-gated test.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.imod_positioning import (  # noqa: E402
    ImodPositioning,
    imod_detector_projection,
    imod_offset_to_warp,
    imod_reconstruction_shift_to_warp,
    imod_xaxis_tilt_to_warp,
    parse_imod_positioning,
    parse_pair_floats,
    parse_scalar,
)

SAMPLE = "$tilt\nTHICKNESS 1200\nXAXISTILT 1.82\nOFFSET -11.5\nSHIFT 0.0 -8.1\n"


def _write(tmp: Path, text: str, name: str = "tilt.com") -> Path:
    p = tmp / name
    p.write_text(text)
    return p


class ParserTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_exact_sample_values_and_types(self):
        pos = parse_imod_positioning(_write(self.tmp, SAMPLE), unbinned_pixel_size_A=2.0)
        self.assertEqual(pos.thickness_unbinned_px, 1200)
        self.assertIsInstance(pos.thickness_unbinned_px, int)
        self.assertEqual(pos.x_axis_tilt_deg, 1.82)
        self.assertEqual(pos.tilt_angle_offset_deg, -11.5)
        self.assertEqual(pos.shift_x_unbinned_px, 0.0)
        self.assertEqual(pos.shift_z_unbinned_px, -8.1)
        self.assertEqual(set(pos.present_fields), {"THICKNESS", "XAXISTILT", "OFFSET", "SHIFT"})
        # physical shift uses the UNBINNED pixel size, not any output/aligned pixel
        self.assertAlmostEqual(pos.shift_z_A, -16.2, places=6)
        self.assertAlmostEqual(pos.shift_x_A, 0.0, places=6)

    def test_absent_values_resolve_to_zero_but_stay_distinguishable(self):
        pos = parse_imod_positioning(_write(self.tmp, "$tilt\nTHICKNESS 500\n"))
        self.assertEqual(pos.tilt_angle_offset_deg, 0.0)
        self.assertEqual(pos.x_axis_tilt_deg, 0.0)
        self.assertEqual(pos.shift_x_unbinned_px, 0.0)
        self.assertNotIn("OFFSET", pos.present_fields)      # absent
        self.assertIn("THICKNESS", pos.present_fields)

    def test_explicit_zero_is_present(self):
        pos = parse_imod_positioning(_write(self.tmp, "$tilt\nOFFSET 0.0\nSHIFT 0 0\n"))
        self.assertIn("OFFSET", pos.present_fields)         # present, value 0
        self.assertIn("SHIFT", pos.present_fields)

    def test_comments_mixed_case_and_equals_syntax(self):
        text = (
            "$tilt\n"
            "# OFFSET 999 should be ignored (comment)\n"
            "  offset = -11.5\n"
            "\tXaxisTilt\t1.82\n"
            "SHIFT = 0.0 -8.1   # trailing comment\n"
        )
        pos = parse_imod_positioning(_write(self.tmp, text), unbinned_pixel_size_A=2.0)
        self.assertEqual(pos.tilt_angle_offset_deg, -11.5)
        self.assertEqual(pos.x_axis_tilt_deg, 1.82)
        self.assertEqual((pos.shift_x_unbinned_px, pos.shift_z_unbinned_px), (0.0, -8.1))

    def test_duplicate_entries_last_active_wins(self):
        text = "$tilt\nOFFSET -5\nOFFSET -11.5\nSHIFT 1 1\nSHIFT 0.0 -8.1\n"
        self.assertEqual(parse_scalar(text, "OFFSET"), -11.5)
        self.assertEqual(parse_pair_floats(text, "SHIFT"), (0.0, -8.1))

    def test_shift_is_float_not_integer_parser(self):
        # a purely integer parser would drop the fractional part
        self.assertEqual(parse_pair_floats("SHIFT -0.5 3.25\n", "SHIFT"), (-0.5, 3.25))

    def test_tilt_com_is_authoritative_over_tilt_log(self):
        com = _write(self.tmp, "$tilt\nOFFSET -11.5\n", "tilt.com")
        log = _write(self.tmp, "OFFSET 42.0\n", "tilt.log")
        pos = parse_imod_positioning(com, tilt_log_path=log)
        self.assertEqual(pos.tilt_angle_offset_deg, -11.5)   # com wins, log ignored
        self.assertEqual(pos.source_kind, "tilt.com")

    def test_tilt_log_used_only_as_recorded_fallback(self):
        # com lacks XAXISTILT; log provides it and the fallback is recorded
        com = _write(self.tmp, "$tilt\nOFFSET -11.5\n", "tilt.com")
        log = _write(self.tmp, "XAXISTILT 1.82\n", "tilt.log")
        pos = parse_imod_positioning(com, tilt_log_path=log)
        self.assertEqual(pos.x_axis_tilt_deg, 1.82)

    def test_nonzero_shift_without_pixel_size_fails_clearly(self):
        with self.assertRaises(ValueError) as cm:
            parse_imod_positioning(_write(self.tmp, "$tilt\nSHIFT 0.0 -8.1\n"))
        self.assertIn("pixel size", str(cm.exception).lower())

    def test_override_precedence_and_recorded(self):
        pos = parse_imod_positioning(
            _write(self.tmp, SAMPLE), unbinned_pixel_size_A=2.0,
            overrides={"tilt_angle_offset_deg": -3.0})
        self.assertEqual(pos.tilt_angle_offset_deg, -3.0)    # override beats tilt.com
        self.assertEqual(pos.x_axis_tilt_deg, 1.82)          # unchanged
        self.assertIn("OFFSET", pos.overridden)


class ConversionFunctionTests(unittest.TestCase):
    def test_offset_applied_exactly_once(self):
        raw = [-60.0, 0.0, 60.0]
        out = imod_offset_to_warp(raw, -11.5)
        self.assertEqual(out["effective_tilt_angles_deg"], [-71.5, -11.5, 48.5])
        self.assertEqual(out["warp_representation"], "level_angle_y")
        self.assertEqual(out["raw_tilt_angles_deg"], raw)   # raw kept separately

    def test_shift_uses_unbinned_pixel_and_maps_to_object_xyz(self):
        # signed IMOD-MRC->Warp frame transform (sign -1): IMOD-MRC [0,-16.2,0] -> Warp [0,0,+16.2]
        out = imod_reconstruction_shift_to_warp(0.0, -8.1, 2.0, tilt_angle_sign=-1)
        self.assertEqual(out["imod_shift_vector_A"], [0.0, -16.2, 0.0])
        self.assertEqual(out["warp_object_shift_A"], [0.0, 0.0, 16.2])    # (X, Y, Z=thickness)
        self.assertEqual(out["orientation_determinant"], 1)

    def test_xaxis_sign_is_explicit(self):
        self.assertEqual(imod_xaxis_tilt_to_warp(1.82, sign=1)["warp_level_angle_x_deg"], 1.82)
        self.assertEqual(imod_xaxis_tilt_to_warp(1.82, sign=-1)["warp_level_angle_x_deg"], -1.82)
        with self.assertRaises(ValueError):
            imod_xaxis_tilt_to_warp(1.82, sign=2)


class ProjectionOracleTests(unittest.TestCase):
    """Numerically establish the IMOD conventions; these do not need warpylib."""

    def test_offset_shifts_effective_angle_once(self):
        # a point on the beam axis (0,0,z) projects to u = z*sin(theta_eff)
        z = 100.0
        for raw in (-60.0, 0.0, 60.0):
            u_off, _ = imod_detector_projection((0, 0, z), raw, offset_deg=-11.5)
            u_expected = z * math.sin(math.radians(raw - 11.5))
            self.assertAlmostEqual(u_off, u_expected, places=6)

    def test_z_shift_is_angle_dependent_not_a_constant_offset(self):
        # SHIFT z = -8.1 -> detector du = -8.1 * sin(theta); zero at 0, opposite at +/-45.
        tilts = [-45.0, 0.0, 45.0]
        deltas = []
        for t in tilts:
            u_shift, _ = imod_detector_projection((0, 0, 0), t, shift_xz_px=(0.0, -8.1))
            u_base, _ = imod_detector_projection((0, 0, 0), t)
            deltas.append(u_shift - u_base)
        # exact per-tilt value
        for t, d in zip(tilts, deltas):
            self.assertAlmostEqual(d, -8.1 * math.sin(math.radians(t)), places=6)
        # angle-dependent: not all equal (a constant image offset would make them equal)
        self.assertGreater(max(deltas) - min(deltas), 1.0)
        self.assertAlmostEqual(deltas[1], 0.0, places=6)                 # zero at zero tilt
        self.assertAlmostEqual(deltas[0], -deltas[2], places=6)          # antisymmetric

    def test_x_shift_projects_with_cosine(self):
        for t in (-30.0, 0.0, 30.0):
            u_shift, _ = imod_detector_projection((0, 0, 0), t, shift_xz_px=(5.0, 0.0))
            self.assertAlmostEqual(u_shift, 5.0 * math.cos(math.radians(t)), places=6)

    def test_xaxis_tilt_couples_y_and_z_with_documented_sign(self):
        # a point on +z, at zero tilt, gains a +v (along-axis) component for +XAXISTILT
        u, v = imod_detector_projection((0, 0, 10.0), 0.0, x_axis_tilt_deg=1.82)
        self.assertAlmostEqual(v, -10.0 * math.sin(math.radians(1.82)), places=6)
        # sign flips with the angle sign
        _, v_neg = imod_detector_projection((0, 0, 10.0), 0.0, x_axis_tilt_deg=-1.82)
        self.assertAlmostEqual(v_neg, -v, places=6)

    def test_backward_compatible_when_all_absent(self):
        # all zero -> plain single-axis projection u = x cos + z sin, v = y
        for t in (-40.0, 0.0, 55.0):
            u, v = imod_detector_projection((3.0, 7.0, -2.0), t)
            self.assertAlmostEqual(u, 3.0 * math.cos(math.radians(t)) + (-2.0) * math.sin(math.radians(t)), places=6)
            self.assertAlmostEqual(v, 7.0, places=6)


class HashTests(unittest.TestCase):
    def _pos(self, **kw):
        base = dict(tilt_angle_offset_deg=-11.5, x_axis_tilt_deg=1.82,
                    shift_x_unbinned_px=0.0, shift_z_unbinned_px=-8.1,
                    unbinned_pixel_size_A=2.0, thickness_unbinned_px=1200)
        base.update(kw)
        return ImodPositioning(**base)

    def test_same_values_same_hash(self):
        self.assertEqual(self._pos().positioning_hash(), self._pos().positioning_hash())

    def test_each_parameter_change_invalidates_hash(self):
        base = self._pos().positioning_hash()
        self.assertNotEqual(base, self._pos(tilt_angle_offset_deg=-11.6).positioning_hash())
        self.assertNotEqual(base, self._pos(x_axis_tilt_deg=1.83).positioning_hash())
        self.assertNotEqual(base, self._pos(shift_x_unbinned_px=0.1).positioning_hash())
        self.assertNotEqual(base, self._pos(shift_z_unbinned_px=-8.0).positioning_hash())
        self.assertNotEqual(base, self._pos(unbinned_pixel_size_A=2.1).positioning_hash())

    def test_provenance_only_fields_do_not_change_hash(self):
        a = self._pos(source_path="/a/tilt.com").positioning_hash()
        b = self._pos(source_path="/b/tilt.com").positioning_hash()
        self.assertEqual(a, b)

    def test_manifest_carries_raw_resolved_units_and_hash(self):
        m = self._pos().to_manifest()
        self.assertEqual(m["tilt_angle_offset_deg"], -11.5)
        self.assertEqual(m["shift_unbinned_px"], [0.0, -8.1])
        self.assertEqual(m["shift_A"], [0.0, -16.2])
        self.assertEqual(m["unbinned_pixel_size_A"], 2.0)
        self.assertIn("positioning_hash", m)
        self.assertEqual(m["contract_version"], 2)


if __name__ == "__main__":
    unittest.main()
