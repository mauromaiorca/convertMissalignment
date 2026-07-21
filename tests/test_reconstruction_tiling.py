"""Reconstruction tiling / locale / contract / pixel-size / seam tests.

All of these run without WarpTools or warpylib (they only inspect config, command
strings, hashes and synthetic numpy volumes).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.reconstruction_tiling import (  # noqa: E402
    DEFAULT_SUBVOLUME_PADDING,
    DEFAULT_SUBVOLUME_SIZE,
    ReconstructionConfigError,
    pixel_size_consistency_report,
    reconstruction_contract_hash,
    reconstruction_identity,
    resolve_tiling,
    resource_preflight,
    warptools_env,
    xml_comma_decimal_fields,
)
from pipeline.seam_diagnostic import seam_metric  # noqa: E402


class TilingResolutionTests(unittest.TestCase):
    def test_defaults(self):
        t = resolve_tiling(None)
        self.assertEqual(t.subvolume_size, 64)
        self.assertEqual(t.subvolume_padding, 6)
        self.assertEqual(DEFAULT_SUBVOLUME_SIZE, 64)
        self.assertEqual(DEFAULT_SUBVOLUME_PADDING, 6)

    def test_reject_padding_below_minimum(self):
        for bad in (5, 4, 3, 1):
            with self.assertRaises(ReconstructionConfigError) as cm:
                resolve_tiling({"subvolume_padding": bad})
            self.assertIn("minimum", str(cm.exception).lower())

    def test_reject_non_integer_and_boolean(self):
        for bad in (0, -2, 6.5, "6", None, True, False):
            with self.assertRaises(ReconstructionConfigError):
                resolve_tiling({"subvolume_padding": bad})
        for bad in (True, "64", 0, -1, 64.5):
            with self.assertRaises(ReconstructionConfigError):
                resolve_tiling({"subvolume_size": bad})

    def test_boolean_true_is_not_accepted_as_one(self):
        # bool is an int subclass; must not sneak through positive-int validation
        with self.assertRaises(ReconstructionConfigError):
            resolve_tiling({"subvolume_padding": True, "subvolume_size": 64})

    def test_whole_float_is_accepted(self):
        t = resolve_tiling({"subvolume_size": 64.0, "subvolume_padding": 6.0})
        self.assertEqual((t.subvolume_size, t.subvolume_padding), (64, 6))

    def test_args_contain_each_option_exactly_once(self):
        args = resolve_tiling({"subvolume_padding": 6}).to_args()
        self.assertEqual(args.count("--subvolume_size"), 1)
        self.assertEqual(args.count("--subvolume_padding"), 1)
        self.assertEqual(args, ["--subvolume_size", "64", "--subvolume_padding", "6"])

    def test_padded_side_and_isotropy(self):
        t = resolve_tiling({"subvolume_padding": 6})
        self.assertEqual(t.padded_side, 768)                 # int(64*6)*2, not hard-coded
        r = t.resolved_interpretation()
        self.assertEqual(r["padded_reconstruction_side_px"], 768)
        self.assertEqual(r["padding_axes"], "XYZ_isotropic")
        self.assertEqual(r["actual_output_overlap_px"], 0)
        self.assertEqual(r["assembly_method"], "non_overlapping_central_crop_direct_copy")

    def test_identity_string(self):
        self.assertEqual(reconstruction_identity(resolve_tiling({"subvolume_padding": 6})),
                         "reconstruction_s64_p6")


class ContractHashTests(unittest.TestCase):
    def _hash(self, **kw):
        base = dict(tiling=resolve_tiling({"subvolume_padding": 6}), output_angpix=17.6,
                    normalize=False, warptools_version="2.0.39", numeric_locale="C")
        base.update(kw)
        return reconstruction_contract_hash(**base)

    def test_stable(self):
        self.assertEqual(self._hash(), self._hash())

    def test_size_and_padding_changes_invalidate(self):
        base = self._hash()
        self.assertNotEqual(base, self._hash(tiling=resolve_tiling({"subvolume_size": 96, "subvolume_padding": 6})))
        self.assertNotEqual(base, self._hash(tiling=resolve_tiling({"subvolume_padding": 8})))

    def test_locale_version_angpix_normalize_invalidate(self):
        base = self._hash()
        self.assertNotEqual(base, self._hash(output_angpix=10.0))
        self.assertNotEqual(base, self._hash(normalize=True))
        self.assertNotEqual(base, self._hash(warptools_version="2.0.40"))
        self.assertNotEqual(base, self._hash(numeric_locale="en_US.UTF-8"))


class LocaleTests(unittest.TestCase):
    def test_env_merges_c_locale_without_deleting(self):
        env = warptools_env({"PATH": "/x", "FOO": "bar"})
        self.assertEqual(env["LC_ALL"], "C")
        self.assertEqual(env["LANG"], "C")
        self.assertEqual(env["PATH"], "/x")
        self.assertEqual(env["FOO"], "bar")

    def test_comma_decimal_detection(self):
        bad = '<Param Name="LevelAngleY">-11,5</Param><PixelSize>2,0</PixelSize>'
        offenders = xml_comma_decimal_fields(bad)
        self.assertIn("LevelAngleY", offenders)
        self.assertIn("PixelSize", offenders)

    def test_clean_xml_passes(self):
        good = '<Param Name="LevelAngleY">-11.5</Param><PixelSize>2.0</PixelSize>'
        self.assertEqual(xml_comma_decimal_fields(good), [])


class ResourceAndPixelTests(unittest.TestCase):
    def test_ratios(self):
        rep = resource_preflight(resolve_tiling({"subvolume_padding": 6}),
                                 n_tilts=41, device_list="0")
        self.assertAlmostEqual(rep["ratio_vs_warp_default_padding_3"], 8.0, places=4)
        self.assertAlmostEqual(rep["ratio_vs_padding_4"], 3.375, places=4)
        self.assertEqual(rep["padded_side_px"], 768)
        self.assertIn("not an exact vram prediction", rep["note"].lower())

    def test_output_pixel_mismatch_is_distinct_error(self):
        rep = pixel_size_consistency_report(
            unbinned_pixel_size_A=2.0, image_binned=8, aligned_pixel_size_A=16.0,
            warp_input_angpix_A=17.6, requested_output_angpix_A=17.6,
            output_voxel_size_A=20.0)
        self.assertFalse(rep["output_voxel_matches_request"])
        self.assertTrue(any("pixel-size error" in p.lower() for p in rep["problems"]))

    def test_output_pixel_within_tolerance_ok(self):
        rep = pixel_size_consistency_report(
            unbinned_pixel_size_A=2.0, image_binned=8, aligned_pixel_size_A=16.0,
            warp_input_angpix_A=17.6, requested_output_angpix_A=17.6,
            output_voxel_size_A=17.61)
        self.assertTrue(rep["output_voxel_matches_request"])
        self.assertEqual(rep["problems"], [])


class SeamMetricTests(unittest.TestCase):
    def _flat(self, size=8):
        # small blocks + light intra-block texture so control differences are non-zero
        # (a perfectly flat block gives control=0 and an infinite, incomparable ratio).
        rng = np.random.default_rng(0)
        return rng.normal(0.0, 0.1, (size * 3, size * 3, size * 3)).astype(np.float32)

    def test_boundary_perpendicular_to_z_shows_in_XY(self):
        size = 8
        vol = self._flat(size)
        # step across every z-boundary (perpendicular to Z -> XY view)
        for z in range(vol.shape[0]):
            vol[z] += 10.0 if (z // size) % 2 else 0.0
        m = seam_metric(vol, size)
        self.assertGreater(m["orientations"]["XY"]["boundary_to_control_ratio"], 5.0)
        self.assertLess(m["orientations"]["YZ"]["boundary_to_control_ratio"], 2.0)

    def test_boundary_perpendicular_to_y_shows_in_XZ(self):
        size = 8
        vol = self._flat(size)
        for y in range(vol.shape[1]):
            vol[:, y, :] += 10.0 if (y // size) % 2 else 0.0
        m = seam_metric(vol, size)
        self.assertGreater(m["orientations"]["XZ"]["boundary_to_control_ratio"], 5.0)

    def test_boundary_perpendicular_to_x_shows_in_YZ(self):
        size = 8
        vol = self._flat(size)
        for x in range(vol.shape[2]):
            vol[:, :, x] += 10.0 if (x // size) % 2 else 0.0
        m = seam_metric(vol, size)
        self.assertGreater(m["orientations"]["YZ"]["boundary_to_control_ratio"], 5.0)

    def test_xz_vs_xy_comparison_reported(self):
        size = 8
        vol = self._flat(size)
        # stronger seam perpendicular to Y (XZ) than perpendicular to Z (XY)
        for y in range(vol.shape[1]):
            vol[:, y, :] += 20.0 if (y // size) % 2 else 0.0
        for z in range(vol.shape[0]):
            vol[z] += 2.0 if (z // size) % 2 else 0.0
        m = seam_metric(vol, size)
        self.assertTrue(m["xz_ratio_higher_than_xy"])


if __name__ == "__main__":
    unittest.main()
