"""Validate the production repair against the REAL testABC fixture (text files copied
into tests/fixtures/testABC/). Confirms the canonical loader normalizes the actual
legacy TOML, the condition->warp-mode/axis-frame map matches the real
etomo_missalign_params.json, and the 2.10 volume invariant CATCHES the doubling that
the shipped Warp XML actually contains."""
from __future__ import annotations

import json
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import project_config as PC
from geometry.volume_frames import imod_mrc_shape_to_warp_xyz

FIX = ROOT / "tests" / "fixtures" / "testABC"


@unittest.skipUnless(FIX.is_dir(), "testABC fixture not present")
class TestABCFixtureTests(unittest.TestCase):
    def test_real_legacy_toml_normalizes(self):
        cfg = tomllib.load(open(FIX / "project_settings.toml", "rb"))
        rc = PC.from_dict(cfg)
        # legacy [project].data_dir/out_dir + [paths].missalign_environment
        self.assertIn("reconst_64x_Vero_02", rc.data_root)
        self.assertTrue(rc.output_dir.endswith("testABC"))
        self.assertEqual(rc.conditions, ["raw_xf_affine_fixed"])
        self.assertEqual(rc.cluster.partition, "vds")        # [slurm].gpu_partition
        self.assertEqual(rc.cluster.constraint, "V100")
        # env path lives under [paths].missalign_environment in the real file
        self.assertTrue(rc.cluster.environment and rc.cluster.environment.endswith("envs/missalign"),
                        f"env not normalized: {rc.cluster.environment}")
        self.assertEqual(rc.cluster.imod_module, "imod")     # [external_tools].imod_module
        # 2.4: the real TOML conflates tltxf_file = the .xtilt path (the defect)
        self.assertTrue(cfg["input"]["tltxf_file"].endswith(".xtilt"))

    def test_condition_mode_and_axis_frame_match_real_extraction(self):
        params = json.load(open(FIX / "etomo_missalign_params.json"))
        for cond, v in params["conditions"].items():
            expected_mode = v["alignment_mode"]
            if cond == "raw_xf_affine_fixed":
                # Historical fixture documents the v6 full-affine encoding.
                # v7 deliberately replaces only this mode with the quarter-turn factorization.
                self.assertEqual(expected_mode, "full-affine")
                self.assertEqual(PC.warp_alignment_mode_for(cond), "quarter-turn-affine")
            else:
                self.assertEqual(PC.warp_alignment_mode_for(cond), expected_mode,
                                 f"{cond}: warp mode mismatch")
            self.assertEqual(PC.axis_frame_for(cond), v["axis_frame"],
                             f"{cond}: axis_frame mismatch")
        # spot-check the one I had wrong before the fixture
        self.assertEqual(PC.warp_alignment_mode_for("raw_xf"), "translation")

    def test_real_geometry_values(self):
        g = json.load(open(FIX / "etomo_missalign_params.json"))["geometry"]
        self.assertAlmostEqual(g["tilt_axis_angle_deg"], 84.0, places=3)
        self.assertIn("align.com", g["tilt_axis_angle_source"])
        self.assertAlmostEqual(g["raw_pixel_size_A"], 1.363, places=2)
        self.assertAlmostEqual(g["aligned_pixel_size_A"], 2.726, places=2)
        self.assertEqual(g["target_volume_shape_xyz"], [2046, 494, 2880])

    def test_volume_invariant_catches_shipped_doubling(self):
        # parse VolumeDimensionsAngstrom from the REAL shipped XML header
        xml = (FIX / "warp_header.xml").read_text()
        m = re.search(r'VolumeDimensionsAngstrom="([^"]+)"', xml)
        self.assertIsNotNone(m)
        vol = [float(x) for x in m.group(1).split(",")]
        params = json.load(open(FIX / "etomo_missalign_params.json"))["geometry"]
        target_shape = params["target_volume_shape_xyz"]      # [2046, 494, 2880]
        target_pixel = params["target_output_pixel_size_A"]   # 2.726
        # the shipped XML FAILS the invariant (it is ~2x the target) -> proves the bug
        self.assertFalse(PC.volume_invariant_ok(vol, target_shape, target_pixel),
                         "shipped XML unexpectedly passed; the 2.10 doubling check is wrong")
        ratio = vol[0] / (target_shape[0] * target_pixel)
        self.assertAlmostEqual(ratio, 2.0, places=1)          # doubled
        # The source target shape is IMOD reconstruction MRC storage order
        # (X,Y_thickness,Z). Warp base XYZ is therefore (X,Z,Y).
        warp_shape = imod_mrc_shape_to_warp_xyz(target_shape)
        correct = [s * target_pixel for s in warp_shape]
        self.assertTrue(PC.volume_invariant_ok(correct, warp_shape, target_pixel))
        # and assert_volume_invariant raises on the shipped values in Warp XYZ
        with self.assertRaises(PC.ConfigError):
            PC.assert_volume_invariant(vol, warp_shape, target_pixel)

    def test_conversion_json_records_raw_shape_at_output_pixel(self):
        # the conversion.json itself shows volume_shape in RAW pixels with output pixel
        # 2.726 -> the root cause the invariant guards against.
        conv = json.load(open(FIX / "conversion.json"))
        self.assertEqual(conv["volume_shape_xyz"], [4092, 988, 5760])  # raw shape
        self.assertAlmostEqual(conv["output_pixel_size_A"], 2.726, places=2)
        self.assertEqual(conv["alignment_mode"], "full-affine")
        self.assertEqual(conv["axis_frame"], "aligned")


if __name__ == "__main__":
    unittest.main()
