"""Unit tests for the Phase-3 WarpTools diagnostic executor."""
from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import warptools_reconstruction as WR


class WarpToolsReconstructionTests(unittest.TestCase):
    def test_conversion_contract_supplies_current_warp_xyz_shape(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {
                "output_pixel_size_A": 2.0,
                "warp_volume_shape_xyz": [80, 60, 25],
                "volume_frame": {
                    "contract_version": 2,
                    "reconstruction_shape_warp_xyz": [80, 60, 25],
                    "current_shape_warp_xyz": [80, 60, 25],
                    "source_shape_imod_mrc_xyz": [80, 25, 60],
                    "projection_quarter_turn_k": 1,
                    "projection_quarter_turn_scope": "detector_frame_only",
                },
            }
            (root / "series.conversion.json").write_text(json.dumps(manifest))
            contract = WR.load_conversion_volume_contract(
                root,
                "series",
                xml_volume_dimensions_A=[160.0, 120.0, 50.0],
            )
            self.assertEqual(contract["shape_warp_xyz"], [80, 60, 25])

    def test_legacy_conversion_without_frame_contract_is_rejected(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "series.conversion.json").write_text(json.dumps({
                "output_pixel_size_A": 2.0,
                "volume_shape_xyz": [60, 25, 80],
            }))
            with self.assertRaisesRegex(
                WR.WarpToolsReconstructionError,
                "legacy or invalid|lacks Warp reconstruction XYZ shape",
            ):
                WR.load_conversion_volume_contract(root, "series")


    def test_contract_v1_odd_quarter_turn_is_rejected(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {
                "output_pixel_size_A": 2.0,
                "warp_volume_shape_xyz": [80, 60, 25],
                "volume_frame": {
                    "contract_version": 1,
                    "base_shape_warp_xyz": [60, 80, 25],
                    "current_shape_warp_xyz": [80, 60, 25],
                    "projection_quarter_turn_k": 1,
                },
            }
            (root / "series.conversion.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                WR.WarpToolsReconstructionError,
                "stale contract-v1 affine conversion",
            ):
                WR.load_conversion_volume_contract(root, "series")

    def test_contract_v1_translation_is_accepted(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {
                "output_pixel_size_A": 2.0,
                "warp_volume_shape_xyz": [60, 80, 25],
                "volume_frame": {
                    "contract_version": 1,
                    "base_shape_warp_xyz": [60, 80, 25],
                    "current_shape_warp_xyz": [60, 80, 25],
                    "projection_quarter_turn_k": 0,
                },
            }
            (root / "series.conversion.json").write_text(json.dumps(manifest))
            contract = WR.load_conversion_volume_contract(
                root,
                "series",
                xml_volume_dimensions_A=[120.0, 160.0, 50.0],
            )
            self.assertEqual(contract["shape_warp_xyz"], [60, 80, 25])
            self.assertTrue(contract["legacy_v1_translation_accepted"])

    def test_constant_dose_is_repaired_with_finite_epsilon_ramp(self):
        values, policy, pre_ok, full_ok = WR.choose_dose_values(
            np.zeros(42), np.zeros(42), 42
        )
        self.assertFalse(pre_ok)
        self.assertFalse(full_ok)
        self.assertEqual(
            policy,
            "synthetic_monotonic_epsilon_for_warp_coordinate_only",
        )
        self.assertEqual(len(values), 42)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(float(np.ptp(values)), 0.0)
        normalized = (values - values.min()) / np.ptp(values)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_identical_valid_dose_is_preserved(self):
        source = np.linspace(0.0, 80.0, 42)
        values, policy, pre_ok, full_ok = WR.choose_dose_values(
            source, source.copy(), 42
        )
        self.assertTrue(pre_ok)
        self.assertTrue(full_ok)
        self.assertEqual(policy, "preserved_identical_source_dose")
        np.testing.assert_allclose(values, source)

    def test_patch_xml_writes_movie_paths_and_nonconstant_dose(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.xml"
            destination = root / "copy.xml"
            source.write_text(
                '<TiltSeries VolumeDimensionsAngstrom="4,4,4">'
                '<Angles>0\n1</Angles><Dose>0\n0</Dose></TiltSeries>\n'
            )
            raw = root / "raw"
            raw.mkdir()
            WR._patch_xml(
                source,
                destination,
                raw_data_dir=raw,
                movie_names=["a.mrc", "b.mrc"],
                dose_values=[0.0, 1e-6],
            )
            xml = ET.parse(destination).getroot()
            self.assertEqual(xml.attrib["DataDirectory"], str(raw))
            movies = xml.find("MoviePath").text.splitlines()
            dose = [float(v) for v in xml.find("Dose").text.splitlines()]
            self.assertEqual(movies, ["a.mrc", "b.mrc"])
            self.assertGreater(max(dose) - min(dose), 0.0)


if __name__ == "__main__":
    unittest.main()
