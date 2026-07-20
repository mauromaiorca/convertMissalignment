"""Canonical project config: one loader, legacy normalization, the three separate
mode concepts (2.7), xtilt vs tltxf (2.4), resolved-geometry validation (2.2/2.11)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import project_config as PC


class ProjectConfigTests(unittest.TestCase):
    def test_legacy_dialect_normalized(self):
        # the setup_missalign_project.py dialect must load via the one canonical loader
        cfg = {
            "project": {"basename": "64x_Vero_02", "data_dir": "/src", "out_dir": "/out"},
            "input": {"conditions": ["raw_xf_affine_fixed"], "aligned_stack": "/src/a_ali.mrc"},
            "slurm": {"gpu_partition": "vds", "gpu_constraint": "V100", "standard_cpus": 16},
        }
        rc = PC.from_dict(cfg)
        self.assertEqual(rc.data_root, "/src")          # [project].data_dir -> [paths].data_root
        self.assertEqual(rc.output_dir, "/out")
        self.assertEqual(rc.conditions, ["raw_xf_affine_fixed"])
        self.assertEqual(rc.cluster.partition, "vds")    # [slurm] -> [cluster]
        self.assertEqual(rc.cluster.cpus, 16)

    def test_three_mode_concepts_separate(self):
        rc = PC.from_dict({
            "project": {"basename": "b"}, "paths": {"data_root": "/s", "output_dir": "/o"},
            "conversion": {"initial_conditions": ["raw_xf_affine_fixed", "ali_identity"]},
            "missalignment": {"refinement_mode": "standard"},
        })
        # condition -> warp mode is derived and DISTINCT from refinement_mode
        self.assertEqual(rc.warp_mode("raw_xf_affine_fixed"), "quarter-turn-affine")
        self.assertEqual(rc.warp_mode("ali_identity"), "identity")
        self.assertEqual(rc.refinement_mode, "standard")
        self.assertNotIn(rc.refinement_mode, PC.WARP_ALIGNMENT_MODES)

    def test_validate_rejects_mode_confusion(self):
        # a refinement mode used as a warp mode must be flagged (2.7)
        rc = PC.from_dict({
            "project": {"basename": "b"}, "paths": {"data_root": "/s", "output_dir": "/o"},
            "conversion": {"initial_conditions": ["c"], "condition_modes": {"c": "rigid"}},
        })
        problems = PC.validate(rc)
        self.assertTrue(any("concept confusion" in p or "not in" in p for p in problems))

    def test_validate_rejects_xtilt_tltxf_conflation(self):
        rc = PC.from_dict({
            "project": {"basename": "b"}, "paths": {"data_root": "/s", "output_dir": "/o"},
            "input": {"xtilt_file": "/s/x.xtilt", "tltxf_file": "/s/x.xtilt"},
        })
        self.assertTrue(any("conflation" in p for p in PC.validate(rc)))

    def test_resolved_geometry_required(self):
        rc = PC.from_dict({
            "project": {"basename": "b"}, "paths": {"data_root": "/s", "output_dir": "/o"},
            "geometry": {},  # empty -> must fail require_geometry (2.2)
        })
        problems = PC.validate(rc, require_geometry=True)
        self.assertTrue(any("tilt_axis_angle_deg" in p for p in problems))

    def test_require_resolved_raises_for_unresolved(self):
        rc = PC.from_dict({"project": {"basename": "b"},
                           "paths": {"data_root": "/s", "output_dir": "/o"}})
        self.assertFalse(rc.resolved)
        with self.assertRaises(PC.ConfigError):
            rc.require_resolved()

    def test_roundtrip_resolved_toml(self):
        rc = PC.from_dict({
            "project": {"basename": "64x_Vero_02"},
            "paths": {"data_root": "/s", "output_dir": "/o"},
            "input": {"aligned_stack": "/s/a_ali.mrc", "final_tilt_file": "/s/a.tlt",
                      "xtilt_file": "/s/a.xtilt", "tltxf_file": "/s/a.tltxf"},
            "geometry": {"tilt_axis_angle_deg": 84.0, "tilt_axis_source": "align.com: RotationAngle",
                         "raw_pixel_size_A": 1.363, "aligned_pixel_size_A": 2.726,
                         "target_volume_shape_xyz": [128, 80, 160], "target_pixel_size_A": 2.726,
                         "target_volume_physical_A": [128 * 2.726, 80 * 2.726, 160 * 2.726]},
            "conversion": {"initial_conditions": ["ali_identity"]},
            "missalignment": {"refinement_mode": "standard", "result_backend": "warp_xml"},
            "provenance": {"resolved": True},
        })
        d = rc.to_dict()
        self.assertEqual(d["geometry"]["tilt_axis_angle_deg"], 84.0)
        self.assertEqual(d["input"]["xtilt_file"], "/s/a.xtilt")
        self.assertEqual(d["input"]["tltxf_file"], "/s/a.tltxf")
        self.assertEqual(d["missalignment"]["result_backend"], "warp_xml")
        self.assertTrue(d["provenance"]["resolved"])
        rc.require_resolved()  # does not raise
        self.assertEqual(PC.validate(rc, require_geometry=True, require_resolved=True), [])

    def test_tomllib_load_real_example(self):
        # the shipped example must load via the canonical loader
        ex = ROOT / "config" / "examples" / "64x_Vero_02_bin8_translation_noctf.toml"
        if ex.is_file():
            rc = PC.load(ex)
            self.assertEqual(rc.basename, "64x_Vero_02")
            self.assertIn("ali_identity", rc.conditions)


if __name__ == "__main__":
    unittest.main()
