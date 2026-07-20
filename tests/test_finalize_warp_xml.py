"""§14: finalize result_backend=warp_xml must invoke export_condition_results.py with the
CORRECT CLI (--params/--warp-dir/--condition/--out-dir/--xml/--rms-tolerance-px/
--max-tolerance-px). The historical call passed a positional `settings` and no --params/
--warp-dir, so it ALWAYS failed at argparse. The full XML->XF numeric path needs warpylib
(cluster-only); locally we prove the command is well-formed and the exporter accepts it,
reaching the warpylib boundary rather than an argparse error."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import finalize as FIN

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
EXPORTER = ROOT / "scripts" / "export_condition_results.py"
PY = sys.executable


def _cfg(tmp: Path, raw, ali, xf, tlt, bn="64x_Vero_02"):
    return {
        "project": {"basename": bn},
        "paths": {"data_root": str(tmp), "output_dir": str(tmp / "out")},
        "input": {"raw_stack": str(raw), "aligned_stack": str(ali),
                  "final_xf_file": str(xf), "final_tilt_file": str(tlt)},
        "geometry": {"tilt_axis_angle_deg": 84.0, "raw_pixel_size_A": 1.363,
                     "aligned_pixel_size_A": 2.726, "target_volume_shape_xyz": [32, 20, 40],
                     "target_pixel_size_A": 2.726},
        "conversion": {"initial_conditions": ["raw_xf_affine_fixed"]},
        "ctf": {"mode": "off"},
        "missalignment": {"refinement_mode": "standard", "result_backend": "warp_xml"},
        "provenance": {"resolved": True},
    }


class BuildCommandTests(unittest.TestCase):
    def test_command_has_required_flags_and_no_positional(self):
        cmd = FIN._build_export_command(
            exporter="/x/export_condition_results.py", params="/p.json", warp_dir="/w",
            condition="raw_xf_affine_fixed", out_dir="/o", xml="/f.xml",
            rms_tol=0.10, max_tol=0.25)
        # the exporter REQUIRES these; the old call omitted them and passed a positional.
        for flag in ("--params", "--warp-dir", "--condition", "--out-dir", "--xml",
                     "--rms-tolerance-px", "--max-tolerance-px"):
            self.assertIn(flag, cmd, flag)
        self.assertEqual(cmd[cmd.index("--condition") + 1], "raw_xf_affine_fixed")
        # no stray positional settings token (everything after the exporter is a flag/value)
        self.assertTrue(str(cmd[1]).endswith("export_condition_results.py"))
        self.assertTrue(str(cmd[2]).startswith("--"))


@unittest.skipUnless(HAVE, "mrcfile needed")
class SynthesizeParamsTests(unittest.TestCase):
    def test_params_shape_matches_exporter_contract(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _cfg(tmp, "/r.mrc", "/a.mrc", "/s.xf", "/s.tlt")
            dest = tmp / "missalign_params.json"
            FIN._synthesize_missalign_params(cfg, "raw_xf_affine_fixed", dest)
            p = json.loads(dest.read_text())
            self.assertEqual(p["series_name"], "64x_Vero_02")
            self.assertEqual(p["files"]["raw_stack"], "/r.mrc")
            self.assertEqual(p["files"]["aligned_stack"], "/a.mrc")
            self.assertIn("raw_xf_affine_fixed", p["conditions"])
            self.assertEqual(p["geometry"]["target_volume_shape_xyz"], [32, 20, 40])
            self.assertAlmostEqual(p["geometry"]["target_output_pixel_size_A"], 2.726, places=3)


@unittest.skipUnless(HAVE, "mrcfile needed")
class ExporterAcceptsCommandTests(unittest.TestCase):
    """The exporter must ACCEPT our CLI and proceed through params/manifest/header parsing,
    failing only at the warpylib boundary — never at argparse (which the old call hit)."""

    def _stack(self, path, n, ny, nx, pix):
        with mrcfile.new(path, overwrite=True) as h:
            h.set_data(np.random.rand(n, ny, nx).astype(np.float32)); h.voxel_size = pix

    def test_reaches_warpylib_not_argparse(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); n = 5
            raw = tmp / "raw.mrc"; ali = tmp / "ali.mrc"
            self._stack(raw, n, 40, 32, 1.363)
            self._stack(ali, n, 20, 16, 2.726)
            xf = tmp / "s.xf"; xf.write_text("\n".join("1 0 0 1 0 0" for _ in range(n)) + "\n")
            tlt = tmp / "s.tlt"; tlt.write_text("\n".join("0.0" for _ in range(n)) + "\n")
            cfg = _cfg(tmp, raw, ali, xf, tlt)
            params = FIN._synthesize_missalign_params(cfg, "raw_xf_affine_fixed",
                                                      tmp / "params.json")
            # a warp dir with EXACTLY one conversion manifest + an XML file
            warp_dir = tmp / "warp" / "warp_raw_xf_affine_fixed"; warp_dir.mkdir(parents=True)
            (warp_dir / "TS.conversion.json").write_text(json.dumps({
                "image_shape_zyx": [n, 20, 16], "output_pixel_size_A": 2.726,
                "output_stack": str(warp_dir / "TS.st")}))
            (warp_dir / "TS.xml").write_text("<TiltSeries></TiltSeries>\n")
            cmd = FIN._build_export_command(
                exporter=EXPORTER, params=params, warp_dir=warp_dir,
                condition="raw_xf_affine_fixed", out_dir=tmp / "out", xml=warp_dir / "TS.xml",
                rms_tol=0.10, max_tol=0.25)
            cp = subprocess.run(cmd, text=True, capture_output=True)
            blob = cp.stdout + cp.stderr
            # PROOF the CLI is correct: no argparse rejection
            self.assertNotIn("the following arguments are required", blob, blob[-500:])
            self.assertNotIn("unrecognized arguments", blob, blob[-500:])
            # it proceeded all the way to the (cluster-only) warpylib requirement
            self.assertIn("warpylib", blob, blob[-800:])


if __name__ == "__main__":
    unittest.main()
