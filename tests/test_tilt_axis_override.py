"""Regression test: the tilt-axis-angle override must reach the params JSON.

Defect (Phase 2/6): ``setup_missalign_project.py`` writes ``tilt_axis_angle_deg``
into ``project_settings.toml`` and exposes ``--tilt-axis-angle``, but neither
``setup_missalign_project.sh`` nor ``01_extract_etomo_params.py`` consumed it,
so the documented override was dead.  When IMOD ``align.com``/mdoc lack a
parseable rotation angle, ``02_convert_using_params.py`` then aborts with
"tilt-axis angle is missing" and the user has no working override.

This test builds a synthetic eTomo directory with NO parseable tilt-axis angle
(no align.com / align.log / mdoc) and verifies that the override reaches the
generated params JSON, both via the ``MISSALIGN_TILT_AXIS_ANGLE`` environment
variable (the path used by the shell front-end) and via the new
``--tilt-axis-angle`` CLI flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mrcfile
    HAVE_MRCFILE = True
except Exception:
    HAVE_MRCFILE = False

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "scripts" / "01_extract_etomo_params.py"


def _make_project(tmp: Path) -> Path:
    series = "synthseries"
    etomo = tmp / "etomo"
    etomo.mkdir()
    n = 3
    with mrcfile.new(etomo / f"{series}.st", overwrite=True) as h:
        h.set_data(np.zeros((n, 16, 24), dtype=np.float32))
        h.voxel_size = 10.0
    (etomo / f"{series}.rawtlt").write_text("\n".join(str(v) for v in (-20.0, 0.0, 20.0)) + "\n")
    (etomo / f"{series}.tlt").write_text("\n".join(str(v) for v in (-20.0, 0.0, 20.0)) + "\n")
    # identity .xf, 3 rows -> no parseable tilt-axis angle anywhere
    (etomo / f"{series}.xf").write_text(
        "".join("   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n" for _ in range(n))
    )
    return etomo


@unittest.skipUnless(HAVE_MRCFILE, "mrcfile unavailable")
class TiltAxisOverrideTests(unittest.TestCase):
    def _run_extract(self, etomo: Path, out: Path, env_extra: dict, cli_extra: list):
        env = dict(os.environ)
        # Ensure nothing from the outer shell leaks a tilt-axis value.
        env.pop("MISSALIGN_TILT_AXIS_ANGLE", None)
        env.update(env_extra)
        cmd = [
            sys.executable, str(EXTRACT),
            "--etomo-dir", str(etomo),
            "--out-dir", str(out),
            "--basename", "synthseries",
            "--imod-dir", str(etomo),
            "--raw-stack", str(etomo / "synthseries.st"),
            "--raw-tilt-file", str(etomo / "synthseries.rawtlt"),
            "--final-tilt-file", str(etomo / "synthseries.tlt"),
            "--final-xf-file", str(etomo / "synthseries.xf"),
            # Provide volume geometry so the only possible warning is the
            # tilt-axis angle, which this test controls.
            "--target-volume-xyz", "24x16x8",
            "--overwrite",
            *cli_extra,
        ]
        cp = subprocess.run(cmd, env=env, text=True, capture_output=True)
        json_path = out / "etomo_missalign_params.json"
        # The JSON is written before the warnings-exit (rc==2). If it is missing
        # the run genuinely failed (e.g. an unrecognised CLI flag, rc==2 from
        # argparse) -- surface stderr so the defect is visible.
        self.assertTrue(
            json_path.is_file(),
            f"params JSON not written (rc={cp.returncode}):\n{cp.stdout}\n{cp.stderr}",
        )
        return json.loads(json_path.read_text()), cp.returncode

    def test_baseline_has_no_parseable_angle(self):
        """Sanity: without an override the angle is null (defect precondition)."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            etomo = _make_project(tmp)
            params, rc = self._run_extract(etomo, tmp / "out0", {}, [])
            self.assertIsNone(params["geometry"]["tilt_axis_angle_deg"])
            self.assertEqual(rc, 2)  # warns about the missing tilt-axis angle

    def test_env_override_reaches_params(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            etomo = _make_project(tmp)
            params, rc = self._run_extract(
                etomo, tmp / "out1", {"MISSALIGN_TILT_AXIS_ANGLE": "42.5"}, []
            )
            self.assertAlmostEqual(params["geometry"]["tilt_axis_angle_deg"], 42.5, places=6)
            self.assertEqual(rc, 0, "override should remove the tilt-axis warning")
            self.assertIn("override", str(params["geometry"]["tilt_axis_angle_source"]).lower())

    def test_cli_override_reaches_params(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            etomo = _make_project(tmp)
            params, rc = self._run_extract(
                etomo, tmp / "out2", {}, ["--tilt-axis-angle", "-7.25"]
            )
            self.assertAlmostEqual(params["geometry"]["tilt_axis_angle_deg"], -7.25, places=6)

    def test_zero_override_is_honored(self):
        """A 0-degree tilt axis is a valid override and must not be dropped."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            etomo = _make_project(tmp)
            params, rc = self._run_extract(
                etomo, tmp / "out3", {"MISSALIGN_TILT_AXIS_ANGLE": "0.0"}, []
            )
            self.assertEqual(params["geometry"]["tilt_axis_angle_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
