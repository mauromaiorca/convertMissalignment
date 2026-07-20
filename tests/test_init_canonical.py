"""`init`: discover+measure ONCE, write one resolved canonical TOML with measured
geometry, tilt axis from align.com (84.0, NOT 0.0), condition->warp-mode mapping,
and separate xtilt/tltxf. Resolves 2.1/2.2/2.3/2.4/2.7/2.11. Requires mrcfile."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from imod_affine import write_xf

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
PREPARE = ROOT / "prepare_imod_to_warp.py"
PY = sys.executable


def _real_like_project(tmp: Path, bn="64x_Vero_02"):
    """Mimic the real dataset: distinct raw/aligned grids + align.com RotationAngle 84.0."""
    data = tmp / "data"; data.mkdir(); n = 5
    # raw bigger + finer pixel; aligned half-size + coarser pixel (like 1.363 vs 2.726)
    with mrcfile.new(data / f"{bn}.mrc", overwrite=True) as h:
        h.set_data(np.zeros((n, 320, 256), np.float32)); h.voxel_size = 1.363
    with mrcfile.new(data / f"{bn}_ali.mrc", overwrite=True) as h:
        h.set_data(np.zeros((n, 160, 128), np.float32)); h.voxel_size = 2.726
    ang = np.linspace(-40, 40, n)
    (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    (data / f"{bn}.rawtlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{bn}.xf", np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    (data / f"{bn}.xtilt").write_text("\n".join("0.0" for _ in range(n)) + "\n")   # SEPARATE
    (data / f"{bn}.tltxf").write_text("\n".join(
        "1 0 0 1 0 0" for _ in range(n)) + "\n")                                    # SEPARATE
    (data / "align.com").write_text("$tiltalign\nRotationAngle\t84.0\n")           # tilt axis source
    (data / "newst.com").write_text("$newstack\n")
    (data / "tilt.com").write_text("$tilt\nTHICKNESS 80\nIMAGEBINNED 1\n")
    s = tmp / "s.toml"
    s.write_text(f'''
[project]
basename = "{bn}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/out"
[conversion]
initial_conditions = ["raw_xf_affine_fixed", "ali_identity"]
[missalignment]
refinement_mode = "standard"
''')
    return data, s, bn


@unittest.skipUnless(HAVE, "mrcfile unavailable")
class InitCanonicalTests(unittest.TestCase):
    def _run(self, args):
        return subprocess.run([PY, str(PREPARE), *args], text=True, capture_output=True)

    def test_init_writes_one_resolved_toml(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _real_like_project(tmp)
            before = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            cp = self._run(["init", str(s)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            resolved = tmp / "out" / "project_settings.toml"
            self.assertTrue(resolved.is_file())
            cfg = tomllib.load(open(resolved, "rb"))
            # 2.2: geometry MEASURED, no empty strings
            self.assertEqual(cfg["geometry"]["raw_shape_xyz"], [256, 320, 5])
            self.assertEqual(cfg["geometry"]["aligned_shape_xyz"], [128, 160, 5])
            self.assertAlmostEqual(cfg["geometry"]["raw_pixel_size_A"], 1.363, places=2)
            self.assertAlmostEqual(cfg["geometry"]["aligned_pixel_size_A"], 2.726, places=2)
            # 2.11: tilt axis 84.0 from align.com, NOT 0.0
            self.assertAlmostEqual(cfg["geometry"]["tilt_axis_angle_deg"], 84.0, places=3)
            self.assertIn("align.com", cfg["geometry"]["tilt_axis_source"])
            # §4: target reconstruction volume = aligned shape + tilt.com THICKNESS,
            # at the TARGET (aligned/output) pixel, with physical == shape x pixel.
            self.assertEqual(cfg["geometry"]["target_volume_shape_xyz"], [128, 80, 160])
            self.assertAlmostEqual(cfg["geometry"]["target_pixel_size_A"], 2.726, places=2)
            self.assertEqual(cfg["datasets"]["native_id"], "2.726Apx")
            self.assertAlmostEqual(cfg["datasets"]["native_pixel_size_A"], 2.726, places=3)
            phys = cfg["geometry"]["target_volume_physical_A"]
            self.assertAlmostEqual(phys[0], 128 * 2.726, places=1)
            self.assertAlmostEqual(phys[1], 80 * 2.726, places=1)
            self.assertIn("THICKNESS", cfg["geometry"]["target_volume_source"])
            # 2.7: condition -> warp mode mapping, separate from refinement_mode
            self.assertEqual(cfg["conversion"]["condition_modes"]["raw_xf_affine_fixed"], "quarter-turn-affine")
            self.assertEqual(cfg["conversion"]["condition_modes"]["ali_identity"], "identity")
            self.assertEqual(cfg["missalignment"]["refinement_mode"], "standard")
            # 2.4: xtilt and tltxf SEPARATE fields, different files
            self.assertTrue(cfg["input"]["xtilt_file"].endswith(".xtilt"))
            self.assertTrue(cfg["input"]["tltxf_file"].endswith(".tltxf"))
            self.assertNotEqual(cfg["input"]["xtilt_file"], cfg["input"]["tltxf_file"])
            # final_xf must NOT be the .tltxf
            self.assertTrue(cfg["input"]["final_xf_file"].endswith(".xf"))
            self.assertFalse(cfg["input"]["final_xf_file"].endswith(".tltxf"))
            self.assertTrue(cfg["reconstruction"]["enabled"])
            self.assertEqual(cfg["reconstruction"]["snapshots"], ["pre_missalign", "smoke", "full"])
            self.assertTrue(cfg["reconstruction"]["warptools"]["enabled"])
            self.assertEqual(cfg["reconstruction"]["warptools"]["executable"], "WarpTools")
            self.assertEqual(cfg["reconstruction"]["imod"]["newst_template"], str(data / "newst.com"))
            self.assertEqual(cfg["reconstruction"]["imod"]["tilt_template"], str(data / "tilt.com"))
            self.assertEqual(cfg["reconstruction"]["imod"]["volume"]["shape_xyz"], [128, 80, 160])
            # provenance + manifests
            self.assertTrue(cfg["provenance"]["resolved"])
            md = tmp / "out" / "provenance"
            for m in ("source_inventory.json", "source_hashes.json", "geometry_manifest.json"):
                self.assertTrue((md / m).is_file(), m)
            # source READ-ONLY
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in data.iterdir()})

    def test_init_refuses_zero_tilt_axis(self):
        # no align.com and no explicit angle -> must FAIL, never default to 0.0
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _real_like_project(tmp)
            (data / "align.com").unlink()
            cp = self._run(["init", str(s)])
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("0.0", cp.stdout + cp.stderr)  # explains refusal to default to 0.0

    def test_prepare_hard_fails_on_unresolved(self):
        # §5 consume-only: prepare must REFUSE an unresolved TOML (no discovery fallback).
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _real_like_project(tmp)   # s.toml is NOT resolved
            cp = self._run(["prepare", str(s)])
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("unresolved config", (cp.stdout + cp.stderr))
            self.assertIn("init", (cp.stdout + cp.stderr))

    def test_init_then_prepare_consumes_resolved(self):
        # the production flow: init -> prepare on the resolved TOML (no migration flag).
        import shutil
        if not shutil.which("newstack"):
            self.skipTest("newstack needed for prepare")
        env = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
               "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _real_like_project(tmp)
            init = subprocess.run([PY, str(PREPARE), "init", str(s)], env=env, text=True, capture_output=True)
            self.assertEqual(init.returncode, 0, init.stdout + init.stderr)
            resolved = tmp / "out" / "project_settings.toml"
            # prepare on the RESOLVED toml without any migration flag -> succeeds
            prep = subprocess.run([PY, str(PREPARE), "prepare", str(resolved), "--allow-unavailable-mode"],
                                  env=env, text=True, capture_output=True)
            self.assertEqual(prep.returncode, 0, prep.stdout[-1500:] + prep.stderr[-1500:])
            self.assertNotIn("unresolved config", prep.stdout + prep.stderr)

    def test_resolved_toml_loads_via_canonical_loader(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _real_like_project(tmp)
            self.assertEqual(self._run(["init", str(s)]).returncode, 0)
            resolved = tmp / "out" / "project_settings.toml"
            sys.path.insert(0, str(ROOT / "scripts"))
            from pipeline import project_config as PC
            rc = PC.load(resolved)
            rc.require_resolved()  # must not raise
            self.assertEqual(PC.validate(rc, require_geometry=True, require_resolved=True), [])
            self.assertEqual(rc.warp_mode("raw_xf_affine_fixed"), "quarter-turn-affine")


if __name__ == "__main__":
    unittest.main()
