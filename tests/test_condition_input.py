"""§7 condition input adapter: the single authority mapping a condition onto the concrete
Warp converter inputs (stack / .xf / alignment mode / axis frame / grids). Two hard rules:
an *_identity condition never consumes a real .xf; an *_xf condition never gets a synthesized
identity .xf. Plus §3 config-expansion validation (no required value empty in a resolved
config) and the end-to-end proof that raw_xf_affine_fixed stages the REAL source .xf."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from imod_affine import write_xf
from pipeline import project_config as PC

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
PREPARE = ROOT / "prepare_imod_to_warp.py"
PY = sys.executable
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _paths(tmp: Path, *, with_xf=True, n=5):
    raw = tmp / "raw.mrc"; raw.write_bytes(b"x")
    ali = tmp / "ali.mrc"; ali.write_bytes(b"x")
    tlt = tmp / "s.tlt"; tlt.write_text("\n".join("0.0" for _ in range(n)) + "\n")
    xf = tmp / "s.xf"
    if with_xf:
        write_xf(xf, np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    return raw, ali, tlt, xf


class AdapterUnitTests(unittest.TestCase):
    def _geom(self):
        return PC.Geometry(raw_shape_xyz=[256, 320, 5], raw_pixel_size_A=1.363,
                           aligned_shape_xyz=[128, 160, 5], aligned_pixel_size_A=2.726,
                           target_volume_shape_xyz=[128, 80, 160], target_pixel_size_A=2.726)

    def test_raw_xf_affine_fixed_uses_raw_stack_and_real_xf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); raw, ali, tlt, xf = _paths(tmp)
            ci = PC.condition_input_from_paths(
                "raw_xf_affine_fixed", raw_stack=str(raw), aligned_stack=str(ali),
                final_xf_file=str(xf), final_tilt_file=str(tlt), geometry=self._geom())
            self.assertEqual(ci.stack, str(raw))           # RAW stack, not aligned
            self.assertEqual(ci.stack_role, "raw_stack")
            self.assertEqual(ci.stack_grid, "raw")
            self.assertEqual(ci.alignment_mode, "quarter-turn-affine")
            self.assertEqual(ci.axis_frame, "aligned")
            self.assertEqual(ci.initial_xf, str(xf))       # the REAL .xf
            self.assertEqual(ci.source_xf, str(xf))
            self.assertFalse(ci.is_identity)
            self.assertEqual(ci.grids["target"].shape_xy, (128, 80))

    def test_ali_identity_uses_aligned_stack_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); raw, ali, tlt, xf = _paths(tmp)
            ci = PC.condition_input_from_paths(
                "ali_identity", raw_stack=str(raw), aligned_stack=str(ali),
                final_xf_file=str(xf), final_tilt_file=str(tlt), geometry=self._geom())
            self.assertEqual(ci.stack, str(ali))           # ALIGNED stack
            self.assertEqual(ci.stack_grid, "aligned")
            self.assertEqual(ci.alignment_mode, "identity")
            self.assertEqual(ci.axis_frame, "aligned")
            self.assertIsNone(ci.initial_xf)               # identity (no real .xf)
            self.assertIsNone(ci.source_xf)
            self.assertTrue(ci.is_identity)

    def test_xf_condition_refuses_missing_xf(self):
        # §7 hard rule: an *_xf condition must NOT silently become identity.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); raw, ali, tlt, xf = _paths(tmp, with_xf=False)
            with self.assertRaises(PC.ConfigError) as cm:
                PC.condition_input_from_paths(
                    "raw_xf_affine_fixed", raw_stack=str(raw), aligned_stack=str(ali),
                    final_xf_file=None, final_tilt_file=str(tlt), geometry=self._geom())
            self.assertIn("identity", str(cm.exception).lower())

    def test_xf_condition_refuses_tltxf_as_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); raw, ali, tlt, xf = _paths(tmp)
            tltxf = tmp / "s.tltxf"; tltxf.write_text("1 0 0 1 0 0\n")
            with self.assertRaises(PC.ConfigError) as cm:
                PC.condition_input_from_paths(
                    "raw_xf_affine_fixed", raw_stack=str(raw), aligned_stack=str(ali),
                    final_xf_file=str(tltxf), final_tilt_file=str(tlt), geometry=self._geom())
            self.assertIn("tltxf", str(cm.exception).lower())

    def test_raw_condition_refuses_missing_raw_stack(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); raw, ali, tlt, xf = _paths(tmp)
            with self.assertRaises(PC.ConfigError):
                PC.condition_input_from_paths(
                    "raw_xf_affine_fixed", raw_stack=None, aligned_stack=str(ali),
                    final_xf_file=str(xf), final_tilt_file=str(tlt), geometry=self._geom())

    def test_require_files_off_skips_isfile_but_keeps_presence(self):
        # require_files=False: no is_file() check, but the *_xf presence rule still fires.
        ci = PC.condition_input_from_paths(
            "raw_xf_affine_fixed", raw_stack="/nope/raw.mrc", aligned_stack="/nope/ali.mrc",
            final_xf_file="/nope/s.xf", final_tilt_file="/nope/s.tlt", require_files=False)
        self.assertEqual(ci.stack, "/nope/raw.mrc")
        self.assertFalse(ci.is_identity)
        with self.assertRaises(PC.ConfigError):
            PC.condition_input_from_paths(
                "raw_xf_affine_fixed", raw_stack="/nope/raw.mrc", aligned_stack=None,
                final_xf_file=None, final_tilt_file="/nope/s.tlt", require_files=False)


class ValidateExpansionTests(unittest.TestCase):
    """§3: a resolved config with an empty required source / inconsistent target fails validate."""

    def _rc(self, **over):
        sources = PC.SourcePaths(raw_stack="/r.mrc", aligned_stack="/a.mrc",
                                 final_xf_file="/s.xf", final_tilt_file="/s.tlt")
        geom = PC.Geometry(tilt_axis_angle_deg=84.0, raw_pixel_size_A=1.363,
                           aligned_pixel_size_A=2.726, target_volume_shape_xyz=[128, 80, 160],
                           target_pixel_size_A=2.726,
                           target_volume_physical_A=[128 * 2.726, 80 * 2.726, 160 * 2.726])
        rc = PC.ResolvedProjectConfig(
            basename="b", data_root="/d", output_dir="/o", sources=sources, geometry=geom,
            conditions=["raw_xf_affine_fixed", "ali_identity"],
            warp_alignment_modes={"raw_xf_affine_fixed": "full-affine", "ali_identity": "identity"},
            refinement_mode="standard", result_backend="warp_xml", ctf_mode="off",
            extra_projection_binning=1, cluster=PC.ClusterConfig(), resolved=True)
        for k, v in over.items():
            setattr(rc, k, v)
        return rc

    def test_clean_resolved_validates(self):
        self.assertEqual(PC.validate(self._rc(), require_geometry=True, require_resolved=True), [])

    def test_empty_required_xf_flagged(self):
        rc = self._rc()
        rc.sources.final_xf_file = ""    # raw_xf_affine_fixed needs it
        probs = PC.validate(rc, require_geometry=True, require_resolved=True)
        self.assertTrue(any("final_xf_file" in p for p in probs), probs)

    def test_inconsistent_target_physical_flagged(self):
        rc = self._rc()
        rc.geometry.target_volume_physical_A = [1.0, 1.0, 1.0]   # != shape*pixel
        probs = PC.validate(rc, require_geometry=True, require_resolved=True)
        self.assertTrue(any("invariant" in p for p in probs), probs)


@unittest.skipUnless(HAVE and NEWSTACK, "mrcfile/newstack needed")
class EndToEndStagingTests(unittest.TestCase):
    """The real proof: prepare raw_xf_affine_fixed records the REAL source .xf for conversion,
    never identity, and feeds the RAW stack — exercised through the canonical init->prepare."""

    def _project(self, tmp: Path, bn="64x_Vero_02", n=5):
        data = tmp / "data"; data.mkdir()
        with mrcfile.new(data / f"{bn}.mrc", overwrite=True) as h:
            h.set_data(np.random.rand(n, 64, 48).astype(np.float32)); h.voxel_size = 1.363
        with mrcfile.new(data / f"{bn}_ali.mrc", overwrite=True) as h:
            h.set_data(np.random.rand(n, 32, 24).astype(np.float32)); h.voxel_size = 2.726
        ang = np.linspace(-40, 40, n)
        (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
        (data / f"{bn}.rawtlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
        # a NON-identity .xf so identity-vs-real is observable
        A = np.stack([np.array([[1.0, 0.02], [-0.02, 1.0]])] * n)
        write_xf(data / f"{bn}.xf", A, np.full((n, 2), 1.5))
        (data / "align.com").write_text("$tiltalign\nRotationAngle\t84.0\n")
        (data / "tilt.com").write_text("$tilt\nTHICKNESS 40\nIMAGEBINNED 1\n")
        s = tmp / "s.toml"
        s.write_text(f'''
[project]
basename = "{bn}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/out"
[conversion]
initial_conditions = ["raw_xf_affine_fixed"]
[ctf]
mode = "off"
[multiresolution]
extra_projection_binning = 1
[missalignment]
refinement_mode = "standard"
''')
        return data, s, bn

    def test_prepare_stages_real_xf_for_raw_xf_affine_fixed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = self._project(tmp)
            init = subprocess.run([PY, str(PREPARE), "init", str(s)], env=ENV, text=True, capture_output=True)
            self.assertEqual(init.returncode, 0, init.stdout + init.stderr)
            resolved = tmp / "out" / "project_settings.toml"
            prep = subprocess.run([PY, str(PREPARE), "prepare", str(resolved)],
                                  env=ENV, text=True, capture_output=True)
            self.assertEqual(prep.returncode, 0, prep.stdout[-2000:] + prep.stderr[-2000:])
            # locate the warp staging manifest (cluster-only path: warpylib absent locally)
            mans = list((tmp / "out").rglob("warp_staging_manifest.json"))
            self.assertTrue(mans, "no warp_staging_manifest.json written")
            man = json.loads(mans[0].read_text())
            self.assertEqual(man["condition"], "raw_xf_affine_fixed")
            self.assertEqual(man["warp_alignment_mode"], "full-affine")
            self.assertEqual(man["axis_frame"], "aligned")
            self.assertFalse(man["is_identity"])                       # NOT identity
            self.assertTrue(str(man["staged_xf"]).endswith(".xf"))     # the REAL .xf
            self.assertTrue(str(man["input_stack"]).endswith(".mrc"))
            self.assertNotIn("_ali.mrc", str(man["input_stack"]))      # RAW, not aligned
            self.assertAlmostEqual(man["tilt_axis_angle_deg"], 84.0, places=3)


if __name__ == "__main__":
    unittest.main()
