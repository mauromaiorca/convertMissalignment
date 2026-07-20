"""Canonical three-phase CLI (Phase 1 verbs) end to end with real IMOD.
Skipped (not faked) without IMOD/mrcfile."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
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
NEWSTACK = shutil.which("newstack")
PREPARE = ROOT / "prepare_imod_to_warp.py"
PY = sys.executable
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _project(tmp: Path, bn="64x_Vero_02"):
    data = tmp / "data"; data.mkdir(); n = 7
    ali = np.random.rand(n, 256, 256).astype(np.float32)
    for nm in (f"{bn}_ali.mrc", f"{bn}.mrc"):
        with mrcfile.new(data / nm, overwrite=True) as h:
            h.set_data(ali); h.voxel_size = 1.36
    ang = np.linspace(-60, 60, n)
    (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    (data / f"{bn}.rawtlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{bn}.xf", np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    (data / "ctfcorrection.com").write_text(
        "$ctfphaseflip -StandardInput\nVoltage 300\nSphericalAberration 2.7\n"
        "AmplitudeContrast 0.07\nInterpolationWidth 20\n")
    s = tmp / "s.toml"
    s.write_text(f'''
[project]
basename = "{bn}"
[paths]
data_root = "{data.as_posix()}"
output_dir = "{tmp.as_posix()}/runs"
[input]
aligned_stack = "{(data / f'{bn}_ali.mrc').as_posix()}"
raw_stack = "{(data / f'{bn}.mrc').as_posix()}"
final_xf_file = "{(data / f'{bn}.xf').as_posix()}"
final_tilt_file = "{(data / f'{bn}.tlt').as_posix()}"
[geometry]
raw_dimensions_xyz = [256, 256, 1]
raw_pixel_size_A = 1.36
[conversion]
initial_conditions = ["ali_identity"]
[multiresolution]
enabled = true
extra_projection_binning = 8
[ctf]
mode = "off"
[missalignment]
refinement_mode = "standard"
''')
    return data, s, bn


def _run(args, env=None):
    return subprocess.run([PY, str(PREPARE), *args], env=env or ENV, text=True, capture_output=True)


class HelpTests(unittest.TestCase):
    def test_subcommands_in_help(self):
        cp = _run(["--help"])
        for verb in ("validate", "prepare", "status", "collect-debug"):
            self.assertIn(verb, cp.stdout, verb)

    def test_each_verb_help(self):
        for verb in ("validate", "prepare", "status", "collect-debug"):
            cp = _run([verb, "--help"])
            self.assertEqual(cp.returncode, 0, verb)
            self.assertIn("settings", cp.stdout.lower())


@unittest.skipUnless(HAVE and NEWSTACK, "IMOD/mrcfile unavailable")
class PhaseOneCliTests(unittest.TestCase):
    def test_validate(self):
        with tempfile.TemporaryDirectory() as td:
            _, s, _ = _project(Path(td))
            cp = _run(["validate", str(s)])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            self.assertIn("configuration OK", cp.stdout)
            self.assertIn("warpylib", cp.stdout)        # capability table shown
            self.assertIn("cluster_only", cp.stdout)    # honest about warpylib/sbatch

    def test_prepare_builds_run_dir_jobs_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, bn = _project(tmp)
            before = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            cp = _run(["prepare", str(s), "--allow-unresolved-legacy"])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            rd = tmp / "runs" / "64x_Vero_02_ali_identity_standard"
            # canonical layout dirs
            for sub in ("manifests", "logs", "diagnostics", "jobs", "final",
                        "missalignment", "warp", "working"):
                self.assertTrue((rd / sub).is_dir(), sub)
            # manifests
            prep = json.loads((rd / "manifests" / "prepare_manifest.json").read_text())
            self.assertIn("run_id", prep)
            self.assertTrue((rd / "manifests" / "source_inventory.json").is_file())
            self.assertTrue((rd / "manifests" / "source_hashes.json").is_file())
            self.assertTrue((rd / "manifests" / "job_graph.json").is_file())
            # jobs generated
            for j in ("phase2.sbatch", "phase3_pre_missalign_reconstruct.sbatch",
                      "phase3_smoke_reconstruct.sbatch",
                      "phase3_full_finalize_and_reconstruct.sbatch"):
                self.assertTrue((rd / "jobs" / j).is_file(), j)
            # events logged
            self.assertTrue((rd / "logs" / "events.jsonl").is_file())
            # source untouched
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in data.iterdir()})

    def test_status_and_collect_debug(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _, s, _ = _project(tmp)
            self.assertEqual(_run(["prepare", str(s), "--allow-unresolved-legacy"]).returncode, 0)
            st = _run(["status", str(s)])
            self.assertEqual(st.returncode, 0, st.stderr)
            self.assertIn("prepare_manifest.json", st.stdout)
            self.assertIn("present", st.stdout)
            db = _run(["collect-debug", str(s)])
            self.assertEqual(db.returncode, 0, db.stderr)
            bundle = next((tmp / "runs" / "64x_Vero_02_ali_identity_standard").glob("debug_bundle_*.tar.gz"))
            with tarfile.open(bundle) as tar:
                names = tar.getnames()
            self.assertTrue(any("events.jsonl" in n for n in names))
            self.assertTrue(any("prepare_manifest.json" in n for n in names))
            self.assertFalse(any(n.endswith(".mrc") for n in names))

    def test_prepare_defers_final_ctf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data, s, _ = _project(tmp)
            # ctf.mode=final defers all CTF to Phase 3 (working CTF not needed)
            txt = s.read_text().replace('mode = "off"', 'mode = "final"')
            s2 = tmp / "s2.toml"; s2.write_text(txt)
            cp = _run(["prepare", str(s2), "--allow-unresolved-legacy"])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            rd = tmp / "runs" / "64x_Vero_02_ali_identity_standard"
            prep = json.loads((rd / "manifests" / "prepare_manifest.json").read_text())
            self.assertTrue(prep["final_ctf_deferred_to_phase3"])


if __name__ == "__main__":
    unittest.main()
