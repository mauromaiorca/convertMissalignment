"""--test-debug diagnostic harness: unit tests (pure helpers, honest states, mutation
detection) + an end-to-end run gated on IMOD. Built on the canonical system; no
duplicate discovery/conversion."""
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
from pipeline import discovery as DISC
from pipeline import project_config as PC
from pipeline import test_debug as TD

try:
    import mrcfile
    HAVE = True
except Exception:
    HAVE = False
NEWSTACK = shutil.which("newstack")
TILT = shutil.which("tilt")
SETUP = ROOT / "setup_missalign_project.py"
PY = sys.executable
ENV = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD"),
       "PATH": "/Applications/IMOD/bin:" + os.environ.get("PATH", "")}


def _project(tmp: Path, *, dirname="lam6_2_ts_004_bin8_Imod", bn="lam6_2_ts_004_bin8", n=21,
             raw=(500, 600), ali=(250, 300), raw_pix=1.363, ali_pix=2.726, axis=84.0,
             extra=()):
    data = tmp / dirname; data.mkdir(parents=True)
    with mrcfile.new(data / f"{bn}.mrc", overwrite=True) as h:
        h.set_data(np.random.rand(n, raw[1], raw[0]).astype(np.float32)); h.voxel_size = raw_pix
    with mrcfile.new(data / f"{bn}_ali.mrc", overwrite=True) as h:
        h.set_data(np.random.rand(n, ali[1], ali[0]).astype(np.float32)); h.voxel_size = ali_pix
    ang = np.linspace(-60, 60, n)
    (data / f"{bn}.tlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    (data / f"{bn}.rawtlt").write_text("\n".join(f"{a:.2f}" for a in ang) + "\n")
    write_xf(data / f"{bn}.xf", np.stack([np.eye(2)] * n), np.zeros((n, 2)))
    (data / f"{bn}.xtilt").write_text("\n".join("0.0" for _ in range(n)) + "\n")
    (data / f"{bn}.tltxf").write_text("\n".join("1 0 0 1 0 0" for _ in range(n)) + "\n")
    (data / "align.com").write_text(f"$tiltalign\nRotationAngle\t{axis}\n")
    (data / "tilt.com").write_text("$tilt\nTHICKNESS 80\nIMAGEBINNED 1\n")
    for f in extra:
        (data / f).write_text("x\n")
    return data, bn


# --------------------------------------------------------------------------- #
# pure-helper unit tests (no IMOD needed)
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_tilt_selection_includes_zero_and_spans(self):
        angles = list(np.linspace(-60, 60, 41))
        idx = TD._select_tilts(angles, 9)
        self.assertEqual(len(idx), 9)
        self.assertEqual(idx, sorted(idx))                       # preserve order
        chosen = [angles[i] for i in idx]
        self.assertLess(min(abs(a) for a in chosen), 3.5)        # includes near-zero
        self.assertLess(chosen[0], -40); self.assertGreater(chosen[-1], 40)  # spans range
        self.assertTrue(any(a < 0 for a in chosen) and any(a > 0 for a in chosen))

    def test_tilt_selection_small_returns_all(self):
        self.assertEqual(TD._select_tilts([-2, 0, 2], 9), [0, 1, 2])

    def test_xf_decompose(self):
        th = np.deg2rad(10)
        A = 1.05 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dec = TD._xf_decompose(A, [3.0, -4.0])
        self.assertAlmostEqual(dec["rotation_deg"], 10, places=3)
        self.assertAlmostEqual(dec["isotropic_scale"], 1.05, places=3)
        self.assertAlmostEqual(dec["translation_norm"], 5.0, places=3)
        self.assertFalse(dec["reflection"])
        self.assertTrue(TD._xf_decompose(np.array([[-1, 0], [0, 1]]), [0, 0])["reflection"])

    def test_redaction(self):
        os.environ["MY_DEBUG_SECRET_TOKEN"] = "supersecret"
        try:
            r = TD._redact_env()
            self.assertEqual(r["MY_DEBUG_SECRET_TOKEN"], "<redacted>")
        finally:
            del os.environ["MY_DEBUG_SECRET_TOKEN"]

    def test_volume_invariant_and_doubled_detection(self):
        # mutation: wrong physical volume / doubled (raw shape x output pixel)
        target_shape, target_pix = [2046, 494, 2880], 2.726
        good = [s * target_pix for s in target_shape]
        doubled = [s * 2 * target_pix for s in target_shape]
        self.assertTrue(PC.volume_invariant_ok(good, target_shape, target_pix))
        self.assertFalse(PC.volume_invariant_ok(doubled, target_shape, target_pix))


@unittest.skipUnless(HAVE, "mrcfile unavailable")
class BasenameTests(unittest.TestCase):
    def test_exact_name_basename_inference(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td))
            inferred, rep = DISC.infer_basename(data)
            self.assertEqual(inferred, bn)                       # exact-name beats _Imod suffix

    def test_ambiguous_fails_with_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "proj"; d.mkdir()
            # two equally-good distinct projects -> ambiguous
            for bn in ("alpha", "beta"):
                with mrcfile.new(d / f"{bn}.mrc", overwrite=True) as h:
                    h.set_data(np.zeros((3, 8, 8), np.float32))
                with mrcfile.new(d / f"{bn}_ali.mrc", overwrite=True) as h:
                    h.set_data(np.zeros((3, 8, 8), np.float32))
                (d / f"{bn}.tlt").write_text("0\n1\n2\n")
                write_xf(d / f"{bn}.xf", np.stack([np.eye(2)] * 3), np.zeros((3, 2)))
            with self.assertRaises(DISC.DiscoveryError) as cm:
                DISC.infer_basename(d)
            self.assertIn("ambiguous", str(cm.exception))
            self.assertIn("--basename", str(cm.exception))


@unittest.skipUnless(HAVE, "mrcfile unavailable")
class NoRediscoveryTests(unittest.TestCase):
    def test_stages_use_resolved_config_not_discovery(self):
        # After canonical_config, monkeypatch discovery to RAISE; downstream stages must
        # still complete (they consume rc.sources, not discovery).
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td))
            opts = TD.DebugOptions(data_dir=str(data), out_dir=str(Path(td) / "foo"), basename=bn)
            lay = TD.DebugLayout(Path(td) / "run").create()
            cfg = TD.stage_canonical_config(opts, lay)
            self.assertTrue(cfg.ok())
            rc = cfg.data["config"]
            orig = DISC.discover_sources
            DISC.discover_sources = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("rediscovery after init!"))
            try:
                r = TD.stage_statistics(opts, lay, rc)   # must not call discovery
            finally:
                DISC.discover_sources = orig
            self.assertIn(r.state, (TD.PASS, TD.WARNING, TD.NOT_RUN_DEP))


@unittest.skipUnless(HAVE and NEWSTACK and TILT, "IMOD newstack/tilt unavailable")
class EndToEndTests(unittest.TestCase):
    def _run(self, data, out, extra=()):
        return subprocess.run([PY, str(SETUP), "--data-dir", str(data), "--test-debug",
                               "--out-dir", str(out), *extra], env=ENV, text=True, capture_output=True)

    def _latest(self, out):
        runs = list((out / "test_debug").glob("lam6_2_ts_004_bin8_*"))
        return [r for r in runs if r.is_dir()][0]

    def test_acceptance_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td))
            before = {p: p.stat().st_mtime_ns for p in data.iterdir()}
            out = Path(td) / "foo"
            cp = self._run(data, out)
            self.assertEqual(cp.returncode, 0, cp.stdout[-1500:] + cp.stderr[-1500:])
            run = self._latest(out)
            for rel in ("results/DEBUG_SUMMARY.json", "results/DEBUG_SUMMARY.md",
                        "config/project_settings.resolved.toml", "jobs/submit_debug.sh",
                        "source_inventory/source_inventory.json"):
                self.assertTrue((run / rel).is_file(), rel)
            self.assertFalse(list((run / "previews").glob("*.png")))  # quick mode skips image statistics/previews
            self.assertTrue(list((run / "bundle").glob("*_debug_bundle.tar.gz")))
            self.assertTrue((out / "test_debug" / "LATEST_DEBUG_RUN").is_file())
            # source unchanged (read-only)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in data.iterdir()})
            summ = json.loads((run / "results" / "DEBUG_SUMMARY.json").read_text())
            self.assertFalse(summ["source_changed"])
            self.assertEqual(summ["counts"]["FAIL"], 0, summ["failures"])
            # honest: warp/missalign not claimed verified
            states = {s["name"]: s["state"] for s in summ["stages"]}
            self.assertEqual(states["warp_diagnostics"], TD.NOT_RUN_DEP)  # warpylib absent

    def test_no_auto_submit_and_no_smoke_flag(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out).returncode, 0)
            run = self._latest(out)
            smoke = (run / "jobs" / "debug_missalignment_smoke.sbatch").read_text()
            self.assertNotRegex(smoke, r"miss-alignment[^\n]*--smoke")  # no flag on the command
            self.assertIn("--config-file", smoke)
            self.assertIn("alignment: global", (run / "missalignment" / "config.smoke.yaml").read_text())
            # nothing submitted (no sbatch invoked): submit script exists but is manual
            self.assertIn("MANUAL", (run / "jobs" / "submit_debug.sh").read_text())

    def test_fixtures_and_section_correspondence(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out, ["--debug-tilt-count", "7", "--test-debug-full"]).returncode, 0)
            run = self._latest(out)
            sel = (run / "fixtures" / "compute_smoke" / "selected_tilts.tsv").read_text().splitlines()
            self.assertEqual(len(sel) - 1, 7)                     # 7 selected
            man = json.loads((run / "fixtures" / "compute_smoke" / "fixture_manifest.json").read_text())
            # every subset has 7 rows; reduced stack has 7 sections
            for role, v in man["subsets"].items():
                if "rows" in v:
                    self.assertEqual(v["rows"], 7, role)
                if "header" in v:
                    self.assertEqual(v["header"]["n_sections"], 7, role)
            # geometry fixture <= 512; smoke <= 256 (measured, not integer-division)
            gman = json.loads((run / "fixtures" / "geometry_all_tilts" / "fixture_manifest.json").read_text())
            for role, v in gman["stacks"].items():
                self.assertLessEqual(max(v["fixture_header"]["shape_xy"]), 512, role)

    def test_xtilt_tltxf_separate_in_resolved_config(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out).returncode, 0)
            run = self._latest(out)
            rc = PC.load(run / "config" / "project_settings.resolved.toml")
            self.assertTrue(rc.sources.xtilt_file and rc.sources.xtilt_file.endswith(".xtilt"))
            self.assertTrue(rc.sources.tltxf_file and rc.sources.tltxf_file.endswith(".tltxf"))
            self.assertNotEqual(rc.sources.xtilt_file, rc.sources.tltxf_file)

    def test_non_divisible_dims_measured(self):
        # 500/3 etc. are not integer; the harness must MEASURE the newstack output.
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td), raw=(514, 614), ali=(257, 307)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out, ["--test-debug-full"]).returncode, 0)
            run = self._latest(out)
            gman = json.loads((run / "fixtures" / "geometry_all_tilts" / "fixture_manifest.json").read_text())
            # output dims come from the real header, not 514 // factor
            self.assertIn("raw_stack", gman["stacks"])
            self.assertLessEqual(max(gman["stacks"]["raw_stack"]["fixture_header"]["shape_xy"]), 512)

    def test_zero_tilt_axis_refused(self):
        # mutation: zero tilt axis -> init refuses (no silent 0.0)
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td))
            (data / "align.com").unlink()                        # remove the axis source
            out = Path(td) / "foo"
            cp = self._run(data, out)
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("0.0", cp.stdout + cp.stderr)

    def test_collect_after_run(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out).returncode, 0)
            run = self._latest(out)
            cp = subprocess.run([PY, str(SETUP), "--test-debug-collect", "--debug-run", str(run)],
                                env=ENV, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertTrue((run / "diagnostics" / "collect_report.json").is_file())

    def test_bundle_excludes_full_stacks_and_redacts(self):
        with tempfile.TemporaryDirectory() as td:
            data, bn = _project(Path(td)); out = Path(td) / "foo"
            self.assertEqual(self._run(data, out).returncode, 0)
            run = self._latest(out)
            bundle = next((run / "bundle").glob("*_debug_bundle.tar.gz"))
            with tarfile.open(bundle) as tar:
                names = tar.getnames()
            # the FULL source stacks must never be in the bundle
            self.assertFalse(any(n.endswith(f"{bn}.mrc") for n in names))
            self.assertTrue(any("DEBUG_SUMMARY.json" in n for n in names))
            self.assertTrue(any("shareable_paths_redacted.json" in n for n in names))
            shareable = json.loads((run / "bundle" / "shareable_paths_redacted.json").read_text())
            self.assertTrue(all("<SOURCE>" not in v or str(data) not in v for v in shareable.values()))


class CommandRunnerTests(unittest.TestCase):
    """§16/§17: external commands are bounded (per-command + global timeout, process-group
    kill) and observable (START/heartbeat/END), output never deadlocks on a full pipe."""

    def test_success_captures_output(self):
        r = TD.run_external_command([sys.executable, "-c", "print('hi'); print('boom', file=__import__('sys').stderr)"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("hi", r.stdout)
        self.assertIn("boom", r.stderr)
        self.assertFalse(r.timed_out)

    def test_nonzero_returncode(self):
        r = TD.run_external_command([sys.executable, "-c", "raise SystemExit(3)"])
        self.assertEqual(r.returncode, 3)
        self.assertFalse(r.timed_out)

    def test_timeout_kills_process_group(self):
        # a child that ignores SIGTERM and sleeps must still be killed via the group SIGKILL,
        # and the whole call must return within a small multiple of the 1s timeout.
        prog = ("import signal,time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(60)\n")
        t0 = TD.time.monotonic()
        r = TD.run_external_command([sys.executable, "-c", prog], timeout_s=1, heartbeat_s=0.3)
        dur = TD.time.monotonic() - t0
        self.assertTrue(r.timed_out)
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(dur, 15.0)   # killed promptly, not 60s

    def test_global_deadline_caps_command(self):
        # an already-past global deadline shrinks the effective timeout to ~0 -> immediate kill.
        prog = "import time; time.sleep(30)"
        r = TD.run_external_command([sys.executable, "-c", prog], timeout_s=600,
                                    deadline=TD.time.monotonic() - 1.0, heartbeat_s=0.3)
        self.assertTrue(r.timed_out)

    def test_missing_binary_does_not_raise(self):
        r = TD.run_external_command(["this_binary_does_not_exist_xyz"])
        self.assertEqual(r.returncode, 127)
        self.assertFalse(r.timed_out)


class SampledHashTests(unittest.TestCase):
    """§17: orchestrate._hash_file uses an exact hash for small files and a bounded sampled
    fingerprint for large ones (never streams a multi-GB stack to detect staleness)."""

    def setUp(self):
        from pipeline import orchestrate as ORCH
        self.ORCH = ORCH

    def test_small_file_exact_hash_stable_and_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "small.bin"
            p.write_bytes(b"abc" * 1000)
            h1 = self.ORCH._hash_file(p)
            self.assertFalse(h1.startswith("s:"))
            self.assertEqual(h1, self.ORCH._hash_file(p))      # stable
            p.write_bytes(b"abd" * 1000)
            self.assertNotEqual(h1, self.ORCH._hash_file(p))   # content-sensitive

    def test_large_file_sampled_bounded_and_sensitive(self):
        big = self.ORCH._FULL_HASH_MAX_BYTES + (4 << 20)       # just over the threshold
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.bin"
            with p.open("wb") as fh:
                fh.write(b"\x00" * big)
            h1 = self.ORCH._hash_file(p)
            self.assertTrue(h1.startswith("s:"))
            self.assertEqual(h1, self.ORCH._hash_file(p))      # deterministic
            # change in the HEAD window must flip the fingerprint
            with p.open("r+b") as fh:
                fh.seek(0); fh.write(b"\xff" * 4096)
            self.assertNotEqual(h1, self.ORCH._hash_file(p))
            # a size change alone must flip the fingerprint (size is folded in)
            with p.open("ab") as fh:
                fh.write(b"\x00" * (1 << 20))
            self.assertNotEqual(h1, self.ORCH._hash_file(p))

    def test_missing_file_is_none(self):
        self.assertIsNone(self.ORCH._hash_file(Path("/no/such/file/xyz.bin")))


if __name__ == "__main__":
    unittest.main()
