"""runlog.py (structured logging, command capture, postmortem, debug bundle) and
discovery.py (deterministic scored source discovery). Pure-Python, no IMOD/torch."""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import discovery as D
from pipeline import runlog as R


class RunLogTests(unittest.TestCase):
    def test_events_jsonl_and_fields(self):
        with tempfile.TemporaryDirectory() as td:
            rl = R.RunLogger(Path(td), run_id="rid1", phase="prepare")
            rl.log_event(step="P01", event="start", status="info", message="hi", data={"k": 1})
            lines = (Path(td) / "logs" / "events.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            for k in ("timestamp_utc", "run_id", "phase", "step", "event", "status",
                      "hostname", "pid", "cwd", "message", "data"):
                self.assertIn(k, rec)
            self.assertEqual(rec["run_id"], "rid1")
            self.assertEqual(rec["data"], {"k": 1})

    def test_command_capture_full_streams(self):
        with tempfile.TemporaryDirectory() as td:
            rl = R.RunLogger(Path(td), run_id="rid", phase="prepare")
            big = "x" * 5000
            res = rl.run_command(["python3", "-c", f"print('{big}'); import sys; sys.stderr.write('err'*100)"],
                                 step="echo_big")
            self.assertEqual(res["return_code"], 0)
            out = Path(res["stdout_path"]).read_text()
            self.assertIn(big, out)                       # FULL stdout, not truncated
            self.assertEqual(json.loads(Path(td, "logs/commands/echo_big.result.json").read_text())["return_code"], 0)

    def test_command_failure_check_raises_with_full_log(self):
        with tempfile.TemporaryDirectory() as td:
            rl = R.RunLogger(Path(td), run_id="rid", phase="prepare")
            with self.assertRaises(R.CommandError):
                rl.run_command(["python3", "-c", "import sys; sys.exit(3)"], step="boom", check=True)
            res = json.loads(Path(td, "logs/commands/boom.result.json").read_text())
            self.assertEqual(res["return_code"], 3)

    def test_missing_executable_recorded_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            rl = R.RunLogger(Path(td), run_id="rid", phase="prepare")
            res = rl.run_command(["this_exe_does_not_exist_xyz"], step="missing")
            self.assertEqual(res["return_code"], 127)
            self.assertEqual(res["signal"], "ENOENT")

    def test_env_redaction(self):
        env = {"PATH": "/bin", "MY_SECRET_TOKEN": "abc", "API_KEY": "xyz", "HOME": "/h"}
        red = R.redact_env(env)
        self.assertEqual(red["MY_SECRET_TOKEN"], "<redacted>")
        self.assertEqual(red["API_KEY"], "<redacted>")
        self.assertEqual(red["PATH"], "/bin")

    def test_postmortem_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            rl = R.RunLogger(Path(td), run_id="rid", phase="prepare")
            rl.log_event(step="P05", event="x", status="ok")
            try:
                raise ValueError("kaboom")
            except ValueError as exc:
                fpath = rl.write_postmortem(exc, step="P05", expected_outputs=[Path(td) / "missing.mrc"])
            pm = Path(td) / "diagnostics" / "postmortem"
            for f in ("failure.json", "traceback.txt", "last_events.jsonl",
                      "filesystem_snapshot.txt", "environment_summary.txt"):
                self.assertTrue((pm / f).is_file(), f)
            failure = json.loads(fpath.read_text())
            self.assertEqual(failure["exception_class"], "ValueError")
            self.assertEqual(failure["expected_outputs"][0]["exists"], False)

    def test_debug_bundle_excludes_images_and_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            rl = R.RunLogger(run, run_id="bid", phase="prepare")
            rl.log_event(step="P01", event="x")
            (run / "manifests").mkdir(exist_ok=True)
            (run / "manifests" / "prepare_manifest.json").write_text('{"ok":true}')
            (run / "working").mkdir(exist_ok=True)
            (run / "working" / "big.mrc").write_bytes(b"\x00" * 1000)  # image -> excluded
            bundle = R.collect_debug_bundle(run, "bid")
            self.assertTrue(bundle.is_file())
            with tarfile.open(bundle) as tar:
                names = tar.getnames()
            self.assertTrue(any("events.jsonl" in n for n in names))
            self.assertTrue(any("prepare_manifest.json" in n for n in names))
            self.assertFalse(any(n.endswith(".mrc") for n in names), "image leaked into bundle")


class DiscoveryTests(unittest.TestCase):
    def _project(self, td, basename="64x_Vero_02"):
        d = Path(td) / "data"; d.mkdir()
        files = [f"{basename}.mrc", f"{basename}_ali.mrc", f"{basename}.xf", f"{basename}.tlt",
                 f"{basename}.rawtlt", f"{basename}.xtilt", f"{basename}.defocus",
                 f"{basename}.mrc.mdoc", "newst.com", "tilt.com", "ctfcorrection.com",
                 f"{basename}_full_rec.mrc",
                 # decoys that must NOT be selected as final_xf:
                 f"{basename}.prexf", f"{basename}_fid.xf", "rotation.xf", f"{basename}.tltxf"]
        for f in files:
            (d / f).write_text("x\n")
        return d

    def test_discovers_all_types_and_rejects_decoys(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._project(td)
            inv = D.discover_sources(d, "64x_Vero_02")
            self.assertTrue(inv.raw_stack.endswith("64x_Vero_02.mrc"))
            self.assertTrue(inv.aligned_stack.endswith("64x_Vero_02_ali.mrc"))
            self.assertTrue(inv.final_xf.endswith("64x_Vero_02.xf"))
            self.assertNotIn("prexf", inv.final_xf)
            self.assertNotIn("_fid", inv.final_xf)
            for f in ("tilt_file", "raw_tilt_file", "xtilt_file", "defocus_file",
                      "mdoc_file", "newst_com", "tilt_com", "ctf_com", "source_reconstruction"):
                self.assertIsNotNone(getattr(inv, f), f)
            self.assertTrue(inv.tilt_com.endswith("tilt.com"))
            self.assertTrue(inv.source_reconstruction.endswith("_full_rec.mrc"))

    def test_raw_not_confused_with_aligned_or_rec(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._project(td)
            inv = D.discover_sources(d, "64x_Vero_02")
            self.assertNotIn("_ali", Path(inv.raw_stack).name)
            self.assertNotIn("_rec", Path(inv.raw_stack).name)

    def test_explicit_override_wins_and_must_exist(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._project(td)
            chosen = d / "custom_ali.mrc"; chosen.write_text("x")
            inv = D.discover_sources(d, "64x_Vero_02", overrides={"aligned_stack": str(chosen)})
            self.assertEqual(inv.aligned_stack, str(chosen))
            with self.assertRaises(D.DiscoveryError):
                D.discover_sources(d, "64x_Vero_02", overrides={"aligned_stack": str(d / "nope.mrc")})

    def test_ambiguous_fails(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "data"; d.mkdir()
            # two equally-scored exact-name defocus candidates in different subdirs
            (d / "a").mkdir(); (d / "b").mkdir()
            (d / "a" / "64x_Vero_02.defocus").write_text("x")
            (d / "b" / "64x_Vero_02.defocus").write_text("x")
            with self.assertRaises(D.DiscoveryError):
                D.discover_sources(d, "64x_Vero_02")

    def test_etomo_backup_com_dirs_deprioritized(self):
        # real cluster case: newst.com/tilt.com exist in the project root AND in eTomo
        # backup/template dirs (dfltcoms/origcoms). The ROOT file must win, not error.
        with tempfile.TemporaryDirectory() as td:
            d = self._project(td)
            for sub in ("dfltcoms", "origcoms"):
                (d / sub).mkdir()
                (d / sub / "newst.com").write_text("x\n")
                (d / sub / "tilt.com").write_text("x\n")
            inv = D.discover_sources(d, "64x_Vero_02")
            self.assertEqual(Path(inv.newst_com).parent, d)      # root, not a backup dir
            self.assertEqual(Path(inv.tilt_com).parent, d)
            self.assertNotIn("dfltcoms", inv.newst_com)
            self.assertNotIn("origcoms", inv.tilt_com)

    def test_root_file_beats_equally_named_nested(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._project(td)
            (d / "sub").mkdir()
            (d / "sub" / "64x_Vero_02.defocus").write_text("x\n")
            (d / "64x_Vero_02.defocus").write_text("x\n")
            inv = D.discover_sources(d, "64x_Vero_02")
            self.assertEqual(Path(inv.defocus_file).parent, d)   # root preferred over sub/

    def test_section_consistency(self):
        inv = D.SourceInventory(basename="b", data_dir="/x", raw_stack="r.mrc",
                                aligned_stack="a.mrc", final_xf="f.xf", tilt_file="t.tlt")
        ok = D.check_section_consistency(inv, measure_mrc=lambda p: 7, count_lines=lambda p: 7)
        self.assertTrue(ok["consistent"])
        with self.assertRaises(D.DiscoveryError):
            D.check_section_consistency(inv, measure_mrc=lambda p: 7,
                                       count_lines=lambda p: 6 if p.endswith(".tlt") else 7)


if __name__ == "__main__":
    unittest.main()
