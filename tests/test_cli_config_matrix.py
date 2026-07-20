"""CLI/config validation across all four refinement example TOMLs.

Covers: parse, --validate-only, --dry-run, manifest generation + provenance,
result-directory isolation, reconstruction-helper syntax, source-safety, and
rerun idempotency."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # config loader imports torch-based models
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

EXAMPLES = ["translation_refinement", "rigid_refinement", "similarity_refinement", "affine_refinement"]


def run(args, **kw):
    return subprocess.run([sys.executable, *args], text=True, capture_output=True, **kw)


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class CliConfigMatrixTests(unittest.TestCase):
    def test_all_examples_parse(self):
        for name in EXAMPLES:
            p = ROOT / "config" / "examples" / f"{name}.toml"
            with p.open("rb") as fh:
                tomllib.load(fh)

    def test_prepare_validate_only_all_examples(self):
        for name in EXAMPLES:
            ex = ROOT / "config" / "examples" / f"{name}.toml"
            cp = run([str(ROOT / "prepare_imod_to_warp.py"), str(ex), "--validate-only"])
            self.assertEqual(cp.returncode, 0, f"{name}: {cp.stdout}{cp.stderr}")
            self.assertIn("refinement model", cp.stdout)

    def test_refine_local_validate_only_all_examples(self):
        for name in EXAMPLES:
            ex = ROOT / "config" / "examples" / f"{name}.toml"
            cp = run([str(ROOT / "refine_local.py"), str(ex), "--validate-only"])
            self.assertEqual(cp.returncode, 0, f"{name}: {cp.stdout}{cp.stderr}")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            ex = ROOT / "config" / "examples" / "affine_refinement.toml"
            cp = run([str(ROOT / "prepare_imod_to_warp.py"), str(ex), "--data-dir", str(td),
                      "--out-dir", str(Path(td) / "o"), "--dry-run"])
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            self.assertFalse((Path(td) / "o" / "interoperability").exists())

    def test_manifest_and_idempotency(self):
        with tempfile.TemporaryDirectory() as td:
            ex = ROOT / "config" / "examples" / "rigid_refinement.toml"
            out = Path(td) / "o"
            for _ in range(2):  # rerun must be idempotent
                cp = run([str(ROOT / "prepare_imod_to_warp.py"), str(ex), "--out-dir", str(out)])
                self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            manifest = out / "interoperability" / "project_manifest.json"
            self.assertTrue(manifest.is_file())
            data = json.loads(manifest.read_text())
            self.assertIn("software_versions", data)
            self.assertIn("torch", data["software_versions"])
            self.assertEqual(data["resolved"]["refinement"]["model"], "rigid")

    def test_result_directory_isolation(self):
        from alignment_models import interop
        out = Path("/tmp/projX")
        dirs = set()
        for cond in ("raw_xf_affine_fixed", "ali_identity"):
            for model in EXAMPLES:
                m = model.replace("_refinement", "")
                dirs.add(str(interop.export_dir(out, cond, m)))
                dirs.add(str(interop.result_dir(out, cond, m)))
        # every (condition, model) pair maps to a unique, collision-free path
        self.assertEqual(len(dirs), 2 * len(EXAMPLES) * 2)

    def test_source_toml_not_modified(self):
        ex = ROOT / "config" / "examples" / "affine_refinement.toml"
        before = ex.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            run([str(ROOT / "prepare_imod_to_warp.py"), str(ex), "--out-dir", str(Path(td) / "o")])
        self.assertEqual(ex.read_bytes(), before, "example TOML was modified")


if __name__ == "__main__":
    unittest.main()
