"""Regression: generated run_missalignment.sh must be safe for paths with spaces.

Defect (Phase 2, C4): ``03_run_missalignment.py`` interpolated ``warp_dir`` into
a ``rm -rf {warp_dir}/iter* ...`` line WITHOUT shell quoting, while the adjacent
``tee`` redirect quoted its path.  On a project path containing a space the
``rm -rf`` would word-split and could delete unintended paths.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts" / "03_run_missalignment.py"


class RunScriptQuotingTests(unittest.TestCase):
    def test_clean_rm_is_space_safe(self):
        condition = "raw_xf_affine_fixed"
        with tempfile.TemporaryDirectory() as td:
            # Deliberately put a space in the path.
            parent = Path(td) / "project with space" / "warp_parent"
            warp_dir = parent / f"warp_{condition}"
            (warp_dir / "tiltstack" / "TS_01").mkdir(parents=True)
            (warp_dir / "series.xml").write_text("<xml/>\n")
            (warp_dir / "tiltstack" / "TS_01" / "TS_01.st").write_text("stack\n")
            params = Path(td) / "params.json"
            params.write_text(json.dumps({"series_name": "series"}))

            cmd = [
                sys.executable, str(RUN),
                "--params", str(params),
                "--warp-parent", str(parent),
                "--conditions", condition,
                "--clean",  # write the cleaning line; no --run so nothing executes
            ]
            cp = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, f"{cp.stdout}\n{cp.stderr}")

            script = (parent / "run_missalignment.sh").read_text()
            rm_lines = [ln for ln in script.splitlines() if ln.strip().startswith("rm -rf")]
            self.assertEqual(len(rm_lines), 1, script)
            rm_line = rm_lines[0]

            # The script resolves --warp-parent, so compare against the resolved
            # path (on macOS /var -> /private/var).
            resolved = str(warp_dir.resolve())
            quoted = shlex.quote(resolved)

            # The quoted directory prefix must be present, and the bare
            # space-containing path must NOT appear unquoted.
            self.assertIn(f"{quoted}/iter*", rm_line)
            self.assertNotIn(f"rm -rf {resolved}/iter*", rm_line)

            # Strongest check: tokenising the command the way a shell would must
            # keep the directory+glob together, never split on the space.
            tokens = shlex.split(rm_line)
            self.assertEqual(tokens[0], "rm")
            self.assertEqual(tokens[1], "-rf")
            self.assertEqual(len(tokens), 7, f"expected 5 targets, got: {tokens}")
            self.assertTrue(all(resolved in tok for tok in tokens[2:]),
                            f"path word-split in: {tokens}")


if __name__ == "__main__":
    unittest.main()
