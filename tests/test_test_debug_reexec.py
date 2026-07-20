"""--test-debug must re-exec under the MissAlignment env's Python when the current
interpreter lacks numpy, or fail with a CLEAR message — never a raw ModuleNotFoundError."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup_missalign_project as SMP


class ReexecTests(unittest.TestCase):
    def test_no_reexec_when_numpy_present(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=object()):
            with mock.patch("os.execv") as ex:
                SMP._maybe_reexec_under_env(["--test-debug"])  # returns, no exec
                ex.assert_not_called()

    def test_reexec_under_env_python_when_numpy_missing(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "env"; (env / "bin").mkdir(parents=True)
            py = env / "bin" / "python"; py.write_text("#!/bin/sh\n"); py.chmod(0o755)
            argv = ["--test-debug", "--missalign-env", str(env), "--data-dir", "/x"]
            with mock.patch.object(importlib.util, "find_spec", return_value=None):
                with mock.patch.dict("os.environ", {}, clear=False):
                    with mock.patch("os.execv") as ex:
                        SMP._maybe_reexec_under_env(argv)
                        ex.assert_called_once()
                        called_py, called_args = ex.call_args[0]
                        self.assertEqual(called_py, str(py))
                        self.assertIn("--missalign-env", called_args)
                        self.assertIn(str(env), called_args)

    def test_clear_error_when_no_env_python(self):
        argv = ["--test-debug", "--missalign-env", "/nonexistent/env", "--data-dir", "/x"]
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            with mock.patch.dict("os.environ", {}, clear=False):
                with mock.patch("os.execv") as ex:
                    import io
                    from contextlib import redirect_stdout
                    buf = io.StringIO()
                    with self.assertRaises(SystemExit) as cm, redirect_stdout(buf):
                        SMP._maybe_reexec_under_env(argv)
                    ex.assert_not_called()
                    self.assertEqual(cm.exception.code, 2)
                    out = buf.getvalue()
                    self.assertIn("numpy", out)
                    self.assertIn("Activate", out)   # actionable, not a traceback

    def test_guard_prevents_reexec_loop(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            with mock.patch.dict("os.environ", {"MISSALIGN_TESTDEBUG_REEXEC": "1"}):
                with mock.patch("os.execv") as ex:
                    with self.assertRaises(SystemExit) as cm:
                        SMP._maybe_reexec_under_env(["--test-debug"])
                    self.assertEqual(cm.exception.code, 2)
                    ex.assert_not_called()


if __name__ == "__main__":
    unittest.main()
