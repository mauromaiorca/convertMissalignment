from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_env


class RuntimeEnvTests(unittest.TestCase):
    def test_direct_configured_interpreter_bypasses_module_probe(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "env"
            (env / "bin").mkdir(parents=True)
            py = env / "bin" / "python"
            py.symlink_to(Path(sys.executable).resolve())
            with mock.patch.object(importlib.util, "find_spec", side_effect=AssertionError("find_spec called")):
                with mock.patch("os.execve") as ex:
                    runtime_env.ensure_scientific_python(
                        script=Path(__file__),
                        argv=[],
                        explicit_env=str(env),
                        required=("numpy", "mrcfile"),
                    )
                    ex.assert_not_called()


if __name__ == "__main__":
    unittest.main()
