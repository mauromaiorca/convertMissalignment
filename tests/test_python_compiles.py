"""Regression: every repository .py file must byte-compile on the running Python.

Defect (Phase: quality checks): ``setup_imod_recon.py`` used a backslash inside
an f-string expression (``f"...{'\\n'.join(...)}..."``), which is a SyntaxError
on Python 3.11 (PEP 701 only permitted it from 3.12).  The project documents
Python 3.11+ as supported, so the file failed to import on the stated minimum.

This guard compiles every tracked .py file under whatever interpreter runs the
suite, so a syntax-level regression on 3.11 (e.g. on Maxwell) is caught.
"""
from __future__ import annotations

import py_compile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PythonCompilesTests(unittest.TestCase):
    def test_all_repository_python_files_compile(self):
        failures = []
        for path in sorted(ROOT.rglob("*.py")):
            # Skip throwaway audit scratch and any cached bytecode dirs.
            if "_audit_artifacts" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:  # pragma: no cover - failure path
                failures.append(f"{path}: {exc.msg}")
        self.assertEqual(failures, [], "files failed to compile:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
