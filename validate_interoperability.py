#!/usr/bin/env python3
"""Run interoperability validation at a chosen level.

Levels
------
- ``math``      : pure affine + constrained-model tests (numpy/torch only).
- ``imod``      : real IMOD ``newstack`` tests (centre convention, .ali
  generation, exact constrained export). Requires ``newstack`` on PATH.
- ``warp``      : real warpylib coordinate tests. Requires ``warpylib``; if it
  is unavailable this is reported as UNAVAILABLE (NOT VERIFIED), never faked.
- ``roundtrip`` : raw/ali composition equivalence (local) + the warpylib
  ``.xf -> XML -> .xf`` round trip (Maxwell).
- ``all``       : every level; unavailable levels are reported explicitly.

Exit status is non-zero only when a *runnable required* check FAILS. An
UNAVAILABLE level (missing warpylib/GPU) is reported but does not, by itself,
fail -- it is recorded as pending Maxwell validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"

LEVEL_MODULES = {
    "math": [
        "test_affine_math", "test_grid_encoding", "test_independent_oracle",
        "test_alignment_models", "test_model_gradients", "test_model_recovery",
        "test_constraints_scopes_reg", "test_adversarial", "test_raw_ali_equivalence",
        "test_interop_export",
    ],
    "imod": [
        "test_imod_center_convention", "test_ali_generation_real_imod",
        "test_constrained_export_real_imod", "test_newst_com_parser",
    ],
    "warp": [
        "test_axis_convention_warp", "test_warpylib_roundtrip", "test_warp_xml_roundtrip",
    ],
    "roundtrip": [
        "test_raw_ali_equivalence", "test_warp_xml_roundtrip",
    ],
    "multiresolution": [
        "test_multires_grids", "test_multires_projection", "test_multires_workflow",
        "test_multires_newstack_real", "test_multires_working_stacks_real",
        "test_multires_restore_real_imod",
    ],
}


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def run_modules(modules: list[str]) -> tuple[int, int, int, int]:
    loader = unittest.TestLoader()
    sys.path.insert(0, str(TESTS))
    suite = unittest.TestSuite()
    for m in modules:
        try:
            suite.addTests(loader.loadTestsFromName(m))
        except Exception as exc:  # pragma: no cover
            print(f"  WARNING: could not load {m}: {exc}")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ran = result.testsRun
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    return ran, ran - failed - skipped, failed, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", nargs="?", default=None, help="Project settings TOML (optional; reserved for data-specific checks)")
    ap.add_argument("--level", choices=["math", "imod", "warp", "roundtrip", "multiresolution", "all"], default="all")
    args = ap.parse_args()

    levels = ["math", "imod", "warp", "roundtrip", "multiresolution"] if args.level == "all" else [args.level]
    newstack = shutil.which("newstack")
    have_warp = have("warpylib")
    have_torch = have("torch")

    overall_fail = 0
    print("=== Interoperability validation ===")
    print(f"torch={'yes' if have_torch else 'NO'}  newstack={'yes' if newstack else 'NO'}  warpylib={'yes' if have_warp else 'NO (Maxwell-pending)'}")
    for level in levels:
        print(f"\n--- level: {level} ---")
        if level == "warp" and not have_warp:
            print("  UNAVAILABLE: warpylib is not installed. Warp coordinate convention, "
                  "offset/movement signs, and tilt-axis convention are NOT VERIFIED locally. "
                  "Run on Maxwell (see CLUSTER_VALIDATION_PLAN.md).")
            continue
        if level == "math" and not have_torch:
            print("  UNAVAILABLE: torch is not installed; constrained-model math cannot run.")
            overall_fail = 1
            continue
        if level == "imod" and not newstack:
            print("  UNAVAILABLE: newstack not on PATH (set IMOD_DIR and PATH). IMOD-backed checks skipped.")
            continue
        modules = LEVEL_MODULES[level]
        # filter warp-only modules out of roundtrip when warpylib missing
        if level == "roundtrip" and not have_warp:
            modules = [m for m in modules if m != "test_warp_xml_roundtrip"]
            print("  NOTE: warpylib .xf->XML->.xf round trip is Maxwell-pending; running local raw/ali equivalence only.")
        ran, passed, failed, skipped = run_modules(modules)
        print(f"  {level}: ran={ran} passed={passed} failed={failed} skipped={skipped}")
        if failed:
            overall_fail = 1

    print("\n=== summary ===")
    print("Required runnable checks:", "FAIL" if overall_fail else "PASS")
    if not have_warp:
        print("Warp/warpylib + GPU checks remain NOT VERIFIED (Maxwell) -- see CLUSTER_VALIDATION_PLAN.md.")
    return overall_fail


if __name__ == "__main__":
    raise SystemExit(main())
