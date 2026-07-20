#!/usr/bin/env python3
"""Version 8 stack-import entry point.

The validated stack-only/legacy-affine path delegates to the production
MissAlignment setup. Movie ingestion and native WarpTools import remain
experimental and are deliberately blocked rather than generating fake jobs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up an operational stack-only Warp/MissAlignment project.")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--basename", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--source-mode", choices=("auto", "movies", "tilt_stack"), default="auto")
    ap.add_argument("--condition", default="raw_xf_affine_fixed")
    ap.add_argument("--alignment-backend", choices=("legacy_affine", "warptools_native"), default="legacy_affine")
    ap.add_argument("--missalign-env", default=None)
    ap.add_argument("--no-prepare", action="store_true")
    ap.add_argument("--local-warp-convert", action="store_true")
    args = ap.parse_args()

    if args.source_mode == "movies":
        print("ERROR: movie ingestion is not operational in this release; no job was generated.", file=sys.stderr)
        return 2
    if args.alignment_backend != "legacy_affine":
        print("ERROR: warptools_native alignment import is not operational in this release; use legacy_affine.", file=sys.stderr)
        return 2

    setup = Path(__file__).resolve().parent / "setup_missalign_project.py"
    cmd = [
        sys.executable, str(setup),
        "--data-dir", str(args.data_dir),
        "--basename", args.basename,
        "--out-dir", str(args.out_dir),
        "--condition", args.condition,
    ]
    if args.missalign_env:
        cmd.extend(("--missalign-env", args.missalign_env))
    if args.no_prepare:
        cmd.append("--no-prepare")
    if args.local_warp_convert:
        cmd.append("--local-warp-convert")
    print("[v8] using the stack-only IMOD-to-Warp pipeline", flush=True)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
