#!/usr/bin/env python3
"""Build a small, transferable eTomo fixture from a real dataset (spec §36).

Intended to run LATER on the cluster against the real dataset. It strongly reduces
the raw + aligned stacks (real IMOD ``newstack -shrink``), copies the metadata/command
files, rewrites absolute GPFS paths out of the command files, measures the output
headers, and archives everything with a manifest. It NEVER modifies the source and
preserves the tilt count.

Local use: point it at a synthetic project to validate the mechanics; the real
portable fixture is produced on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pipeline.discovery import discover_sources  # noqa: E402


def _measure(path, mrcfile):
    with mrcfile.open(path, permissive=True) as h:
        d = h.data
        nsec = int(d.shape[0]) if d.ndim == 3 else 1
        ny, nx = (int(d.shape[-2]), int(d.shape[-1]))
        return {"shape_xy": [nx, ny], "n_sections": nsec, "pixel_A": float(h.voxel_size.x)}


def _sha(path, limit=8 << 20):
    h = hashlib.sha256(); p = Path(path); size = p.stat().st_size
    with p.open("rb") as fh:
        h.update(fh.read(limit))
    return {"size": size, "sha256_head": h.hexdigest()[:24]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--basename", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shrink", type=int, default=8)
    ap.add_argument("--strip-prefix", default="/gpfs", help="rewrite this absolute prefix out of .com files")
    args = ap.parse_args()
    try:
        import mrcfile
    except Exception:
        print("ERROR: mrcfile required"); return 2
    newstack = shutil.which("newstack")
    if not newstack:
        print("ERROR: IMOD newstack required (run on a host with IMOD)"); return 2

    inv = discover_sources(args.data_dir, args.basename)
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}
    manifest = {"basename": args.basename, "source_dir": str(args.data_dir),
                "shrink": args.shrink, "files": {}, "commands": []}

    # reduce stacks (never touch source)
    for role, suffix in (("raw_stack", "_raw_binF.mrc"), ("aligned_stack", "_ali_binF.mrc")):
        src = getattr(inv, role)
        if not src:
            continue
        before = {p: p.stat().st_mtime_ns for p in args.data_dir.iterdir() if p.is_file()}
        dst = out / (args.basename + suffix.replace("F", str(args.shrink)))
        cmd = [newstack, "-input", str(src), "-output", str(dst), "-shrink", str(float(args.shrink)), "-float", "0"]
        cp = subprocess.run(cmd, env=env, text=True, capture_output=True)
        manifest["commands"].append({"cmd": cmd, "rc": cp.returncode})
        if cp.returncode != 0 or not dst.is_file():
            print(f"ERROR: newstack failed for {role}: {cp.stderr[-200:]}"); return 1
        manifest["files"][role] = {"path": str(dst), "header": _measure(dst, mrcfile), **_sha(dst)}
        after = {p: p.stat().st_mtime_ns for p in args.data_dir.iterdir() if p.is_file()}
        if before != after:
            print("ERROR: source directory changed during fixture build"); return 1

    # copy metadata + command files; strip absolute prefixes from .com
    for role in ("final_xf", "tilt_file", "raw_tilt_file", "xtilt_file", "defocus_file",
                 "mdoc_file", "newst_com", "tilt_com", "ctf_com"):
        src = getattr(inv, role)
        if not src:
            continue
        dst = out / Path(src).name
        text = Path(src).read_text(errors="replace")
        if dst.suffix == ".com":
            text = "\n".join(ln for ln in text.splitlines()).replace(args.strip_prefix, "<DATA>")
            dst.write_text(text + "\n")
        else:
            shutil.copy2(src, dst)
        manifest["files"][role] = {"path": str(dst), **_sha(dst)}

    # tilt-count preservation check
    counts = {k: v["header"]["n_sections"] for k, v in manifest["files"].items() if "header" in v}
    manifest["tilt_count_preserved"] = len(set(counts.values())) <= 1
    (out / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    archive = out.parent / f"{args.basename}_portable_fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(f"[fixture] wrote {archive} (tilt_count_preserved={manifest['tilt_count_preserved']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
