#!/usr/bin/env python3
"""Confirm the custom .xf import matches the INSTALLED Warp ``ts_import_alignments``.

Cluster-only (needs WarpTools + warpylib). Stages a set of ``.xf`` rows (in the validated
Warp view order) into an ISOLATED workspace under ``.internal/warp_xf_import_reference/``,
runs the official importer, reads back ``TiltAxisAngles`` / ``TiltAxisOffsetX`` /
``TiltAxisOffsetY`` from the resulting Warp metadata, and compares them against
``imod_affine.imod_xf_row_to_warp_alignment`` for every row. Production metadata is never
touched; only the ``.tlt`` is provided because the importer requires it (it is not compared).

    LC_ALL=C LANG=C WarpTools ts_import_alignments \
        --settings <isolated-settings> --alignments <isolated-alignments> \
        --alignment_angpix <alignment-pixel-size>

Run on the cluster with the project's staged rows, e.g.:

    python scripts/pipeline/warp_xf_import_reference.py \
        --xf <staged>/TS_tomo2_raw_xf_translation.xf \
        --tlt <staged>/TS_tomo2_raw_xf_translation.rawtlt \
        --alignment-angpix 2.2 \
        --workspace <project>/.internal/warp_xf_import_reference \
        --settings <existing Warp settings .settings> \
        --out <workspace>/parity_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import imod_xf_row_to_warp_alignment, read_xf  # noqa: E402

ANGLE_TOL_DEG = 1e-3
OFFSET_TOL_A = 1e-2


def _warptools_env() -> dict:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def _read_back_warp_alignment(xml_path: Path) -> dict:
    """Read TiltAxisAngles / TiltAxisOffsetX / TiltAxisOffsetY from Warp metadata (warpylib)."""
    import warpylib  # cluster only
    from warpylib import TiltSeries
    ts = TiltSeries(path=str(xml_path))
    if hasattr(ts, "load_meta"):
        ts.load_meta(str(xml_path))

    def _list(name):
        v = getattr(ts, name, None)
        if v is None:
            return None
        try:
            return [float(x) for x in v.tolist()]
        except AttributeError:
            return [float(x) for x in v]
    return {
        "tilt_axis_angles": _list("tilt_axis_angles"),
        "tilt_axis_offset_x": _list("tilt_axis_offset_x"),
        "tilt_axis_offset_y": _list("tilt_axis_offset_y"),
    }


def compare_helper_to_readback(xf_rows, alignment_angpix: float, readback: dict) -> dict:
    """Compare the custom helper against read-back official values, row by row."""
    n = len(xf_rows)
    max_angle_res = 0.0
    max_offset_res = 0.0
    per_row = []
    ra = readback.get("tilt_axis_angles") or [None] * n
    rx = readback.get("tilt_axis_offset_x") or [None] * n
    ry = readback.get("tilt_axis_offset_y") or [None] * n
    for i, row in enumerate(xf_rows):
        angle, ox, oy = imod_xf_row_to_warp_alignment(row, alignment_angpix)
        entry = {"row": i, "helper": {"angle": angle, "offset_x": ox, "offset_y": oy},
                 "official": {"angle": ra[i], "offset_x": rx[i], "offset_y": ry[i]}}
        if ra[i] is not None:
            entry["angle_residual_deg"] = abs(angle - ra[i])
            max_angle_res = max(max_angle_res, entry["angle_residual_deg"])
        if rx[i] is not None and ry[i] is not None:
            res = max(abs(ox - rx[i]), abs(oy - ry[i]))
            entry["offset_residual_A"] = res
            max_offset_res = max(max_offset_res, res)
        per_row.append(entry)
    return {
        "n_rows": n,
        "alignment_angpix": float(alignment_angpix),
        "max_angle_residual_deg": max_angle_res,
        "max_offset_residual_A": max_offset_res,
        "angle_within_tol": max_angle_res <= ANGLE_TOL_DEG,
        "offset_within_tol": max_offset_res <= OFFSET_TOL_A,
        "parity": bool(max_angle_res <= ANGLE_TOL_DEG and max_offset_res <= OFFSET_TOL_A),
        "tolerances": {"angle_deg": ANGLE_TOL_DEG, "offset_A": OFFSET_TOL_A},
        "per_row": per_row,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Official ts_import_alignments parity check (cluster).")
    p.add_argument("--xf", type=Path, required=True, help="staged .xf (Warp view order)")
    p.add_argument("--tlt", type=Path, required=True, help=".tlt required by the importer (not compared)")
    p.add_argument("--alignment-angpix", type=float, required=True)
    p.add_argument("--workspace", type=Path, required=True,
                   help="isolated dir, e.g. <project>/.internal/warp_xf_import_reference")
    p.add_argument("--settings", type=Path, default=None, help="existing Warp .settings to reuse")
    p.add_argument("--warptools", default="WarpTools")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-import", action="store_true",
                   help="only compute helper values (no WarpTools); for staging the workspace")
    args = p.parse_args(argv)

    ws = args.workspace
    ws.mkdir(parents=True, exist_ok=True)
    matrices, shifts = read_xf(args.xf)
    xf_rows = [[m[0, 0], m[0, 1], m[1, 0], m[1, 1], s[0], s[1]]
               for m, s in zip(matrices, shifts)]
    # stage isolated copies (never touch production)
    (ws / "alignments.xf").write_text(args.xf.read_text())
    (ws / "alignments.tlt").write_text(args.tlt.read_text())

    readback = {"tilt_axis_angles": None, "tilt_axis_offset_x": None, "tilt_axis_offset_y": None}
    import_status = "skipped"
    if not args.skip_import:
        if args.settings is None:
            print("ERROR: --settings is required to run the official importer (or use --skip-import)")
            return 2
        xml_out = ws / "reference.xml"
        cmd = [args.warptools, "ts_import_alignments",
               "--settings", str(args.settings),
               "--alignments", str(ws / "alignments.xf"),
               "--alignment_angpix", str(args.alignment_angpix)]
        print("[reference] " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=_warptools_env(), cwd=str(ws))
            readback = _read_back_warp_alignment(xml_out if xml_out.is_file() else args.settings)
            import_status = "ran"
        except Exception as exc:  # pragma: no cover - cluster only
            import_status = f"failed: {exc}"
            print(f"[reference] official import failed: {exc}")

    report = compare_helper_to_readback(xf_rows, args.alignment_angpix, readback)
    report["import_status"] = import_status
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if (args.skip_import or report["parity"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
