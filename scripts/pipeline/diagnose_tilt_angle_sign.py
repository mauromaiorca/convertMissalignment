#!/usr/bin/env python3
"""Compare an IMOD ``clip rotx`` reference with the Warp sign +1 and sign -1 reconstructions.

Establishes whether the IMOD->Warp tilt-angle sign -1 agrees in geometric handedness with the
IMOD reconstruction convention (via ``clip rotx``, which rotates IMOD's Y-thickness native
volume into the Z-thickness/handedness the comparison needs). It does NOT claim physical
specimen handedness.

Generation (cluster; WarpTools + IMOD) — keep EVERYTHING identical except the sign:

    # sign +1 candidate is the existing reconstruction.
    # sign -1 candidate: set imod_to_warp_tilt_angle_sign = -1 (the default) and reconstruct;
    # place both under .internal/attempts/reconstruction/<id>/tilt_angle_sign_audit/.
    clip rotx <native-IMOD-reconstruction> <audit-dir>/imod_rotx.mrc

Then compare (this script, needs only numpy + mrcfile):

    python scripts/pipeline/diagnose_tilt_angle_sign.py \
        --imod-rotx <audit-dir>/imod_rotx.mrc \
        --warp-plus1 <existing sign +1 rec.mrc> \
        --warp-minus1 <new sign -1 rec.mrc> \
        --out <audit-dir>/tilt_angle_sign_comparison.json

If sign -1 does NOT agree with clip rotx, DO NOT change BASE_AXIS_PERMUTATION automatically;
report the discrepancy for a separate tilt-axis / volume-frame audit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def normalized_cross_correlation(a, b) -> float:
    """Zero-mean normalised cross-correlation of two volumes (in [-1, 1])."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _read_mrc(path: Path) -> np.ndarray:
    import mrcfile
    with mrcfile.open(str(path), permissive=True) as m:
        return np.asarray(m.data, dtype=np.float32)


def run_clip_rotx(native_reconstruction: Path, out_path: Path, *, clip_executable: str = "clip") -> Path:
    """Create the hand-preserving IMOD Z-thickness reference via ``clip rotx`` (needs IMOD)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [clip_executable, "rotx", str(native_reconstruction), str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def compare(imod_rotx: Path, warp_plus1: Path, warp_minus1: Path) -> dict:
    """Compare each Warp reconstruction against the IMOD clip-rotx reference by NCC."""
    ref = _read_mrc(imod_rotx)
    report = {"reference": str(imod_rotx), "shapes": {"reference": list(ref.shape)}, "ncc": {}}
    for name, path in (("warp_sign_plus1", warp_plus1), ("warp_sign_minus1", warp_minus1)):
        vol = _read_mrc(path)
        report["shapes"][name] = list(vol.shape)
        if vol.shape != ref.shape:
            report["ncc"][name] = None
            report.setdefault("problems", []).append(
                f"{name} shape {vol.shape} != reference {ref.shape}; cannot compare directly")
            continue
        report["ncc"][name] = normalized_cross_correlation(vol, ref)
    p1, m1 = report["ncc"].get("warp_sign_plus1"), report["ncc"].get("warp_sign_minus1")
    if p1 is not None and m1 is not None:
        report["agrees_with_clip_rotx"] = "warp_sign_minus1" if m1 > p1 else "warp_sign_plus1"
        report["minus1_agrees"] = bool(m1 > p1)
    else:
        report["agrees_with_clip_rotx"] = "undetermined"
        report["minus1_agrees"] = None
    report["note"] = (
        "Agreement is with the IMOD reconstruction convention (clip rotx), NOT a claim of "
        "physical specimen handedness. If sign -1 does not agree, do not change "
        "BASE_AXIS_PERMUTATION; report for a separate tilt-axis / volume-frame audit.")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare Warp sign +1/-1 reconstructions to IMOD clip rotx.")
    p.add_argument("--imod-rotx", type=Path, required=True, help="clip-rotx reference MRC")
    p.add_argument("--warp-plus1", type=Path, required=True, help="Warp sign +1 reconstruction MRC")
    p.add_argument("--warp-minus1", type=Path, required=True, help="Warp sign -1 reconstruction MRC")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--native-reconstruction", type=Path, default=None,
                   help="if given, run `clip rotx` on it first to produce --imod-rotx")
    p.add_argument("--clip-executable", default="clip")
    args = p.parse_args(argv)

    if args.native_reconstruction is not None:
        run_clip_rotx(args.native_reconstruction, args.imod_rotx, clip_executable=args.clip_executable)

    report = compare(args.imod_rotx, args.warp_plus1, args.warp_minus1)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
