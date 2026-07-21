"""Quantitative block-boundary (seam) diagnostic for tiled Warp reconstructions.

Measures the voxel discontinuity across the planes where reconstruction blocks meet
(multiples of ``subvolume_size``) relative to nearby non-boundary planes, per axis.

Volume convention: an MRC volume read as a numpy array has shape ``(nz, ny, nx)``,
i.e. axis 0 = Z, axis 1 = Y, axis 2 = X. A block boundary perpendicular to an axis is
a plane at ``k * subvolume_size`` along that axis. Viewing-orientation labels follow
the "missing axis" convention:

    XY view  <- boundaries perpendicular to Z  (axis 0)
    XZ view  <- boundaries perpendicular to Y  (axis 1)
    YZ view  <- boundaries perpendicular to X  (axis 2)

The metric is independent of display contrast: it is the mean absolute voxel
difference across boundary planes divided by the same quantity at nearby control
planes. A ratio near 1 means no visible seam; a ratio >> 1 means a strong seam.
The tool does NOT assume the artefact is equally visible in all orientations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# axis index (numpy z,y,x) -> viewing-orientation label (missing-axis convention)
_AXIS_TO_VIEW = {0: "XY", 1: "XZ", 2: "YZ"}


def _plane_pair_diff(volume, index: int, axis: int) -> float:
    import numpy as np

    a = np.take(volume, index, axis=axis).astype("float64")
    b = np.take(volume, index - 1, axis=axis).astype("float64")
    return float(np.mean(np.abs(a - b)))


def _axis_seam(volume, axis: int, subvolume_size: int) -> dict:
    n = volume.shape[axis]
    boundaries = [p for p in range(subvolume_size, n, subvolume_size) if 1 <= p < n]
    # control planes: mid-block positions that are not themselves boundaries
    controls = []
    for p in boundaries:
        c = p + max(1, subvolume_size // 2)
        if c < n and (c % subvolume_size) != 0:
            controls.append(c)
    if not controls:
        controls = [p for p in range(2, n) if (p % subvolume_size) != 0][: max(1, len(boundaries))]

    boundary_diff = (
        sum(_plane_pair_diff(volume, p, axis) for p in boundaries) / len(boundaries)
        if boundaries else 0.0
    )
    control_diff = (
        sum(_plane_pair_diff(volume, c, axis) for c in controls) / len(controls)
        if controls else 0.0
    )
    ratio = (boundary_diff / control_diff) if control_diff > 0 else (
        float("inf") if boundary_diff > 0 else 1.0)
    return {
        "boundary_difference": boundary_diff,
        "local_control_difference": control_diff,
        "boundary_to_control_ratio": ratio,
        "number_of_boundaries": len(boundaries),
    }


def seam_metric(volume, subvolume_size: int) -> dict:
    """Per-orientation boundary-to-control seam metric for a 3-D volume (z,y,x)."""
    if subvolume_size <= 0:
        raise ValueError("subvolume_size must be positive")
    per_view = {}
    for axis, view in _AXIS_TO_VIEW.items():
        per_view[view] = _axis_seam(volume, axis, subvolume_size)
    xy = per_view["XY"]["boundary_to_control_ratio"]
    xz = per_view["XZ"]["boundary_to_control_ratio"]
    return {
        "subvolume_size": int(subvolume_size),
        "orientations": per_view,
        "xz_boundary_ratio": xz,
        "xy_boundary_ratio": xy,
        "xz_ratio_higher_than_xy": bool(xz > xy),
    }


def _read_mrc(path: Path):
    import mrcfile

    with mrcfile.open(str(path), permissive=True) as handle:
        return handle.data.copy()


def _human_report(result: dict) -> str:
    lines = [f"seam diagnostic (subvolume_size={result['subvolume_size']})"]
    for view, m in result["orientations"].items():
        lines.append(
            f"  {view}: ratio={m['boundary_to_control_ratio']:.3f}  "
            f"boundary={m['boundary_difference']:.4g}  control={m['local_control_difference']:.4g}  "
            f"n={m['number_of_boundaries']}")
    lines.append(f"  XZ ratio higher than XY: {result['xz_ratio_higher_than_xy']}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="seam_diagnostic",
        description="Measure block-boundary seam contrast in XY/XZ/YZ for a tiled reconstruction.")
    p.add_argument("volume", type=Path, help="reconstruction MRC")
    p.add_argument("--subvolume-size", type=int, default=64)
    p.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--report", type=Path, default=None, help="write the human report here")
    args = p.parse_args(argv)

    volume = _read_mrc(args.volume)
    result = seam_metric(volume, args.subvolume_size)
    result["volume"] = str(args.volume)
    text = _human_report(result)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    if args.report:
        args.report.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
