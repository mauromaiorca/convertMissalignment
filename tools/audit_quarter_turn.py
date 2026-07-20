#!/usr/bin/env python3
"""Report the v7 quarter-turn factorization for an IMOD .xf file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.quarter_turn import factor_affines  # noqa: E402
from imod_affine import read_xf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xf", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    matrices, shifts = read_xf(args.xf)
    factored = factor_affines(matrices, shifts)
    report = factored.to_dict()
    report.update(
        {
            "xf": str(args.xf.resolve()),
            "n_tilts": len(matrices),
            "shift_x_range_px": [float(shifts[:, 0].min()), float(shifts[:, 0].max())],
            "shift_y_range_px": [float(shifts[:, 1].min()), float(shifts[:, 1].max())],
            "determinant_range": [
                float(np.linalg.det(matrices).min()),
                float(np.linalg.det(matrices).max()),
            ],
        }
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"XF: {report['xf']}")
        print(f"tilts: {report['n_tilts']}")
        print(
            "quarter turn: "
            f"np.rot90(k={report['np_rot90_k']}), "
            f"angle={report['quarter_turn_angle_deg']:.3f} deg"
        )
        print(
            "residual rotation: "
            f"median={report['residual_rotation_median_abs_deg']:.3f} deg, "
            f"max={report['residual_rotation_max_abs_deg']:.3f} deg"
        )
        print(f"recomposition error: {report['max_recomposition_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
