#!/usr/bin/env python3
"""Half-set splitting for half-map reconstructions (angle- or index-based)."""
from __future__ import annotations

from pathlib import Path


def split_halfsets(tilt_angles, *, mode: str = "angle") -> dict:
    """Return {'even': [idx...], 'odd': [idx...]} index lists for the two halves.

    angle: sort by tilt angle, alternate. index: alternate by acquisition index.
    Both halves preserve the full angular range as much as possible.
    """
    n = len(tilt_angles)
    if mode == "angle":
        order = sorted(range(n), key=lambda i: float(tilt_angles[i]))
    elif mode == "index":
        order = list(range(n))
    else:
        raise ValueError(f"unknown half split mode {mode!r}")
    even = [order[i] for i in range(0, n, 2)]
    odd = [order[i] for i in range(1, n, 2)]
    return {"even": even, "odd": odd, "mode": mode}


def write_half_tilt_files(out_dir: Path, basename: str, tilt_angles, halves: dict) -> dict:
    """Write per-half .tlt files; return their paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for half in ("even", "odd"):
        idx = halves[half]
        p = out_dir / f"{basename}_{half}.tlt"
        p.write_text("\n".join(f"{float(tilt_angles[i]):.2f}" for i in idx) + "\n")
        paths[half] = str(p)
    return paths
