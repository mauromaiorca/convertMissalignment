#!/usr/bin/env python3
"""Create angle-balanced IMOD half-set parameter files for tilt-series reconstruction.

The script does not touch image data. It subsets small IMOD parameter files:
  - .tlt tilt angles
  - .xf alignment transforms
  - optional .xtilt X-axis tilt file

It also writes a manifest with a SectionsToRead string for newstack, so the
original raw stack can be read by absolute path without copying or symlinking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def read_nonempty_lines(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def read_angles(path: Path) -> List[float]:
    angles: List[float] = []
    for lineno, line in enumerate(read_nonempty_lines(path), 1):
        try:
            angles.append(float(line.split()[0]))
        except Exception as exc:
            raise SystemExit(f"ERROR: cannot parse tilt angle in {path}:{lineno}: {line!r}") from exc
    return angles


def write_selected_lines(in_path: Path, out_path: Path, indices: Iterable[int]) -> None:
    lines = read_nonempty_lines(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines[i] for i in indices) + "\n")


def comma_list(indices: Iterable[int]) -> str:
    # Explicit comma-separated list is safest for alternating half-sets.
    return ",".join(str(i) for i in indices)


def summarize_angles(label: str, indices: List[int], angles: List[float]) -> str:
    vals = [angles[i] for i in indices]
    if not vals:
        return f"{label}: n=0"
    return (
        f"{label}: n={len(vals)}, "
        f"angle_min={min(vals):.3f}, angle_max={max(vals):.3f}, "
        f"angle_mean={sum(vals)/len(vals):.3f}, sections={comma_list(indices)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Create angle/index balanced IMOD half-set parameter files.")
    p.add_argument("--tlt", required=True, type=Path, help="Input full .tlt file")
    p.add_argument("--xf", required=True, type=Path, help="Input full .xf file")
    p.add_argument("--xtilt", type=Path, default=None, help="Optional full .xtilt file")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for half_a/half_b parameter files")
    p.add_argument("--basename", required=True, help="Dataset basename, e.g. lam8_ts_004")
    p.add_argument("--mode", choices=["angle", "index"], default="angle",
                   help="angle: sort by angle and alternate half assignment; index: original even/odd index split")
    args = p.parse_args()

    for path in (args.tlt, args.xf):
        if not path.exists():
            raise SystemExit(f"ERROR: missing file: {path}")
    if args.xtilt is not None and not args.xtilt.exists():
        raise SystemExit(f"ERROR: missing xtilt file: {args.xtilt}")

    angles = read_angles(args.tlt)
    xf_lines = read_nonempty_lines(args.xf)
    if len(angles) != len(xf_lines):
        raise SystemExit(f"ERROR: .tlt has {len(angles)} rows but .xf has {len(xf_lines)} rows")
    if args.xtilt is not None:
        xtilt_lines = read_nonempty_lines(args.xtilt)
        if len(xtilt_lines) != len(angles):
            raise SystemExit(f"ERROR: .xtilt has {len(xtilt_lines)} rows but .tlt has {len(angles)} rows")

    n = len(angles)
    all_indices = list(range(n))

    if args.mode == "angle":
        sorted_by_angle = sorted(all_indices, key=lambda i: (angles[i], i))
        half_a = sorted(sorted_by_angle[0::2])
        half_b = sorted(sorted_by_angle[1::2])
    else:
        half_a = all_indices[0::2]
        half_b = all_indices[1::2]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "halfset_manifest.tsv"
    summary = args.out_dir / "halfset_summary.txt"

    rows = []
    for label, indices in (("half_a", half_a), ("half_b", half_b)):
        d = args.out_dir / label
        d.mkdir(parents=True, exist_ok=True)
        write_selected_lines(args.tlt, d / f"{args.basename}.tlt", indices)
        write_selected_lines(args.xf, d / f"{args.basename}.xf", indices)
        xtilt_out = ""
        if args.xtilt is not None:
            write_selected_lines(args.xtilt, d / f"{args.basename}.xtilt", indices)
            xtilt_out = str(d / f"{args.basename}.xtilt")
        rows.append((label, comma_list(indices), str(d / f"{args.basename}.tlt"), str(d / f"{args.basename}.xf"), xtilt_out))

    with manifest.open("w") as f:
        f.write("label\tsections_to_read\ttlt\txf\txtilt\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    summary.write_text(
        "IMOD half-set split\n"
        f"mode: {args.mode}\n"
        f"n_tilts: {n}\n"
        + summarize_angles("half_a", half_a, angles) + "\n"
        + summarize_angles("half_b", half_b, angles) + "\n"
        + "\nNote: section indices are 0-based for newstack SectionsToRead.\n"
    )

    print(f"Wrote half-set manifest: {manifest}")
    print(summary.read_text().rstrip())


if __name__ == "__main__":
    main()
