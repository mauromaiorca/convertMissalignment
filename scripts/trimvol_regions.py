#!/usr/bin/env python3
"""
Crop cubic/square regions from IMOD tomograms with trimvol.

Given one or more center coordinates (X,Y,Z) and a box size, this script
computes trimvol -x/-y/-z start/end ranges and crops the same region from
full, half_a, and half_b reconstructions for one or more alignment cases.

Default coordinate convention is IMOD/3dmod coordinates, i.e. 1-based.
Use --index-coordinates if your centers are 0-based array indices.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


class Center(NamedTuple):
    x: int
    y: int
    z: int


class AxisRange(NamedTuple):
    start: int
    end: int
    shifted: bool


HEAVY_EXTS = {".mrc", ".rec", ".st", ".map"}


def parse_center(text: str) -> Center:
    """Parse centers supplied as 'x,y,z', '(x,y,z)', or 'x:y:z'."""
    cleaned = text.strip().strip("()[]{}")
    parts = re.split(r"[,;:\s]+", cleaned)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Center must have three values x,y,z; got: {text!r}")
    try:
        return Center(*(int(round(float(p))) for p in parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric center: {text!r}") from exc


def read_mrc_dimensions(path: Path) -> tuple[int, int, int]:
    """Read NX, NY, NZ from the MRC header using only Python stdlib."""
    with path.open("rb") as f:
        header = f.read(12)
    if len(header) != 12:
        raise ValueError(f"File too small to be MRC-like: {path}")

    nx, ny, nz = struct.unpack("<3i", header)
    if 0 < nx < 1_000_000 and 0 < ny < 1_000_000 and 0 < nz < 1_000_000:
        return nx, ny, nz

    # Fallback for a rare big-endian header.
    nx, ny, nz = struct.unpack(">3i", header)
    if 0 < nx < 1_000_000 and 0 < ny < 1_000_000 and 0 < nz < 1_000_000:
        return nx, ny, nz

    raise ValueError(f"Could not parse plausible MRC dimensions from: {path}")


def centered_range(center: int, box: int, dim: int, index_coordinates: bool, fit_mode: str) -> AxisRange:
    """
    Return an inclusive start/end range for trimvol.

    For even box sizes there is no single central voxel; this convention places
    the supplied center just left/lower of the geometric center, e.g. box=364 gives
    [center-181, center+182] in 1-based coordinates.
    """
    if box <= 0:
        raise ValueError("box size must be positive")
    if box > dim:
        raise ValueError(f"box size {box} is larger than dimension {dim}")

    lo = 0 if index_coordinates else 1
    hi = dim - 1 if index_coordinates else dim

    start = center - ((box - 1) // 2)
    end = start + box - 1
    shifted = False

    if fit_mode == "strict":
        if start < lo or end > hi:
            raise ValueError(
                f"requested range {start}:{end} is outside bounds {lo}:{hi}; "
                "use --fit-mode shift to keep box size and move it inside the volume"
            )
        return AxisRange(start, end, shifted)

    if fit_mode != "shift":
        raise ValueError(f"unknown fit mode: {fit_mode}")

    if start < lo:
        delta = lo - start
        start += delta
        end += delta
        shifted = True
    if end > hi:
        delta = end - hi
        start -= delta
        end -= delta
        shifted = True
    if start < lo or end > hi:
        raise ValueError(f"cannot fit range of size {box} within bounds {lo}:{hi}")

    return AxisRange(start, end, shifted)


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def default_input_path(recon_root: Path, case: str, subset: str, dataset: str, volume_name: str) -> Path:
    return recon_root / case / subset / dataset / volume_name


def make_output_path(out_root: Path, region_id: str, case: str, subset: str, dataset: str, box: int) -> Path:
    filename = f"{dataset}_{safe_label(region_id)}_{case}_{subset}_box{box}.mrc"
    return out_root / region_id / case / subset / filename


def run_trimvol(
    trimvol: str,
    input_path: Path,
    output_path: Path,
    xr: AxisRange,
    yr: AxisRange,
    zr: AxisRange,
    mode: int | None,
    index_coordinates: bool,
    dry_run: bool,
) -> list[str]:
    cmd = [trimvol]
    if index_coordinates:
        cmd.append("-i")
    cmd += [
        "-x", str(xr.start), str(xr.end),
        "-y", str(yr.start), str(yr.end),
        "-z", str(zr.start), str(zr.end),
    ]
    if mode is not None:
        cmd += ["-mode", str(mode)]
    cmd += [str(input_path), str(output_path)]

    print("Running:", " ".join(cmd), flush=True)
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(
        description="Crop matching full/half tomogram regions with IMOD trimvol."
    )
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--recon-root", type=Path, default=None,
                   help="Default: PROJECT/imod_recon")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Default: RECON_ROOT/_crops")
    p.add_argument("--dataset", default="lam8_ts_004")
    p.add_argument("--volume-name", default="lam8_ts_004.rec")
    p.add_argument("--case", dest="cases", action="append", choices=["original", "missalign_raw_xf"],
                   help="Case to crop. Repeatable. Default: both.")
    p.add_argument("--subset", dest="subsets", action="append", choices=["full", "half_a", "half_b"],
                   help="Subset to crop. Repeatable. Default: full, half_a, half_b.")
    p.add_argument("--center", dest="centers", action="append", type=parse_center, required=True,
                   help="Region center as x,y,z. Repeat for multiple regions.")
    p.add_argument("--box-size", type=int, required=True)
    p.add_argument("--index-coordinates", action="store_true",
                   help="Use trimvol -i and interpret coordinates as 0-based indices. Default is 1-based IMOD/3dmod coordinates.")
    p.add_argument("--fit-mode", choices=["shift", "strict"], default="shift",
                   help="shift keeps box size and moves boxes inside volume if near an edge; strict errors out instead. Default: shift.")
    p.add_argument("--mode", type=int, default=2,
                   help="trimvol output mode. Default 2=float. Use --mode -1 to omit.")
    p.add_argument("--trimvol", default="trimvol")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    recon_root = args.recon_root or (args.project / "imod_recon")
    out_root = args.out_root or (recon_root / "_crops")
    cases = args.cases or ["original", "missalign_raw_xf"]
    subsets = args.subsets or ["full", "half_a", "half_b"]
    mode = None if args.mode == -1 else args.mode

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / f"trimvol_crops_box{args.box_size}.tsv"

    n_done = 0
    n_missing = 0
    n_failed = 0

    with manifest_path.open("w", newline="") as mf:
        writer = csv.writer(mf, delimiter="\t")
        writer.writerow([
            "region_id", "center_x", "center_y", "center_z", "box_size",
            "case", "subset", "input", "output", "nx", "ny", "nz",
            "x_start", "x_end", "y_start", "y_end", "z_start", "z_end",
            "coordinate_mode", "fit_mode", "shifted_to_fit", "status",
        ])

        for i, center in enumerate(args.centers, start=1):
            region_id = f"region_{i:03d}_x{center.x}_y{center.y}_z{center.z}_box{args.box_size}"
            for case in cases:
                for subset in subsets:
                    input_path = default_input_path(recon_root, case, subset, args.dataset, args.volume_name)
                    output_path = make_output_path(out_root, region_id, case, subset, args.dataset, args.box_size)

                    if not input_path.exists():
                        msg = "missing_input"
                        print(f"WARNING: missing input, skipping: {input_path}", file=sys.stderr)
                        writer.writerow([
                            region_id, center.x, center.y, center.z, args.box_size,
                            case, subset, str(input_path), str(output_path), "", "", "",
                            "", "", "", "", "", "",
                            "index" if args.index_coordinates else "3dmod_1based",
                            args.fit_mode, "", msg,
                        ])
                        n_missing += 1
                        if not args.skip_missing:
                            return 2
                        continue

                    try:
                        nx, ny, nz = read_mrc_dimensions(input_path)
                        xr = centered_range(center.x, args.box_size, nx, args.index_coordinates, args.fit_mode)
                        yr = centered_range(center.y, args.box_size, ny, args.index_coordinates, args.fit_mode)
                        zr = centered_range(center.z, args.box_size, nz, args.index_coordinates, args.fit_mode)
                        shifted = xr.shifted or yr.shifted or zr.shifted
                        cmd = run_trimvol(
                            args.trimvol, input_path, output_path,
                            xr, yr, zr, mode, args.index_coordinates, args.dry_run,
                        )
                        status = "dry_run" if args.dry_run else "done"
                        n_done += 1
                    except Exception as exc:
                        print(f"ERROR for {case}/{subset} {region_id}: {exc}", file=sys.stderr)
                        nx = ny = nz = ""
                        xr = yr = zr = AxisRange("", "", False)  # type: ignore[arg-type]
                        shifted = ""
                        status = f"failed: {exc}"
                        n_failed += 1
                        if not args.skip_missing:
                            return 3

                    writer.writerow([
                        region_id, center.x, center.y, center.z, args.box_size,
                        case, subset, str(input_path), str(output_path), nx, ny, nz,
                        xr.start, xr.end, yr.start, yr.end, zr.start, zr.end,
                        "index" if args.index_coordinates else "3dmod_1based",
                        args.fit_mode, shifted, status,
                    ])

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Output root: {out_root}")
    print(f"Done crops: {n_done}; missing inputs: {n_missing}; failed: {n_failed}")
    return 0 if n_failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
