#!/usr/bin/env python3
"""Export a Warp/MissAlignment XML geometry to IMOD affine transforms.

The exporter evaluates the complete 2-D Warp sampling map (offsets plus
GridMovementX/Y), fits an affine map for every tilt, reports the non-affine
residual, and writes either:

* a raw->final ``.xf`` for raw-stack conditions, or
* a residual ali->final ``.xf`` plus a composed raw->final ``.xf`` for
  ``ali_identity`` conditions.

An export is refused when the fitted map is not sufficiently affine unless
``--allow-approximate`` is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from warpylib import TiltSeries

from imod_affine import compose_xf, fit_affine, read_xf, residual_statistics, write_xf


def parse_xy(text: str) -> tuple[int, int]:
    parts = [p for p in text.replace("x", ",").replace("X", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y")
    values = tuple(int(p) for p in parts)
    if min(values) <= 0:
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return values


def grid_points_centered_A(shape_xy: tuple[int, int], pixel_size_A: float, nx: int, ny: int) -> np.ndarray:
    dims = np.asarray(shape_xy, dtype=float) * float(pixel_size_A)
    xs = np.linspace(-dims[0] / 2.0, dims[0] / 2.0, nx)
    ys = np.linspace(-dims[1] / 2.0, dims[1] / 2.0, ny)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


def tensor_1d(value: Any, n: int) -> np.ndarray:
    import torch
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy().astype(float, copy=False).reshape(-1)
    else:
        array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 1 and n > 1:
        array = np.repeat(array, n)
    if len(array) != n:
        raise ValueError(f"expected {n} values, got {len(array)}")
    return array


def evaluate_warp_sampling_map(
    ts: "TiltSeries",
    tilt_index: int,
    final_centered_points_A: np.ndarray,
    input_shape_xy: tuple[int, int],
    input_pixel_size_A: float,
) -> np.ndarray:
    """Map final/aligned-centred Å to input-image-centred Å.

    Warp's 2-D order is: add per-tilt offset and image centre, evaluate the
    movement fields at normalized absolute coordinates, then subtract them.
    """
    n_tilts = int(ts.n_tilts)
    offsets_x = tensor_1d(ts.tilt_axis_offset_x, n_tilts)
    offsets_y = tensor_1d(ts.tilt_axis_offset_y, n_tilts)
    offset = np.array([offsets_x[tilt_index], offsets_y[tilt_index]], dtype=float)

    dims_A = np.asarray(input_shape_xy, dtype=float) * float(input_pixel_size_A)
    center_A = dims_A / 2.0
    absolute = final_centered_points_A + center_A + offset
    normalized_xy = absolute / dims_A
    normalized_z = np.full((len(absolute), 1), 0.5 if n_tilts == 1 else tilt_index / (n_tilts - 1), dtype=float)
    import torch
    coords = torch.tensor(np.column_stack([normalized_xy, normalized_z]), dtype=torch.float32)
    with torch.no_grad():
        movement_x = ts.grid_movement_x.get_interpolated(coords).detach().cpu().numpy().reshape(-1)
        movement_y = ts.grid_movement_y.get_interpolated(coords).detach().cpu().numpy().reshape(-1)
    movement = np.column_stack([movement_x, movement_y])
    rounding = np.asarray(ts.size_rounding_factors.detach().cpu().numpy() if hasattr(ts.size_rounding_factors, "detach") else ts.size_rounding_factors, dtype=float).reshape(-1)
    rounding_xy = rounding[:2] if len(rounding) >= 2 else np.ones(2, dtype=float)
    sampled_absolute = (absolute - movement) * rounding_xy
    return sampled_absolute - center_A


def physical_forward_to_pixel_xf(
    inverse_matrix: np.ndarray,
    inverse_shift_A: np.ndarray,
    input_pixel_size_A: float,
    final_pixel_size_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert ``input = B*final+b`` and express input->final in pixel units."""
    forward_physical = np.linalg.inv(inverse_matrix)
    forward_shift_A = -forward_physical @ inverse_shift_A
    matrix_px = (float(input_pixel_size_A) / float(final_pixel_size_A)) * forward_physical
    shift_px = forward_shift_A / float(final_pixel_size_A)
    return matrix_px, shift_px


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--source-frame", choices=("raw", "ali"), required=True)
    parser.add_argument("--input-shape", required=True, type=parse_xy, metavar="X,Y")
    parser.add_argument("--final-shape", type=parse_xy, default=None, metavar="X,Y")
    parser.add_argument("--input-pixel-size", required=True, type=float, help="Pixel size of the stack referenced by the Warp XML.")
    parser.add_argument("--source-xf-pixel-size", type=float, default=None, help="Pixel size of the original source stack to which the exported raw XF will be applied.")
    parser.add_argument("--final-pixel-size", type=float, default=None, help="Desired pixel size of the final aligned IMOD stack.")
    parser.add_argument("--original-raw-pixel-size", type=float, default=None)
    parser.add_argument("--original-ali-pixel-size", type=float, default=None)
    parser.add_argument("--original-xf", type=Path, default=None, help="Required for ali source when a composed raw->final XF is requested.")
    parser.add_argument("--raw-shape", type=parse_xy, default=None, metavar="X,Y")
    parser.add_argument("--out-xf", required=True, type=Path, help="Raw->final XF (raw source) or composed raw->final XF (ali source).")
    parser.add_argument("--out-residual-xf", type=Path, default=None, help="Optional ali->final residual XF.")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-tsv", type=Path, default=None)
    parser.add_argument("--sample-grid", type=int, nargs=2, default=(17, 13), metavar=("NX", "NY"))
    parser.add_argument("--rms-tolerance-px", type=float, default=0.10)
    parser.add_argument("--max-tolerance-px", type=float, default=0.25)
    parser.add_argument("--allow-approximate", action="store_true")
    args = parser.parse_args()

    xml = args.xml.resolve()
    if not xml.is_file():
        raise SystemExit(f"ERROR: XML not found: {xml}")
    input_pixel = float(args.input_pixel_size)
    source_xf_pixel = float(args.source_xf_pixel_size or input_pixel)
    final_pixel = float(args.final_pixel_size or source_xf_pixel)
    if input_pixel <= 0 or source_xf_pixel <= 0 or final_pixel <= 0:
        raise SystemExit("ERROR: pixel sizes must be positive")
    input_shape = args.input_shape
    final_shape = args.final_shape or input_shape

    try:
        from warpylib import TiltSeries
    except ModuleNotFoundError as exc:
        raise SystemExit("ERROR: warpylib is required to export Warp XML geometry") from exc
    ts = TiltSeries(path=str(xml))
    ts.load_meta(str(xml))
    n_tilts = int(ts.n_tilts)
    final_points_A = grid_points_centered_A(final_shape, final_pixel, *args.sample_grid)

    residual_matrices: list[np.ndarray] = []
    residual_shifts: list[np.ndarray] = []
    forward_physical_matrices: list[np.ndarray] = []
    forward_physical_shifts_A: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    failed = False
    for tilt in range(n_tilts):
        sampled_input_A = evaluate_warp_sampling_map(ts, tilt, final_points_A, input_shape, input_pixel)
        inverse_matrix, inverse_shift, residual_vectors = fit_affine(final_points_A, sampled_input_A)
        stats_A = residual_statistics(residual_vectors)
        stats_px = {key: value / input_pixel for key, value in stats_A.items()}
        forward_physical = np.linalg.inv(inverse_matrix)
        forward_shift_A = -forward_physical @ inverse_shift
        forward_matrix, forward_shift = physical_forward_to_pixel_xf(
            inverse_matrix, inverse_shift, input_pixel, final_pixel
        )
        forward_physical_matrices.append(forward_physical)
        forward_physical_shifts_A.append(forward_shift_A)
        residual_matrices.append(forward_matrix)
        residual_shifts.append(forward_shift)
        ok = stats_px["rms"] <= args.rms_tolerance_px and stats_px["max"] <= args.max_tolerance_px
        failed |= not ok
        reports.append({
            "tilt_index": tilt,
            "status": "PASS" if ok else "FAIL",
            "inverse_map_matrix_physical": inverse_matrix.tolist(),
            "inverse_map_shift_A": inverse_shift.tolist(),
            "forward_xf_matrix": forward_matrix.tolist(),
            "forward_xf_shift_px": forward_shift.tolist(),
            "residual_A": stats_A,
            "residual_px": stats_px,
        })

    residual_matrices_np = np.asarray(residual_matrices)
    residual_shifts_np = np.asarray(residual_shifts)
    forward_physical_matrices_np = np.asarray(forward_physical_matrices)
    forward_physical_shifts_A_np = np.asarray(forward_physical_shifts_A)

    if args.source_frame == "raw":
        final_matrices = (source_xf_pixel / final_pixel) * forward_physical_matrices_np
        final_shifts = forward_physical_shifts_A_np / final_pixel
    else:
        if args.out_residual_xf:
            write_xf(args.out_residual_xf, residual_matrices_np, residual_shifts_np)
        if args.original_xf is None:
            raise SystemExit("ERROR: --original-xf is required for --source-frame ali")
        original_matrices, original_shifts = read_xf(args.original_xf)
        if len(original_matrices) != n_tilts:
            raise SystemExit(f"ERROR: original XF has {len(original_matrices)} rows; XML has {n_tilts} tilts")
        original_raw_pixel = float(args.original_raw_pixel_size or source_xf_pixel)
        original_ali_pixel = float(args.original_ali_pixel_size or input_pixel)
        if original_raw_pixel <= 0 or original_ali_pixel <= 0:
            raise SystemExit("ERROR: original raw/ali pixel sizes must be positive")
        final_matrices = []
        final_shifts = []
        for tilt in range(n_tilts):
            original_matrix_physical = (original_ali_pixel / original_raw_pixel) * original_matrices[tilt]
            original_shift_A = original_ali_pixel * original_shifts[tilt]
            residual_matrix_physical = forward_physical_matrices_np[tilt]
            residual_shift_A = forward_physical_shifts_A_np[tilt]
            composed_matrix_physical = residual_matrix_physical @ original_matrix_physical
            composed_shift_A = residual_matrix_physical @ original_shift_A + residual_shift_A
            final_matrices.append((original_raw_pixel / final_pixel) * composed_matrix_physical)
            final_shifts.append(composed_shift_A / final_pixel)
        final_matrices = np.asarray(final_matrices)
        final_shifts = np.asarray(final_shifts)

    report = {
        "schema_version": 1,
        "xml": str(xml),
        "source_frame": args.source_frame,
        "n_tilts": n_tilts,
        "input_shape_xy": list(input_shape),
        "final_shape_xy": list(final_shape),
        "input_pixel_size_A": input_pixel,
        "source_xf_pixel_size_A": source_xf_pixel,
        "final_pixel_size_A": final_pixel,
        "original_raw_pixel_size_A": args.original_raw_pixel_size,
        "original_ali_pixel_size_A": args.original_ali_pixel_size,
        "sample_grid_xy": list(args.sample_grid),
        "rms_tolerance_px": args.rms_tolerance_px,
        "max_tolerance_px": args.max_tolerance_px,
        "status": "FAIL" if failed else "PASS",
        "tilts": reports,
    }
    report_json = args.report_json or args.out_xf.with_suffix(args.out_xf.suffix + ".validation.json")
    report_tsv = args.report_tsv or args.out_xf.with_suffix(args.out_xf.suffix + ".validation.tsv")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with report_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["tilt", "status", "rms_px", "p95_px", "max_px"])
        for item in reports:
            p = item["residual_px"]
            writer.writerow([item["tilt_index"], item["status"], p["rms"], p["p95"], p["max"]])

    if failed and not args.allow_approximate:
        raise SystemExit(
            f"ERROR: the Warp map is not sufficiently affine; see {report_json}. "
            "Use --allow-approximate only when an approximate XF is scientifically acceptable."
        )
    write_xf(args.out_xf, np.asarray(final_matrices), np.asarray(final_shifts))
    print(f"Wrote: {args.out_xf}")
    if args.source_frame == "ali" and args.out_residual_xf:
        print(f"Wrote residual: {args.out_residual_xf}")
    print(f"Validation: {report_json} ({report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
