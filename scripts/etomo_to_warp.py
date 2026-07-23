#!/usr/bin/env python3
"""Convert staged IMOD tilt-series inputs into Warp/warpylib metadata.

Supported alignment modes
-------------------------
``identity``
    No offsets or movement grids. Intended for an IMOD-generated ``.ali``.
``translation``
    Convert only the translational component of the IMOD transform.
``full-affine``
    Legacy v6 representation: encode the complete inverse affine as Warp
    ``GridMovementX/Y`` while leaving the raw stack orientation unchanged.
``quarter-turn-affine``
    v7 representation: select one common exact 0/90/180/270-degree stack
    permutation, materialize it losslessly in the stack, then encode only the
    residual affine in ``GridMovementX/Y``. This keeps large near-quarter-turn
    rotations out of the Warp movement field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch
from warpylib import CubicGrid, TiltSeries
from warpylib.ops.rescale import rescale

from geometry.quarter_turn import factor_affines, transform_shape_xy
from geometry.volume_frames import volume_frame_manifest
from imod_affine import (
    build_movement_grid_values,
    diagnose_matrix,
    inverse_physical_map,
    read_xf,
    transform_axis_angle_raw_to_aligned,
    warp_tilt_axis_angle_from_xf,
    write_xf,
)
from imod_affine import WARP_AXIS_ANGLE_CONVENTION_VERSION

ALIGNMENT_MODES = (
    "identity",
    "translation",
    "full-affine",
    "quarter-turn-affine",
)
AXIS_FRAMES = ("raw", "aligned")


def read_tilt_angles(path: Path) -> list[float]:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"no tilt angles in {path}")
    return values


def create_zero_grid(n_tilts: int) -> CubicGrid:
    return CubicGrid((1, 1, max(1, n_tilts)))


def apply_imod_positioning(ts, positioning, *, level_angle_x_sign: int = -1,
                           imod_to_warp_tilt_angle_sign: int = -1) -> dict:
    """Apply IMOD tilt.com positioning to a Warp ``TiltSeries`` (mandatory-policy mapping).

    OFFSET    -> baked into ``ts.angles = sign*(tlt+OFFSET)`` by the caller; here
                 ``ts.level_angle_y = 0`` (OFFSET applied EXACTLY ONCE, sharing the tilt rotation
                 order with LevelAngleX rather than a separate global rotation).
    XAXISTILT -> ``ts.level_angle_x = level_angle_x_sign * XAXISTILT`` (sign confirmed only by
                 the cluster validation script; NOT inferred from field names)
    SHIFT     -> object-space translation via ``ts.apply_tomogram_shift_3d([x, y, z] Angstrom)``,
                 composed with (not replacing) the existing .xf-derived per-tilt offsets.

    OFFSET is never applied twice: it is in ``ts.angles`` and ``ts.level_angle_y`` stays 0.
    Raises if a non-zero SHIFT is requested but the installed warpylib cannot represent it.
    """
    import torch

    from geometry.imod_positioning import validate_tilt_angle_sign
    tilt_angle_sign = validate_tilt_angle_sign(imod_to_warp_tilt_angle_sign)
    offset = float(positioning.tilt_angle_offset_deg)
    xaxis = float(positioning.x_axis_tilt_deg)
    ts.level_angle_y = 0.0                             # OFFSET is baked into ts.angles
    ts.level_angle_x = float(level_angle_x_sign) * xaxis
    applied = {
        "imod_offset_deg": offset,
        "imod_to_warp_tilt_angle_sign": tilt_angle_sign,
        "offset_representation": "baked_into_angles",
        "warp_level_angle_y_deg": 0.0,
        "warp_level_angle_x_deg": float(level_angle_x_sign) * xaxis,
        "level_angle_x_sign": int(level_angle_x_sign),
        "level_angle_x_sign_validated": False,          # confirmed by validate_warp_positioning.py
        "shift_representation": "none",
        "positioning_hash": positioning.positioning_hash(),
    }
    if positioning.has_nonzero_shift:
        if positioning.unbinned_pixel_size_A is None:
            raise ValueError("non-zero tilt.com SHIFT requires the unbinned IMOD pixel size")
        # ONE canonical SHIFT representation: the signed IMOD-MRC -> Warp frame transform.
        # Built in native IMOD-MRC order [sx_A, sz_A, 0] and transformed once; NOT [sx, 0, sz].
        # The .xf-derived per-view TiltAxisOffsetX/Y are a SEPARATE mechanism and are not
        # touched here (no projected SHIFT is added to them).
        from geometry.imod_positioning import imod_reconstruction_shift_to_warp
        shift_map = imod_reconstruction_shift_to_warp(
            positioning.shift_x_unbinned_px, positioning.shift_z_unbinned_px,
            positioning.unbinned_pixel_size_A, tilt_angle_sign=tilt_angle_sign)
        warp_shift_A = shift_map["warp_object_shift_A"]
        method = getattr(ts, "apply_tomogram_shift_3d", None)
        if not callable(method):
            raise ValueError(
                "tilt.com SHIFT is non-zero but the installed warpylib TiltSeries has no "
                "apply_tomogram_shift_3d(); it cannot represent the reconstruction shift. "
                "Use a warpylib version that supports it, or represent the shift as constant "
                "GridVolumeWarp values.")
        method(torch.tensor(warp_shift_A, dtype=torch.float32))
        applied["shift_representation"] = "apply_tomogram_shift_3d"
        applied["warp_object_shift_A"] = warp_shift_A
        applied["imod_shift_vector_A"] = shift_map["imod_shift_vector_A"]
        applied["orientation_matrix_imod_mrc_to_warp"] = shift_map["orientation_matrix_imod_mrc_to_warp"]
        applied["orientation_determinant"] = shift_map["orientation_determinant"]
    return applied


def process_tilt_series(
    folder_path: Path,
    output_directory: Path,
    tilt_axis_angle: float,
    volume_shape: tuple[int, int, int],
    output_pixel_size: float | None,
    alignment_mode: str,
    axis_frame: str,
    grid_shape_xy: tuple[int, int],
    positioning=None,
    level_angle_x_sign: int = -1,
    imod_to_warp_tilt_angle_sign: int = -1,
) -> tuple[TiltSeries, Path]:
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment mode {alignment_mode!r}")
    if axis_frame not in AXIS_FRAMES:
        raise ValueError(f"unknown axis frame {axis_frame!r}")

    folder_name = folder_path.name
    rawtlt_path = folder_path / f"{folder_name}.rawtlt"
    mrc_path = folder_path / f"{folder_name}.st"
    xf_path = folder_path / f"{folder_name}.xf"
    source_xf_path = folder_path / f"{folder_name}.source.xf"

    for path, desc in (
        (rawtlt_path, "tilt-angle"),
        (mrc_path, "image-stack"),
        (xf_path, "transform"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{desc} file not found: {path}")

    print(f"Processing {folder_name} ({alignment_mode}, axis frame={axis_frame})...")
    tilt_angles = read_tilt_angles(rawtlt_path)
    n_tilts = len(tilt_angles)

    with mrcfile.open(mrc_path, permissive=True) as handle:
        images = torch.tensor(np.asarray(handle.data, dtype=np.float32))
        input_pixel_size = float(handle.voxel_size.x)
    if images.ndim != 3:
        raise ValueError(f"expected a 3-D tilt stack, got {tuple(images.shape)}")
    if images.shape[0] != n_tilts:
        raise ValueError(
            f"stack has {images.shape[0]} sections, tilt file has {n_tilts} rows"
        )
    if input_pixel_size <= 0:
        raise ValueError(f"invalid/non-positive MRC pixel size in {mrc_path}")

    matrices_original, shifts_original = read_xf(xf_path)
    if len(matrices_original) != n_tilts:
        raise ValueError(f"XF has {len(matrices_original)} rows, expected {n_tilts}")

    axis_matrices_original = matrices_original
    if source_xf_path.exists():
        axis_matrices_original, _ = read_xf(source_xf_path)
        if len(axis_matrices_original) != n_tilts:
            raise ValueError(
                f"source XF has {len(axis_matrices_original)} rows, expected {n_tilts}"
            )

    matrices = matrices_original.copy()
    shifts = shifts_original.copy()
    axis_matrices = axis_matrices_original.copy()
    axis_input_angle = float(tilt_axis_angle)
    quarter_turn_manifest: dict[str, Any] | None = None
    quarter_turn_k = 0

    original_height, original_width = map(int, images.shape[-2:])
    reference_shape_xy = (original_width, original_height)
    source_volume_shape_imod_mrc_xyz = tuple(int(x) for x in volume_shape)

    if alignment_mode == "quarter-turn-affine":
        selected = factor_affines(axis_matrices_original, np.zeros((n_tilts, 2)))
        quarter_turn_k = selected.k
        actual = factor_affines(matrices_original, shifts_original, k=quarter_turn_k)
        axis_factor = factor_affines(
            axis_matrices_original,
            np.zeros((n_tilts, 2)),
            k=quarter_turn_k,
        )
        matrices = actual.residual_matrices
        shifts = actual.residual_shifts
        axis_matrices = axis_factor.residual_matrices
        axis_input_angle = transform_axis_angle_raw_to_aligned(
            tilt_axis_angle,
            selected.quarter_turn_matrix,
        )
        images = torch.rot90(images, k=quarter_turn_k, dims=(-2, -1)).contiguous()
        reference_shape_xy = transform_shape_xy(
            (original_width, original_height), quarter_turn_k
        )
        quarter_turn_manifest = {
            **selected.to_dict(),
            "raw_shape_xy": [original_width, original_height],
            "rotated_shape_xy": list(reference_shape_xy),
            "raw_tilt_axis_angle_deg": float(tilt_axis_angle),
            "rotated_tilt_axis_angle_deg": float(axis_input_angle),
            "pixel_value_policy": "exact permutation; no interpolation",
            "detector_rotation_axis_normal_in_warp_frame": "Z",
            "volume_frame_policy": (
                "The quarter turn is a detector-frame basis change only; "
                "Warp reconstruction-volume XYZ extents remain unchanged."
            ),
            "quantitative_use": (
                "Allowed as an exact detector-coordinate reindexing. The transform and "
                "frame change must remain attached to the artifact."
            ),
        }

    # IMOD -> Warp tilt-angle sign, validated once and used for BOTH the signed angles and the
    # signed volume-frame orientation (SHIFT). The shape permutation is unaffected.
    from geometry.imod_positioning import validate_tilt_angle_sign
    tilt_angle_sign = validate_tilt_angle_sign(imod_to_warp_tilt_angle_sign)

    volume_frame = volume_frame_manifest(
        source_volume_shape_imod_mrc_xyz,
        quarter_turn_k=quarter_turn_k,
        tilt_angle_sign=tilt_angle_sign,
    )
    warp_volume_shape_xyz = tuple(volume_frame["current_shape_warp_xyz"])

    final_pixel_size = input_pixel_size
    downsampling_correction = None
    pre_rescale_height, pre_rescale_width = map(int, images.shape[-2:])
    if output_pixel_size is not None and output_pixel_size > input_pixel_size:
        factor = float(output_pixel_size) / input_pixel_size
        scaled_width = max(2, int(round(pre_rescale_width / factor / 2)) * 2)
        scaled_height = max(2, int(round(pre_rescale_height / factor / 2)) * 2)
        if (scaled_height, scaled_width) != (pre_rescale_height, pre_rescale_width):
            images = rescale(images, size=(scaled_height, scaled_width))
        # Isotropic physical downsample; the declared pixel is from X (5760/8=720 -> 17.6). Y can
        # round (e.g. 4092/8=511.5 -> 512), so its EFFECTIVE pixel differs slightly. Record the
        # per-axis factors + half-pixel rounding instead of declaring both axes exactly 17.6.
        final_pixel_size = input_pixel_size * pre_rescale_width / scaled_width
        eff_px_x = input_pixel_size * pre_rescale_width / scaled_width
        eff_px_y = input_pixel_size * pre_rescale_height / scaled_height
        downsampling_correction = {
            "isotropic_physical_downsample": True,
            "input_shape_xy": [pre_rescale_width, pre_rescale_height],
            "output_shape_xy": [scaled_width, scaled_height],
            "ideal_factor": factor,
            "ideal_output_xy": [pre_rescale_width / factor, pre_rescale_height / factor],
            "actual_factor_xy": [pre_rescale_width / scaled_width, pre_rescale_height / scaled_height],
            "effective_pixel_size_xy_A": [eff_px_x, eff_px_y],
            "declared_pixel_size_A": final_pixel_size,   # X (isotropic Warp voxel)
            "even_size_rounding_note": "e.g. 4092/8=511.5 -> 512 (+0.5 px); Y pixel != X pixel",
        }
    elif output_pixel_size is not None and output_pixel_size < input_pixel_size - 1e-6:
        raise ValueError(
            f"output pixel size {output_pixel_size} is smaller than input {input_pixel_size}; "
            "upsampling is deliberately disabled"
        )

    stack_dir = output_directory / "tiltstack" / folder_name
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stack_dir / f"{folder_name}.st"
    xml_path = output_directory / f"{folder_name}.xml"

    # Angles = sign * (tlt + OFFSET), applied EXACTLY ONCE (view order preserved: warp row i ==
    # source section i). OFFSET is baked into the per-view Angles (LevelAngleY = 0) so it shares
    # the tilt rotation order with a non-zero LevelAngleX; the raw .tlt on disk is untouched. The
    # tlt comes from tomo2.tlt (the resolved final tilt file), never .rawtlt.
    _offset_deg = float(positioning.tilt_angle_offset_deg) if positioning is not None else 0.0
    warp_angles = [tilt_angle_sign * (float(a) + _offset_deg) for a in tilt_angles]

    ts = TiltSeries(path=str(xml_path), n_tilts=n_tilts)
    ts.angles = torch.tensor(warp_angles, dtype=torch.float32)

    # Per-view Warp TiltAxisAngle = EACH source .xf polar rotation DIRECTLY (never a fixed
    # align.com value; matches Warp's official importer). No 180 deg adjustment: the .xf branch
    # (~-95.5) is already the directed axis for the negated tilt angles. axis_input_angle
    # (align.com, e.g. 84.1) is provenance/fallback only. The aligned-frame condition keeps its
    # own per-view mapping (a separate mechanism, not the fixed-value error).
    source_axis_angles_deg: list[float] = []
    axis_direction_adjustments_deg: list[float] = []
    if axis_frame == "aligned":
        axis_angles = [
            transform_axis_angle_raw_to_aligned(axis_input_angle, matrix)
            for matrix in axis_matrices
        ]
    else:
        axis_angles = []
        for matrix in axis_matrices:
            warp_axis, imod_axis, adjust = warp_tilt_axis_angle_from_xf(
                matrix, angle_sign=tilt_angle_sign, reference_angle_deg=axis_input_angle)
            axis_angles.append(warp_axis)
            source_axis_angles_deg.append(imod_axis)
            axis_direction_adjustments_deg.append(adjust)
    ts.tilt_axis_angles = torch.tensor(axis_angles, dtype=torch.float32)

    # IMOD tilt.com positioning (OFFSET/XAXISTILT/SHIFT). Applied after the raw angles and
    # tilt-axis angles are set; the raw .tlt stays in ts.angles. No-op when positioning is
    # None (backward compatible). The .xf-derived offsets/movement grids below are composed
    # with, not replaced by, this positioning.
    warp_positioning_applied = None
    if positioning is not None:
        warp_positioning_applied = apply_imod_positioning(
            ts, positioning, level_angle_x_sign=level_angle_x_sign,
            imod_to_warp_tilt_angle_sign=tilt_angle_sign)
        # OFFSET applied EXACTLY ONCE: effective Warp angle == sign * (tlt + OFFSET). Angles
        # hold sign*tlt (OFFSET not baked in); LevelAngleY holds sign*OFFSET. Guard against a
        # future double application.
        _offset = float(positioning.tilt_angle_offset_deg)
        _level_y = float(warp_positioning_applied["warp_level_angle_y_deg"])
        for _i, _tlt in enumerate(tilt_angles):
            _effective_warp = warp_angles[_i] + _level_y
            _expected = tilt_angle_sign * (float(_tlt) + _offset)
            if abs(_effective_warp - _expected) > 1e-6:
                raise AssertionError(
                    f"effective Warp angle {_effective_warp} != sign*(tlt+OFFSET) {_expected} "
                    f"at view {_i} (OFFSET applied twice or baked into Angles)")

    height, width = map(int, images.shape[-2:])
    ts.image_dimensions_physical = torch.tensor(
        [width * final_pixel_size, height * final_pixel_size], dtype=torch.float32
    )
    ts.volume_dimensions_physical = torch.tensor(
        [value * final_pixel_size for value in warp_volume_shape_xyz],
        dtype=torch.float32,
    )
    ref_width, ref_height = reference_shape_xy
    ts.size_rounding_factors = torch.tensor(
        [
            width / (ref_width * input_pixel_size / final_pixel_size),
            height / (ref_height * input_pixel_size / final_pixel_size),
            1.0,
        ],
        dtype=torch.float32,
    )

    manifest: dict[str, Any] = {
        "schema_version": 4,
        "series": folder_name,
        "alignment_mode": alignment_mode,
        "axis_frame": axis_frame,
        "raw_tilt_axis_angle_deg": float(tilt_axis_angle),
        "axis_angle_before_residual_affine_deg": float(axis_input_angle),
        "warp_tilt_axis_angles_deg": axis_angles,
        "tilt_axis_angle_provenance": {
            "warp_axis_angle_convention_version": WARP_AXIS_ANGLE_CONVENTION_VERSION,
            "source": ("per_view_xf_polar_rotation" if axis_frame != "aligned"
                       else "per_view_raw_to_aligned_axis_transform"),
            "initial_axis_estimate_deg": float(axis_input_angle),   # align.com; reference/fallback only
            "imod_to_warp_tilt_angle_sign": int(tilt_angle_sign),
            "source_axis_angle_deg": source_axis_angles_deg,        # per-view IMOD .xf rotation
            "axis_direction_adjustment_deg": axis_direction_adjustments_deg,  # 0 (no reversal)
            "final_warp_axis_angle_deg": [float(a) for a in axis_angles],     # == source .xf rotation
            "tilt_axis_angles_hash": hashlib.sha256(
                json.dumps([round(float(a), 6) for a in axis_angles]).encode()).hexdigest(),
        },
        "input_stack": str(mrc_path.resolve()),
        "output_stack": str(stack_path.resolve()),
        "input_pixel_size_A": input_pixel_size,
        "output_pixel_size_A": final_pixel_size,
        "input_shape_zyx": [n_tilts, original_height, original_width],
        "image_shape_zyx": [int(x) for x in images.shape],
        # Legacy alias: source IMOD reconstruction MRC storage order.
        "volume_shape_xyz": list(source_volume_shape_imod_mrc_xyz),
        "volume_shape_frame": "imod_reconstruction_mrc_xyz__y_is_thickness",
        "warp_volume_shape_xyz": list(warp_volume_shape_xyz),
        "volume_frame": volume_frame,
        "xf_file": str(xf_path.resolve()),
        "source_xf_file": str(source_xf_path.resolve()) if source_xf_path.exists() else None,
        "movement_grid_shape_xy": list(grid_shape_xy),
        "coordinate_frame": (
            "quarter_turn_stack" if alignment_mode == "quarter-turn-affine" else "raw_stack"
        ),
        "branch": "quantitative",
        "half_set_policy": "no half-set assignment or mixing occurs during tilt-series conversion",
        "allowed_uses": [
            "geometry validation",
            "particle picking initialization",
            "MissAlignment input after PRE acceptance",
            "quantitative reconstruction with attached provenance",
        ],
        "forbidden_uses": [
            "silent replacement of source observations",
            "FSC or resolution claims before cluster validation",
        ],
        "resampling_history": [
            {
                "operation": "quarter_turn" if quarter_turn_manifest is not None else "none",
                "np_rot90_k": int(quarter_turn_k),
                "interpolation": "none",
            },
            {
                "operation": "isotropic_downsample" if final_pixel_size > input_pixel_size + 1e-6 else "none",
                "input_pixel_size_A": input_pixel_size,
                "output_pixel_size_A": final_pixel_size,
                "output_shape_yx": [height, width],
            },
        ],
    }
    if quarter_turn_manifest is not None:
        residual_xf_path = output_directory / f"{folder_name}.residual.xf"
        write_xf(residual_xf_path, matrices, shifts)
        manifest["quarter_turn"] = quarter_turn_manifest
        manifest["residual_xf_file"] = str(residual_xf_path.resolve())

    # Direct-stack view order is identity (warp row i == source section i); the tilt-angle
    # sign is recorded so the Warp->IMOD export can apply the exact inverse.
    from geometry.imod_positioning import (
        tilt_angle_convention_manifest, tilt_view_order_identity)
    manifest["tilt_view_order"] = tilt_view_order_identity(n_tilts)
    manifest["tilt_angle_convention"] = tilt_angle_convention_manifest(tilt_angle_sign)
    if downsampling_correction is not None:
        manifest["downsampling_correction"] = downsampling_correction

    if positioning is not None:
        manifest["imod_positioning"] = positioning.to_manifest()
        manifest["warp_positioning_applied"] = warp_positioning_applied
        # Extend the volume-frame entry with the SHIFT vectors in both frames (the orientation
        # matrix/determinant/handedness/shape_permutation are already in volume_frame).
        if warp_positioning_applied and "imod_shift_vector_A" in warp_positioning_applied:
            manifest["volume_frame"]["imod_shift_vector_A"] = warp_positioning_applied["imod_shift_vector_A"]
            manifest["volume_frame"]["warp_shift_vector_A"] = warp_positioning_applied["warp_object_shift_A"]

    if alignment_mode == "identity":
        ts.tilt_axis_offset_x = torch.zeros(n_tilts, dtype=torch.float32)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts, dtype=torch.float32)
        ts.grid_movement_x = create_zero_grid(n_tilts)
        ts.grid_movement_y = create_zero_grid(n_tilts)
        manifest["offsets_xy_A"] = [[0.0, 0.0] for _ in range(n_tilts)]
    elif alignment_mode == "translation":
        offsets = np.zeros((n_tilts, 2), dtype=float)
        for index, (matrix, shift) in enumerate(zip(matrices, shifts, strict=True)):
            _, inverse_shift = inverse_physical_map(
                matrix, shift, input_pixel_size, input_pixel_size
            )
            offsets[index] = inverse_shift
        ts.tilt_axis_offset_x = torch.tensor(offsets[:, 0], dtype=torch.float32)
        ts.tilt_axis_offset_y = torch.tensor(offsets[:, 1], dtype=torch.float32)
        ts.grid_movement_x = create_zero_grid(n_tilts)
        ts.grid_movement_y = create_zero_grid(n_tilts)
        manifest["offsets_xy_A"] = offsets.tolist()
    elif alignment_mode in ("full-affine", "quarter-turn-affine"):
        values_x, values_y, offsets = build_movement_grid_values(
            matrices,
            shifts,
            raw_shape_xy=reference_shape_xy,
            raw_pixel_size_A=input_pixel_size,
            aligned_pixel_size_A=input_pixel_size,
            grid_shape_xy=grid_shape_xy,
            grid_image_shape_xy=(width, height),
            grid_image_pixel_size_A=final_pixel_size,
        )
        gx, gy = grid_shape_xy
        ts.tilt_axis_offset_x = torch.tensor(offsets[:, 0], dtype=torch.float32)
        ts.tilt_axis_offset_y = torch.tensor(offsets[:, 1], dtype=torch.float32)
        ts.grid_movement_x = CubicGrid(
            (gx, gy, n_tilts), torch.tensor(values_x, dtype=torch.float32)
        )
        ts.grid_movement_y = CubicGrid(
            (gx, gy, n_tilts), torch.tensor(values_y, dtype=torch.float32)
        )
        manifest["offsets_xy_A"] = offsets.tolist()
        manifest["matrix_diagnostics"] = [
            diagnose_matrix(matrix).__dict__ for matrix in matrices
        ]
        manifest["movement_grid_range_A"] = {
            "x": [float(np.min(values_x)), float(np.max(values_x))],
            "y": [float(np.min(values_y)), float(np.max(values_y))],
        }
    else:  # pragma: no cover
        raise ValueError(alignment_mode)

    with mrcfile.new(stack_path, overwrite=True) as handle:
        handle.set_data(images.cpu().numpy().astype(np.float32, copy=False))
        handle.voxel_size = final_pixel_size
        handle.update_header_stats()
    ts.save_meta(str(xml_path))

    manifest_path = output_directory / f"{folder_name}.conversion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  Final pixel size: {final_pixel_size} Å")
    if quarter_turn_manifest is not None:
        print(
            "  Quarter turn: "
            f"np.rot90(k={quarter_turn_k}), residual rotation max "
            f"{quarter_turn_manifest['residual_rotation_max_abs_deg']:.3f}°"
        )
    print(f"  Created: {xml_path}")
    print(f"  Created: {stack_path}")
    print(f"  Manifest: {manifest_path}")
    return ts, xml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("warp_tiltseries"))
    parser.add_argument("--tilt-axis-angle", type=float, required=True)
    parser.add_argument(
        "--volume-shape",
        type=int,
        nargs=3,
        required=True,
        metavar=("IMOD_MRC_X", "IMOD_MRC_Y_THICKNESS", "IMOD_MRC_Z"),
        help=(
            "source IMOD reconstruction MRC shape. The converter maps "
            "IMOD-MRC (X,Y,Z) to Warp volume (X,Z,Y). Any selected detector-plane "
            "quarter turn changes projection coordinates only and does not swap "
            "the Warp reconstruction-volume X/Y extents"
        ),
    )
    parser.add_argument("--output-pixel-size", type=float, default=None)
    parser.add_argument("--alignment-mode", choices=ALIGNMENT_MODES, default="translation")
    parser.add_argument("--axis-frame", choices=AXIS_FRAMES, default="raw")
    parser.add_argument(
        "--movement-grid-shape",
        type=int,
        nargs=2,
        default=(5, 5),
        metavar=("NX", "NY"),
    )
    parser.add_argument(
        "--imod-positioning-json",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file holding the resolved [geometry.imod_positioning] table "
            "(OFFSET/XAXISTILT/SHIFT/THICKNESS). When given, the positioning is applied "
            "to every TiltSeries so the legacy path preserves it like the canonical path."
        ),
    )
    parser.add_argument("--level-angle-x-sign", type=int, choices=(1, -1), default=-1)
    parser.add_argument("--imod-tilt-angle-sign", type=int, choices=(1, -1), default=None,
                        help="IMOD->Warp tilt-angle sign; defaults to the positioning table "
                             "value or -1.")
    args = parser.parse_args()

    from geometry.imod_positioning import IMOD_TO_WARP_TILT_ANGLE_SIGN, validate_tilt_angle_sign
    positioning = None
    tilt_angle_sign = IMOD_TO_WARP_TILT_ANGLE_SIGN
    if args.imod_positioning_json is not None:
        if not args.imod_positioning_json.is_file():
            raise SystemExit(f"ERROR: --imod-positioning-json not found: {args.imod_positioning_json}")
        from geometry.imod_positioning import from_toml_table
        positioning = from_toml_table(json.loads(args.imod_positioning_json.read_text()))
        tilt_angle_sign = positioning.imod_to_warp_tilt_angle_sign
    if args.imod_tilt_angle_sign is not None:
        tilt_angle_sign = validate_tilt_angle_sign(args.imod_tilt_angle_sign)

    input_directory = args.input_dir.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    folders = sorted(path for path in input_directory.glob("TS_*") if path.is_dir())
    if not folders:
        raise SystemExit(f"ERROR: no TS_* folders found in {input_directory}")
    print(f"Found {len(folders)} TS_* folders")
    for folder in folders:
        process_tilt_series(
            folder,
            output_directory,
            float(args.tilt_axis_angle),
            tuple(int(x) for x in args.volume_shape),
            args.output_pixel_size,
            args.alignment_mode,
            args.axis_frame,
            tuple(int(x) for x in args.movement_grid_shape),
            positioning=positioning,
            level_angle_x_sign=int(args.level_angle_x_sign),
            imod_to_warp_tilt_angle_sign=tilt_angle_sign,
        )
    print("\nConversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
