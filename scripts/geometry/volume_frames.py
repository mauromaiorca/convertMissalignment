#!/usr/bin/env python3
"""Explicit volume- and detector-axis contracts for IMOD and Warp.

IMOD tilt reconstruction files use MRC storage order ``(X, Y, Z)`` where the
reconstruction thickness is stored in MRC ``Y`` and the detector/tilt-axis
extent is stored in MRC ``Z``. Warp's logical tomogram dimensions are
``(X, Y, Z)`` with thickness in Warp ``Z``.

Therefore the storage-to-Warp volume mapping is::

    Warp volume (X, Y, Z) = IMOD-MRC (X, Z, Y)

The lossless ``np.rot90`` used by ``quarter-turn-affine`` acts only on the 2-D
detector frame. It swaps the projection-image width and height for odd quarter
turns and requires the corresponding axis, shift and affine-coordinate
transforms. It does *not* rotate the requested reconstruction-volume frame or
exchange the Warp volume X/Y extents.
"""
from __future__ import annotations

from typing import Sequence

IMOD_MRC_FRAME = "imod_reconstruction_mrc_xyz__y_is_thickness"
WARP_FRAME = "warp_tomogram_xyz__z_is_thickness"
DETECTOR_FRAME = "warp_projection_detector_xy"
VOLUME_FRAME_CONTRACT_VERSION = 2
BASE_AXIS_PERMUTATION = (0, 2, 1)


def _shape3(shape: Sequence[int], *, label: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"{label} must contain three positive integers, got {shape!r}")
    return values


def _shape2(shape: Sequence[int], *, label: str) -> tuple[int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 2 or any(v <= 0 for v in values):
        raise ValueError(f"{label} must contain two positive integers, got {shape!r}")
    return values


def imod_mrc_shape_to_warp_xyz(shape_imod_mrc_xyz: Sequence[int]) -> tuple[int, int, int]:
    """Map IMOD reconstruction MRC storage extents to Warp logical XYZ."""
    shape = _shape3(shape_imod_mrc_xyz, label="IMOD MRC volume shape")
    return tuple(shape[index] for index in BASE_AXIS_PERMUTATION)


def apply_projection_quarter_turn_to_detector_shape_xy(
    shape_xy: Sequence[int],
    k: int,
) -> tuple[int, int]:
    """Return detector-image width/height after ``np.rot90(..., k)``.

    Odd quarter turns swap detector X/Y extents. This operation is deliberately
    separate from the 3-D reconstruction-volume contract.
    """
    x, y = _shape2(shape_xy, label="detector shape")
    return (y, x) if int(k) % 2 else (x, y)


def apply_projection_quarter_turn_to_warp_shape(
    shape_warp_xyz: Sequence[int],
    k: int,
) -> tuple[int, int, int]:
    """Compatibility wrapper returning the unchanged Warp volume shape.

    Earlier alpha versions incorrectly treated the detector quarter turn as a
    rotation of the reconstruction-volume frame. The parameter ``k`` is kept
    for API compatibility, but volume extents are invariant under this detector
    basis change.
    """
    del k
    return _shape3(shape_warp_xyz, label="Warp volume shape")


def imod_mrc_shape_to_current_warp_xyz(
    shape_imod_mrc_xyz: Sequence[int],
    *,
    quarter_turn_k: int = 0,
) -> tuple[int, int, int]:
    """Return the reconstruction volume shape in Warp XYZ.

    ``quarter_turn_k`` is accepted for compatibility and provenance. It affects
    detector coordinates, not the Warp reconstruction-volume extents.
    """
    del quarter_turn_k
    return imod_mrc_shape_to_warp_xyz(shape_imod_mrc_xyz)


def volume_frame_manifest(
    shape_imod_mrc_xyz: Sequence[int],
    *,
    quarter_turn_k: int,
) -> dict:
    source = _shape3(shape_imod_mrc_xyz, label="IMOD MRC volume shape")
    base = imod_mrc_shape_to_warp_xyz(source)
    k = int(quarter_turn_k) % 4
    return {
        "contract_version": VOLUME_FRAME_CONTRACT_VERSION,
        "source_frame": IMOD_MRC_FRAME,
        "target_frame": WARP_FRAME,
        "source_shape_imod_mrc_xyz": list(source),
        "base_axis_permutation_imod_mrc_to_warp": list(BASE_AXIS_PERMUTATION),
        "base_shape_warp_xyz": list(base),
        "projection_quarter_turn_k": k,
        "projection_quarter_turn_scope": "detector_frame_only",
        "projection_detector_frame": DETECTOR_FRAME,
        "reconstruction_shape_warp_xyz": list(base),
        # Compatibility key consumed by existing reconstruction code.
        "current_shape_warp_xyz": list(base),
        "volume_shape_invariant_under_detector_quarter_turn": True,
        "thickness_mapping": "IMOD-MRC Y -> Warp Z",
        "detector_vertical_mapping": "IMOD-MRC Z -> Warp Y",
        "quarter_turn_effect": (
            "projection detector X/Y, tilt-axis angle, shifts and affine coordinates "
            "are transformed; Warp reconstruction-volume XYZ extents are unchanged"
        ),
    }
