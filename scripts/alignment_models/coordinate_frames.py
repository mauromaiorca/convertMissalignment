#!/usr/bin/env python3
"""Single geometry/composition layer for constrained refinement.

This module owns *all* coordinate-frame conversions so that sign, unit, centre,
and axis-order logic is never duplicated across scripts. It is the code
counterpart of ``docs/interoperability/COORDINATE_CONVENTIONS.md``; any change
here must be reflected there.

Frames (all 2-D, ``(x, y)`` with ``x`` the fast/column axis, matching IMOD
``.xf`` and the last axis of an MRC ``[z, y, x]`` array)
------------------------------------------------------------------------------
- **raw IMOD pixel-centred**   : pixels, centred at the raw image centre.
- **aligned IMOD pixel-centred**: pixels, centred at the aligned image centre.
- **raw physical (A)**         : Angstrom, ``x_A = x_px * p_raw``.
- **aligned physical (A)**     : Angstrom, ``x_A = x_px * p_ali``.
- **centred geometric**        : relative to an image centre (pixel or physical).
- **absolute array indices**   : 0-based pixel indices ``(x, y)``.
- **homogeneous**              : 3x3 matrix acting on ``[x, y, 1]`` (absolute).

Centre conventions (verified in the V&V audit; do **not** change without
re-deriving against real IMOD ``newstack``)
------------------------------------------------------------------------------
- **IMOD pixel centre** = ``((nx - 1) / 2, (ny - 1) / 2)`` (0-based). This is the
  geometric centre ``(nx + 1) / 2`` in IMOD's 1-based coordinates and matches
  real ``newstack`` to < 0.005 px. Provided by ``imod_affine.image_center_xy``
  (convention ``"imod"``). Never use ``nx / 2`` as the IMOD centre.
- **physical model centre** = ``image_dimensions_physical / 2 = shape_xy * p / 2``
  (continuous). Residual refinement models act around this centre. It differs
  from the IMOD pixel centre by half a pixel; the difference is carried
  explicitly through the homogeneous matrices, so exact ``.xf`` export remains
  correct. See ``COORDINATE_CONVENTIONS.md`` and audit ISSUE-006.

Representation strategy
-----------------------
Every transform is reduced to an **absolute homogeneous 3x3** matrix in a named
frame (pixel or physical). Composition is plain matrix multiplication on
absolute homogeneous matrices, which is centre/origin-agnostic and therefore
robust. The only places a centre enters are:

1. building a model matrix from ``(A, t)`` around the physical centre, and
2. converting an absolute-pixel homogeneous matrix to/from an IMOD ``.xf`` row,
   which uses the verified IMOD pixel centre.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Reuse the audited IMOD-pixel helpers (already use the (n-1)/2 centre).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import (  # noqa: E402
    homogeneous_to_xf,
    image_center_xy,
    xf_to_homogeneous,
)

__all__ = [
    "physical_center_xy",
    "imod_pixel_center_xy",
    "make_homogeneous",
    "split_homogeneous",
    "model_to_abs_physical",
    "abs_physical_from_model_centered",
    "imod_xf_to_abs_physical",
    "abs_physical_to_imod_xf",
    "pixel_homogeneous_to_physical",
    "physical_homogeneous_to_pixel",
    "apply_homogeneous",
    "compose_homogeneous",
    "invert_homogeneous",
]


def _shape(shape_xy: Sequence[float]) -> np.ndarray:
    s = np.asarray(shape_xy, dtype=float)
    if s.shape != (2,) or np.any(s <= 0):
        raise ValueError(f"expected positive (X, Y) shape, got {shape_xy!r}")
    return s


def physical_center_xy(shape_xy: Sequence[int], pixel_size_A: float) -> np.ndarray:
    """Physical model centre ``shape * pixel / 2`` (Angstrom)."""
    if pixel_size_A <= 0:
        raise ValueError("pixel size must be positive")
    return _shape(shape_xy) * float(pixel_size_A) / 2.0


def imod_pixel_center_xy(shape_xy: Sequence[int]) -> np.ndarray:
    """IMOD pixel centre ``(n-1)/2`` (delegates to the audited helper)."""
    return image_center_xy(shape_xy, "imod")


def make_homogeneous(matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 3x3 homogeneous matrix ``[[A, t], [0, 1]]`` (absolute frame)."""
    A = np.asarray(matrix, dtype=float).reshape(2, 2)
    t = np.asarray(translation, dtype=float).reshape(2)
    H = np.eye(3, dtype=float)
    H[:2, :2] = A
    H[:2, 2] = t
    return H


def split_homogeneous(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, t)`` from a 3x3 homogeneous matrix; validate affine row."""
    H = np.asarray(H, dtype=float).reshape(3, 3)
    if not np.allclose(H[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("not a 2-D affine homogeneous matrix (bad last row)")
    return H[:2, :2].copy(), H[:2, 2].copy()


def model_to_abs_physical(
    matrix: np.ndarray, translation_A: np.ndarray, center_phys_A: np.ndarray
) -> np.ndarray:
    """Centred-physical model ``q_c = A p_c + t`` -> absolute-physical homogeneous.

    With ``p_c = p - c`` and ``q_c = q - c``: ``q = A p + (t + (I - A) c)``.
    """
    A = np.asarray(matrix, dtype=float).reshape(2, 2)
    t = np.asarray(translation_A, dtype=float).reshape(2)
    c = np.asarray(center_phys_A, dtype=float).reshape(2)
    return make_homogeneous(A, t + (np.eye(2) - A) @ c)


def abs_physical_from_model_centered(
    matrix: np.ndarray,
    translation_A: np.ndarray,
    shape_xy: Sequence[int],
    pixel_size_A: float,
) -> np.ndarray:
    """Convenience: model ``(A, t)`` -> absolute-physical homogeneous using the
    physical centre of the given aligned image."""
    c = physical_center_xy(shape_xy, pixel_size_A)
    return model_to_abs_physical(matrix, translation_A, c)


def pixel_homogeneous_to_physical(
    H_pixel: np.ndarray, p_in_A: float, p_out_A: float
) -> np.ndarray:
    """Absolute-pixel homogeneous -> absolute-physical homogeneous.

    ``x_A = x_px * p_in``, ``y_A = y_px * p_out`` (isotropic per image).
    ``A_phys = (p_out / p_in) A_px``; ``t_phys = p_out * t_px``.
    """
    A, t = split_homogeneous(H_pixel)
    return make_homogeneous((p_out_A / p_in_A) * A, p_out_A * t)


def physical_homogeneous_to_pixel(
    H_phys: np.ndarray, p_in_A: float, p_out_A: float
) -> np.ndarray:
    """Inverse of :func:`pixel_homogeneous_to_physical`."""
    A, t = split_homogeneous(H_phys)
    return make_homogeneous((p_in_A / p_out_A) * A, t / p_out_A)


def imod_xf_to_abs_physical(
    matrix_xf: np.ndarray,
    shift_xf_px: np.ndarray,
    in_shape_xy: Sequence[int],
    out_shape_xy: Sequence[int],
    p_in_A: float,
    p_out_A: float,
) -> np.ndarray:
    """IMOD ``.xf`` (centred pixel, (n-1)/2) -> absolute-physical homogeneous."""
    H_px = xf_to_homogeneous(matrix_xf, shift_xf_px, in_shape_xy, out_shape_xy)
    return pixel_homogeneous_to_physical(H_px, p_in_A, p_out_A)


def abs_physical_to_imod_xf(
    H_phys: np.ndarray,
    in_shape_xy: Sequence[int],
    out_shape_xy: Sequence[int],
    p_in_A: float,
    p_out_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-physical homogeneous -> IMOD ``.xf`` ``(A, d)`` (centred pixel).

    The IMOD ``(n-1)/2`` centre is applied here (and only here) via
    ``imod_affine.homogeneous_to_xf``, so the exported row is exact.
    """
    H_px = physical_homogeneous_to_pixel(H_phys, p_in_A, p_out_A)
    return homogeneous_to_xf(H_px, in_shape_xy, out_shape_xy)


def apply_homogeneous(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a homogeneous matrix to ``(N, 2)`` absolute points."""
    H = np.asarray(H, dtype=float).reshape(3, 3)
    p = np.asarray(points_xy, dtype=float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    out = p @ H[:2, :2].T + H[:2, 2]
    return out[0] if single else out


def compose_homogeneous(*matrices: np.ndarray) -> np.ndarray:
    """Compose homogeneous matrices left-to-right as ``M[-1] @ ... @ M[0]``.

    ``compose_homogeneous(H0, DeltaH)`` returns ``DeltaH @ H0`` -- i.e. apply
    ``H0`` first, then ``DeltaH``. This is the canonical ``Hfinal`` order.
    """
    if not matrices:
        raise ValueError("need at least one matrix")
    result = np.asarray(matrices[0], dtype=float).reshape(3, 3)
    for H in matrices[1:]:
        result = np.asarray(H, dtype=float).reshape(3, 3) @ result
    return result


def invert_homogeneous(H: np.ndarray) -> np.ndarray:
    """Invert a 3x3 affine homogeneous matrix (raises if singular)."""
    A, t = split_homogeneous(H)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise ValueError(f"singular/near-singular transform (det={det:.3e})")
    inv = np.linalg.inv(A)
    return make_homogeneous(inv, -inv @ t)
