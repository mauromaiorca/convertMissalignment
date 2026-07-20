#!/usr/bin/env python3
"""Exact quarter-turn factorization for centred IMOD 2-D affine transforms.

The v7 converter avoids encoding a near-90-degree global rotation as a very
large Warp movement field.  A single lossless quarter turn is materialized in
the tilt stack, and only the residual affine is encoded in Warp metadata.

Coordinate convention
---------------------
``np.rot90(image, k)`` is applied over the final ``(Y, X)`` axes.  In centred
``(X, Y)`` coordinates the corresponding raw->quarter-turn matrices are::

    k=0: [[ 1,  0], [ 0,  1]]
    k=1: [[ 0,  1], [-1,  0]]
    k=2: [[-1,  0], [ 0, -1]]
    k=3: [[ 0, -1], [ 1,  0]]

For an original centred transform ``a = A @ r + d`` and ``q = Q @ r``, the
residual transform is exactly ``a = (A @ Q.T) @ q + d``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class QuarterTurnFactorization:
    k: int
    quarter_turn_matrix: np.ndarray
    residual_matrices: np.ndarray
    residual_shifts: np.ndarray
    original_rotation_deg: np.ndarray
    residual_rotation_deg: np.ndarray
    max_recomposition_error: float

    @property
    def quarter_turn_angle_deg(self) -> float:
        q = self.quarter_turn_matrix
        return float(np.degrees(np.arctan2(q[1, 0], q[0, 0])))

    @property
    def residual_rotation_max_abs_deg(self) -> float:
        return float(np.max(np.abs(self.residual_rotation_deg)))

    @property
    def residual_rotation_median_abs_deg(self) -> float:
        return float(np.median(np.abs(self.residual_rotation_deg)))

    def to_dict(self) -> dict:
        return {
            "np_rot90_k": int(self.k),
            "quarter_turn_angle_deg": self.quarter_turn_angle_deg,
            "quarter_turn_matrix_raw_to_rotated": self.quarter_turn_matrix.tolist(),
            "original_rotation_deg": self.original_rotation_deg.tolist(),
            "residual_rotation_deg": self.residual_rotation_deg.tolist(),
            "residual_rotation_max_abs_deg": self.residual_rotation_max_abs_deg,
            "residual_rotation_median_abs_deg": self.residual_rotation_median_abs_deg,
            "max_recomposition_error": self.max_recomposition_error,
            "factorization": "A_original = A_residual @ Q",
            "stack_operation": "numpy/torch rot90 over (Y,X), no interpolation",
        }


def _normalise_k(k: int) -> int:
    return int(k) % 4


def quarter_turn_matrix(k: int) -> np.ndarray:
    """Return the centred raw->rotated matrix matching ``np.rot90(..., k)``."""

    matrices = (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [-1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
    )
    return matrices[_normalise_k(k)].copy()


def transform_shape_xy(shape_xy: Sequence[int], k: int) -> tuple[int, int]:
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    if nx <= 0 or ny <= 0:
        raise ValueError(f"invalid image shape {shape_xy!r}")
    return (ny, nx) if _normalise_k(k) % 2 else (nx, ny)


def apply_quarter_turn_numpy(array: np.ndarray, k: int) -> np.ndarray:
    """Losslessly permute the final Y/X axes using the selected quarter turn."""

    data = np.asarray(array)
    if data.ndim < 2:
        raise ValueError("quarter-turn input must have at least two dimensions")
    return np.ascontiguousarray(np.rot90(data, k=_normalise_k(k), axes=(-2, -1)))


def _polar_rotation_deg(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float).reshape(2, 2)
    u, _, vt = np.linalg.svd(m)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _wrap_180(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def choose_quarter_turn(matrices: np.ndarray) -> int:
    """Choose one common quarter turn minimizing residual polar rotation.

    A single stack permutation must be shared by all tilts.  Candidates are
    ranked by maximum absolute residual rotation, then median absolute residual
    rotation, then by the smallest absolute quarter-turn magnitude.
    """

    mats = np.asarray(matrices, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if mats.shape[1:] != (2, 2):
        raise ValueError(f"expected matrices shaped (N,2,2), got {mats.shape}")

    candidates: list[tuple[tuple[float, float, float], int]] = []
    for k in range(4):
        q = quarter_turn_matrix(k)
        residual = mats @ q.T
        angles = np.asarray([_wrap_180(_polar_rotation_deg(m)) for m in residual])
        quarter_angle = abs(_wrap_180(_polar_rotation_deg(q)))
        score = (
            float(np.max(np.abs(angles))),
            float(np.median(np.abs(angles))),
            quarter_angle,
        )
        candidates.append((score, k))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def factor_affines(
    matrices: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    k: int | None = None,
    tolerance: float = 1e-12,
) -> QuarterTurnFactorization:
    """Factor centred transforms as ``A_original = A_residual @ Q``.

    Shifts remain unchanged because both the original and quarter-turned frames
    use their own geometric image centres.
    """

    mats = np.asarray(matrices, dtype=float)
    shifts = np.asarray(shifts_xy, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if shifts.ndim == 1:
        shifts = shifts[None, ...]
    if mats.shape[1:] != (2, 2) or shifts.shape != (len(mats), 2):
        raise ValueError(
            f"invalid affine arrays: matrices={mats.shape}, shifts={shifts.shape}"
        )
    selected = choose_quarter_turn(mats) if k is None else _normalise_k(k)
    q = quarter_turn_matrix(selected)
    residual = mats @ q.T
    recomposed = residual @ q
    error = float(np.max(np.abs(recomposed - mats)))
    if error > tolerance:
        raise ValueError(
            f"quarter-turn factorization failed: max recomposition error {error}"
        )
    original_angles = np.asarray([_polar_rotation_deg(m) for m in mats], dtype=float)
    residual_angles = np.asarray([_polar_rotation_deg(m) for m in residual], dtype=float)
    return QuarterTurnFactorization(
        k=selected,
        quarter_turn_matrix=q,
        residual_matrices=residual,
        residual_shifts=shifts.copy(),
        original_rotation_deg=original_angles,
        residual_rotation_deg=residual_angles,
        max_recomposition_error=error,
    )
