#!/usr/bin/env python3
"""Transform composition: ``Hfinal_i = DeltaH_i @ H0_i``.

``H0_i`` is the initial raw->aligned transform implied by the *initial
condition*; ``DeltaH_i`` is the residual produced by the *refinement model*.
All matrices are absolute-physical homogeneous 3x3 (Angstrom) so composition is
plain matrix multiplication and is centre/pixel-size agnostic.

Initial conditions (independent of the refinement model):

- ``raw_identity``        : ``H0 = I``.
- ``raw_xf_translation``  : ``H0`` = IMOD ``.xf`` translation only (``A = I``).
- ``raw_xf_affine_fixed`` : ``H0`` = full IMOD ``.xf`` affine.
- ``ali_identity``        : ``H0`` = full IMOD ``.xf`` affine (already in the
  ``.ali`` pixels); ``DeltaH`` then acts ali->final.

The order ``DeltaH @ H0`` (apply ``H0`` first, then ``DeltaH``) is canonical and
must never silently become ``H0 @ DeltaH``; regression tests enforce this.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import coordinate_frames as cf

RAW_CONDITIONS = ("raw_identity", "raw_xf_translation", "raw_xf_affine_fixed")
ALL_CONDITIONS = RAW_CONDITIONS + ("ali_identity",)


def initial_homogeneous_per_tilt(
    condition: str,
    matrices_xf: np.ndarray | None,
    shifts_xf_px: np.ndarray | None,
    raw_shape_xy: Sequence[int],
    aligned_shape_xy: Sequence[int],
    p_raw_A: float,
    p_aligned_A: float,
) -> np.ndarray:
    """Per-tilt ``H0`` (absolute-physical homogeneous), raw->aligned."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"unknown initial condition {condition!r}")
    if condition == "raw_identity":
        # Number of tilts must come from the xf if present, else require it.
        if matrices_xf is None:
            raise ValueError("raw_identity needs the tilt count via matrices_xf")
        n = len(np.atleast_3d(np.asarray(matrices_xf)).reshape(-1, 2, 2))
        return np.stack([np.eye(3) for _ in range(n)])

    A = np.asarray(matrices_xf, dtype=float).reshape(-1, 2, 2)
    d = np.asarray(shifts_xf_px, dtype=float).reshape(-1, 2)
    if len(A) != len(d):
        raise ValueError("matrix/shift count mismatch")
    out = []
    for Ai, di in zip(A, d):
        if condition == "raw_xf_translation":
            Ai = np.eye(2)  # translation-only initial condition
        H = cf.imod_xf_to_abs_physical(
            Ai, di, raw_shape_xy, aligned_shape_xy, p_raw_A, p_aligned_A
        )
        out.append(H)
    return np.stack(out)


def residual_homogeneous_per_tilt(
    model, params, aligned_shape_xy: Sequence[int], p_aligned_A: float
) -> np.ndarray:
    """Per-tilt ``DeltaH`` (absolute-physical homogeneous), aligned->final."""
    n = model.as_tensor(params).shape[0]
    center = cf.physical_center_xy(aligned_shape_xy, p_aligned_A)
    centers = np.tile(center, (n, 1))
    H = model.homogeneous_physical(params, centers)
    return H.detach().cpu().numpy()


def compose_final_per_tilt(H0: np.ndarray, deltaH: np.ndarray) -> np.ndarray:
    """``Hfinal_i = DeltaH_i @ H0_i`` per tilt."""
    H0 = np.asarray(H0, dtype=float)
    deltaH = np.asarray(deltaH, dtype=float)
    if H0.shape != deltaH.shape or H0.shape[1:] != (3, 3):
        raise ValueError(f"shape mismatch H0={H0.shape} deltaH={deltaH.shape}")
    return np.stack([cf.compose_homogeneous(h0, dh) for h0, dh in zip(H0, deltaH)])
