#!/usr/bin/env python3
"""Transfer of homogeneous transforms and projection geometry between grids.

All transforms are absolute homogeneous matrices in PIXEL/VOXEL coordinates of
their stated grid. Conventions match the rest of the project: a 2-D ``.xf``
maps raw->aligned about the IMOD ``(n-1)/2`` centre, and final composition is
``Hfinal = DeltaH @ H0`` (apply ``H0`` first).

Grid maps (all 3x3 for detector, 4x4 for volume), produced by
``Grid2D.mapping_to`` / ``Grid3D.mapping_to``:

- ``G_r``: working raw detector pixels  -> source raw detector pixels
- ``G_a``: working aligned detector pixels -> source aligned detector pixels
- ``G_d``: working detector pixels -> source detector pixels (for projection)
- ``G_v``: working volume voxels  -> source volume voxels

Derivations are in ``MULTIRESOLUTION_PROJECTION_GEOMETRY.md``.
"""
from __future__ import annotations

import numpy as np


def _h(M, n):
    M = np.asarray(M, float)
    if M.shape != (n, n):
        raise ValueError(f"expected {n}x{n} matrix, got {M.shape}")
    return M


# --- detector-grid transform transfer (raw->aligned H0) --------------------

def h0_working(H0_source: np.ndarray, G_r: np.ndarray, G_a: np.ndarray) -> np.ndarray:
    """``H0_working = inv(G_a) @ H0_source @ G_r``.

    Chain: working raw -> source raw -> source aligned -> working aligned.
    """
    return np.linalg.inv(_h(G_a, 3)) @ _h(H0_source, 3) @ _h(G_r, 3)


def h0_source_from_working(H0_working: np.ndarray, G_r: np.ndarray, G_a: np.ndarray) -> np.ndarray:
    """Inverse of :func:`h0_working`: ``H0_source = G_a @ H0_working @ inv(G_r)``."""
    return _h(G_a, 3) @ _h(H0_working, 3) @ np.linalg.inv(_h(G_r, 3))


# --- residual restore (working aligned residual -> source) -----------------

def deltaH_source(DeltaH_working: np.ndarray, G_a: np.ndarray) -> np.ndarray:
    """``DeltaH_source = G_a @ DeltaH_working @ inv(G_a)`` (conjugation by G_a)."""
    Ga = _h(G_a, 3)
    return Ga @ _h(DeltaH_working, 3) @ np.linalg.inv(Ga)


def hfinal_source(DeltaH_working: np.ndarray, H0_source: np.ndarray, G_a: np.ndarray) -> np.ndarray:
    """``Hfinal_source = DeltaH_source @ H0_source`` (canonical restore)."""
    return deltaH_source(DeltaH_working, G_a) @ _h(H0_source, 3)


def hfinal_source_via_working(H0_working, DeltaH_working, G_a, G_r) -> np.ndarray:
    """Equivalent route: ``Hfinal_source = G_a @ (DeltaH_working @ H0_working) @ inv(G_r)``.

    Numerically equal to :func:`hfinal_source` (proved in the tests).
    """
    Hfinal_working = _h(DeltaH_working, 3) @ _h(H0_working, 3)
    return _h(G_a, 3) @ Hfinal_working @ np.linalg.inv(_h(G_r, 3))


def restore_hfinal_working_to_source(Hfinal_working, G_a, G_r) -> np.ndarray:
    """Restore a complete working-grid raw->final transform to the source grid:
    ``Hfinal_source = G_a @ Hfinal_working @ inv(G_r)``.

    Used for a real affine2d Warp result already expressed as a single working
    raw->final ``.xf`` per tilt (no separate H0/DeltaH split needed). This is the
    complete homogeneous transform -- never a per-column translation rescale.
    """
    return _h(G_a, 3) @ _h(Hfinal_working, 3) @ np.linalg.inv(_h(G_r, 3))


# --- grid-independent physical residual ------------------------------------

def deltaH_physical(DeltaH_working: np.ndarray, Q_aligned_working: np.ndarray) -> np.ndarray:
    """Working-pixel residual -> aligned PHYSICAL frame: ``Q @ DeltaH @ inv(Q)``."""
    Q = _h(Q_aligned_working, 3)
    return Q @ _h(DeltaH_working, 3) @ np.linalg.inv(Q)


def deltaH_export(DeltaH_physical: np.ndarray, Q_aligned_export: np.ndarray) -> np.ndarray:
    """Physical residual -> export-grid pixels: ``inv(Q) @ DeltaH_physical @ Q``."""
    Q = _h(Q_aligned_export, 3)
    return np.linalg.inv(Q) @ _h(DeltaH_physical, 3) @ Q


# --- projection-matrix transfer (3x4) --------------------------------------

def projection_working(P_source: np.ndarray, G_d: np.ndarray, G_v: np.ndarray) -> np.ndarray:
    """``P_working = inv(G_d) @ P_source @ G_v`` (3x4).

    Chain: working voxel(4) -> source voxel(4) -> source detector(3) ->
    working detector(3). ``P_*`` return HOMOGENEOUS detector coordinates
    ``[u, v, w]`` (Euclidean ``= [u/w, v/w]``).
    """
    P = np.asarray(P_source, float)
    if P.shape != (3, 4):
        raise ValueError(f"P_source must be 3x4, got {P.shape}")
    return np.linalg.inv(_h(G_d, 3)) @ P @ _h(G_v, 4)


def project_euclidean(P: np.ndarray, voxels_xyz: np.ndarray) -> np.ndarray:
    """Apply a 3x4 projection to (N,3) voxel coords -> (N,2) Euclidean detector."""
    v = np.atleast_2d(np.asarray(voxels_xyz, float))
    h = np.column_stack([v, np.ones(len(v))])
    d = (np.asarray(P, float) @ h.T).T  # (N,3) homogeneous
    w = d[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("projection produced a zero homogeneous w (degenerate; "
                         "the parallel-beam model keeps w=1 -- check P)")
    eucl = d[:, :2] / w
    return eucl[0] if np.ndim(voxels_xyz) == 1 else eucl
