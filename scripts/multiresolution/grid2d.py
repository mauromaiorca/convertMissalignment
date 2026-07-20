#!/usr/bin/env python3
"""Canonical 2-D pixel-grid representation for multiresolution geometry.

A ``Grid2D`` carries an explicit affine ``Q`` mapping homogeneous pixel
coordinates ``[x, y, 1]`` to a shared physical frame (Angstrom)::

    X = Q @ [x, y, 1]^T

For an axis-aligned grid::

    Q = [[px,  0, ox - px*cx],
         [ 0, py, oy - py*cy],
         [ 0,  0,          1]]

where ``px, py`` are Angstrom/pixel, ``cx, cy`` are the pixel centres, and
``ox, oy`` are the physical coordinates assigned to the grid centre.

Centre convention: ``cx = (nx - 1) / 2``, ``cy = (ny - 1) / 2`` (the verified
IMOD convention; see ``docs/interoperability/COORDINATE_CONVENTIONS.md``). This
is used unless real-``newstack`` characterization proves a specific operation
needs a different mapping, which is then stored explicitly in ``Q``.

The working->source pixel mapping is ``G = inv(Q_source) @ Q_working`` and is
the matrix saved in the resolved manifest. ``Q`` carries pixel SIZE and the
physical-centre offset, so transferring transforms with these matrices is NOT
"scale the translation by the binning factor" -- the ``(B-1)/2``-style centre
offset emerges naturally and is generally non-zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def pixel_center(n: int) -> float:
    """IMOD 0-based pixel centre ``(n-1)/2``."""
    return (float(n) - 1.0) / 2.0


@dataclass
class Grid2D:
    name: str
    shape_xy: tuple[int, int]
    pixel_size_xy_A: tuple[float, float]
    center_xy_px: tuple[float, float]
    physical_origin_xy_A: tuple[float, float] = (0.0, 0.0)
    role: str = ""
    source_file: str | None = None
    axis_basis: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))

    @classmethod
    def axis_aligned(cls, name, shape_xy, pixel_size_A, *, role="", source_file=None,
                     physical_origin_xy_A=(0.0, 0.0)):
        nx, ny = int(shape_xy[0]), int(shape_xy[1])
        if isinstance(pixel_size_A, (int, float)):
            px = py = float(pixel_size_A)
        else:
            px, py = float(pixel_size_A[0]), float(pixel_size_A[1])
        if nx <= 0 or ny <= 0 or px <= 0 or py <= 0:
            raise ValueError(f"{name}: positive shape and pixel size required")
        return cls(name=name, shape_xy=(nx, ny), pixel_size_xy_A=(px, py),
                   center_xy_px=(pixel_center(nx), pixel_center(ny)),
                   physical_origin_xy_A=tuple(map(float, physical_origin_xy_A)),
                   role=role, source_file=source_file)

    @property
    def Q(self) -> np.ndarray:
        px, py = self.pixel_size_xy_A
        cx, cy = self.center_xy_px
        ox, oy = self.physical_origin_xy_A
        b = np.asarray(self.axis_basis, dtype=float)  # rows are pixel-axis directions in physical
        Q = np.eye(3)
        Q[:2, :2] = b.T * np.array([px, py])  # columns scaled by pixel size
        Q[:2, 2] = np.array([ox, oy]) - Q[:2, :2] @ np.array([cx, cy])
        return Q

    @property
    def Q_inv(self) -> np.ndarray:
        return np.linalg.inv(self.Q)

    def pixel_to_physical(self, pts_xy: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_xy, float))
        h = np.column_stack([p, np.ones(len(p))])
        out = (self.Q @ h.T).T[:, :2]
        return out[0] if np.ndim(pts_xy) == 1 else out

    def physical_to_pixel(self, pts_xy: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_xy, float))
        h = np.column_stack([p, np.ones(len(p))])
        out = (self.Q_inv @ h.T).T[:, :2]
        return out[0] if np.ndim(pts_xy) == 1 else out

    def mapping_to(self, other: "Grid2D") -> np.ndarray:
        """Homogeneous 3x3 mapping THIS grid's pixels -> ``other``'s pixels."""
        return other.Q_inv @ self.Q

    def to_dict(self) -> dict:
        return {
            "name": self.name, "role": self.role, "source_file": self.source_file,
            "shape_xy": list(self.shape_xy),
            "pixel_size_xy_A": list(self.pixel_size_xy_A),
            "center_xy_px": list(self.center_xy_px),
            "physical_origin_xy_A": list(self.physical_origin_xy_A),
            "axis_basis": [list(r) for r in self.axis_basis],
            "Q": self.Q.tolist(),
        }


def integer_binned_grid(source: Grid2D, factor: int, out_shape_xy: Sequence[int] | None = None,
                        *, name: str | None = None, role: str = "working") -> Grid2D:
    """Construct the working grid for an isotropic integer binning of ``source``.

    The physical field centre is preserved (same physical origin). The working
    pixel size is ``factor * source pixel size``. The output shape is measured
    from the real reduction when provided (``out_shape_xy``); otherwise the
    nominal ``floor(n / factor)`` is used (and must be reconciled with the
    measured header before trusting geometry).
    """
    if factor not in (2, 4, 8):
        raise ValueError(f"unsupported binning factor {factor!r}; allowed: 2, 4, 8")
    px, py = source.pixel_size_xy_A
    nx, ny = source.shape_xy
    if out_shape_xy is None:
        out_shape_xy = (nx // factor, ny // factor)
    return Grid2D.axis_aligned(
        name or f"{source.name}_bin{factor}",
        (int(out_shape_xy[0]), int(out_shape_xy[1])),
        (px * factor, py * factor),
        role=role,
        physical_origin_xy_A=source.physical_origin_xy_A,
    )
