#!/usr/bin/env python3
"""Canonical 3-D voxel-grid representation for multiresolution geometry.

``X = Q @ [x, y, z, 1]^T`` with::

    Q = [[px,  0,  0, ox - px*cx],
         [ 0, py,  0, oy - py*cy],
         [ 0,  0, pz, oz - pz*cz],
         [ 0,  0,  0,           1]]

Centre convention ``c = (n - 1) / 2`` per axis. The working->source voxel
mapping is ``G = inv(Q_source) @ Q_working``. A preview volume MUST carry its
own ``Grid3D`` (its geometry is measured from the real ``binvol`` header, never
inferred from a label such as ``xy2``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .grid2d import pixel_center


@dataclass
class Grid3D:
    name: str
    shape_xyz: tuple[int, int, int]
    voxel_size_xyz_A: tuple[float, float, float]
    center_xyz_vox: tuple[float, float, float]
    physical_origin_xyz_A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    role: str = ""
    source_file: str | None = None
    anisotropic: bool = False

    @classmethod
    def axis_aligned(cls, name, shape_xyz, voxel_size_A, *, role="", source_file=None,
                     physical_origin_xyz_A=(0.0, 0.0, 0.0)):
        nx, ny, nz = (int(v) for v in shape_xyz)
        if isinstance(voxel_size_A, (int, float)):
            px = py = pz = float(voxel_size_A)
        else:
            px, py, pz = (float(v) for v in voxel_size_A)
        if min(nx, ny, nz) <= 0 or min(px, py, pz) <= 0:
            raise ValueError(f"{name}: positive shape and voxel size required")
        aniso = not (abs(px - py) < 1e-9 and abs(py - pz) < 1e-9)
        return cls(name=name, shape_xyz=(nx, ny, nz), voxel_size_xyz_A=(px, py, pz),
                   center_xyz_vox=(pixel_center(nx), pixel_center(ny), pixel_center(nz)),
                   physical_origin_xyz_A=tuple(map(float, physical_origin_xyz_A)),
                   role=role, source_file=source_file, anisotropic=aniso)

    @property
    def Q(self) -> np.ndarray:
        px, py, pz = self.voxel_size_xyz_A
        cx, cy, cz = self.center_xyz_vox
        o = np.asarray(self.physical_origin_xyz_A, float)
        Q = np.eye(4)
        Q[0, 0], Q[1, 1], Q[2, 2] = px, py, pz
        Q[:3, 3] = o - np.array([px * cx, py * cy, pz * cz])
        return Q

    @property
    def Q_inv(self) -> np.ndarray:
        return np.linalg.inv(self.Q)

    def voxel_to_physical(self, pts_xyz: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_xyz, float))
        h = np.column_stack([p, np.ones(len(p))])
        out = (self.Q @ h.T).T[:, :3]
        return out[0] if np.ndim(pts_xyz) == 1 else out

    def physical_to_voxel(self, pts_xyz: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_xyz, float))
        h = np.column_stack([p, np.ones(len(p))])
        out = (self.Q_inv @ h.T).T[:, :3]
        return out[0] if np.ndim(pts_xyz) == 1 else out

    def mapping_to(self, other: "Grid3D") -> np.ndarray:
        """Homogeneous 4x4 mapping THIS grid's voxels -> ``other``'s voxels."""
        return other.Q_inv @ self.Q

    def to_dict(self) -> dict:
        return {
            "name": self.name, "role": self.role, "source_file": self.source_file,
            "shape_xyz": list(self.shape_xyz),
            "voxel_size_xyz_A": list(self.voxel_size_xyz_A),
            "center_xyz_vox": list(self.center_xyz_vox),
            "physical_origin_xyz_A": list(self.physical_origin_xyz_A),
            "anisotropic": self.anisotropic,
            "Q": self.Q.tolist(),
        }


def preview_grid_from(working: Grid3D, xb: int, yb: int, zb: int, *,
                      out_shape_xyz: Sequence[int] | None = None, name: str | None = None) -> Grid3D:
    """Build a preview ``Grid3D`` for X/Y(/Z) binning of a working volume.

    Geometry should be reconciled with the real ``binvol`` header; when measured
    dims are supplied they are used verbatim.
    """
    px, py, pz = working.voxel_size_xyz_A
    nx, ny, nz = working.shape_xyz
    if out_shape_xyz is None:
        out_shape_xyz = (nx // xb, ny // yb, nz // zb)
    g = Grid3D.axis_aligned(
        name or f"{working.name}_preview_x{xb}y{yb}z{zb}",
        out_shape_xyz, (px * xb, py * yb, pz * zb),
        role="visualization_only",
        physical_origin_xyz_A=working.physical_origin_xyz_A,
    )
    return g
