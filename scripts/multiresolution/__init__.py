#!/usr/bin/env python3
"""Multiresolution working-data geometry: explicit grids and transform transfer.

Pure-numpy grid algebra (``Grid2D``, ``Grid3D``), homogeneous transform transfer
between source/working/preview/export grids, an analytic parallel-beam
projector, and the residual-restore formulas. The mathematical specification is
``docs/interoperability/MULTIRESOLUTION_MATH.md`` and the derivations in
``MULTIRESOLUTION_PROJECTION_GEOMETRY.md``.
"""
from __future__ import annotations

from . import projector, restore, transfer, workflow
from .grid2d import Grid2D, integer_binned_grid, pixel_center
from .grid3d import Grid3D, preview_grid_from
from .workflow import MultiresError, MultiresPlan, build_plan, validate_request

__all__ = [
    "Grid2D", "Grid3D", "integer_binned_grid", "preview_grid_from",
    "pixel_center", "transfer", "projector", "restore", "workflow",
    "MultiresError", "MultiresPlan", "build_plan", "validate_request",
]
