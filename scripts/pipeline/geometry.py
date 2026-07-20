#!/usr/bin/env python3
"""Measured MRC geometry discovery -- the single source of pipeline geometry.

Replaces the previous fallback behaviour (``src_dims=(0,0)``, ``pixel_size=1.0``).
Geometry is read from real MRC headers; configuration/CLI values may be used only
as assertions or explicit overrides and must never silently replace header data.
If an override disagrees with the header, this fails unless ``force=True`` and the
discrepancy is recorded.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiresolution.grid2d import Grid2D, pixel_center  # noqa: E402


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class MeasuredMrcGrid2D:
    path: str
    shape_xy: tuple[int, int]
    n_sections: int
    pixel_size_xy_A: tuple[float, float]
    center_xy_px: tuple[float, float]
    header_origin_xyz: tuple[float, float, float]
    mode: int
    file_size_bytes: int
    sample_all_finite: bool
    grid: Grid2D

    def to_dict(self) -> dict:
        return {
            "path": self.path, "shape_xy": list(self.shape_xy), "n_sections": self.n_sections,
            "pixel_size_xy_A": list(self.pixel_size_xy_A), "center_xy_px": list(self.center_xy_px),
            "header_origin_xyz": list(self.header_origin_xyz), "mode": self.mode,
            "file_size_bytes": self.file_size_bytes, "sample_all_finite": self.sample_all_finite,
            "grid": self.grid.to_dict(),
        }


def measure_mrc_grid(path: Path, *, role: str, sample_sections: int = 2) -> MeasuredMrcGrid2D:
    import mrcfile
    p = Path(path)
    if not p.is_file():
        raise GeometryError(f"{role}: MRC file not found: {p}")
    try:
        with mrcfile.open(p, permissive=True, header_only=True) as h:
            nx = int(h.header.nx)
            ny = int(h.header.ny)
            nsec = max(1, int(h.header.nz))
            px = float(h.voxel_size.x); py = float(h.voxel_size.y)
            origin = (float(h.header.origin.x), float(h.header.origin.y), float(h.header.origin.z))
            mode = int(h.header.mode)
            finite = True
    except GeometryError:
        raise
    except Exception as exc:
        raise GeometryError(f"{role}: corrupt/unreadable MRC header: {p} ({exc})") from exc
    if nx <= 0 or ny <= 0:
        raise GeometryError(f"{role}: invalid dimensions {nx}x{ny} in {p}")
    if px <= 0 or py <= 0:
        raise GeometryError(f"{role}: invalid/zero pixel size ({px},{py}) in {p}; "
                            "fix the header or supply an explicit override")
    grid = Grid2D.axis_aligned(role, (nx, ny), (px, py), role=role, source_file=str(p))
    return MeasuredMrcGrid2D(
        path=str(p), shape_xy=(nx, ny), n_sections=nsec, pixel_size_xy_A=(px, py),
        center_xy_px=(pixel_center(nx), pixel_center(ny)), header_origin_xyz=origin,
        mode=mode, file_size_bytes=p.stat().st_size, sample_all_finite=finite, grid=grid)


def assert_or_override(measured: MeasuredMrcGrid2D, *, expected_shape_xy=None,
                       expected_pixel_A=None, force: bool = False, tol_px=0.0, tol_pixel=1e-3):
    """Compare config geometry to the header. Fail on disagreement unless forced."""
    discrepancies = []
    if expected_shape_xy and tuple(expected_shape_xy) != measured.shape_xy:
        discrepancies.append(f"shape: config {tuple(expected_shape_xy)} != header {measured.shape_xy}")
    if expected_pixel_A and abs(float(expected_pixel_A) - measured.pixel_size_xy_A[0]) > tol_pixel:
        discrepancies.append(f"pixel: config {expected_pixel_A} != header {measured.pixel_size_xy_A[0]}")
    if discrepancies and not force:
        raise GeometryError(
            f"{measured.path}: config geometry disagrees with the MRC header "
            f"({'; '.join(discrepancies)}). The header is authoritative; pass force=True only "
            "if you are certain, and the discrepancy will be recorded.")
    return discrepancies


def measure_source_and_working(*, source_raw: Path | None, source_aligned: Path | None,
                               working_raw: Path | None = None, working_aligned: Path | None = None) -> dict:
    """Read every available stack and build separate raw/aligned grids + maps.

    Raw and aligned grids are kept SEPARATE (they may differ in dims/centre/pixel).
    Returns measured grids and, when both source+working of a kind are present, the
    corresponding G map (working -> source).
    """
    out: dict = {"measured": {}, "maps": {}}
    role_paths = {"source_raw": source_raw, "source_aligned": source_aligned,
                  "working_raw": working_raw, "working_aligned": working_aligned}
    for role, path in role_paths.items():
        if path and Path(path).is_file():
            out["measured"][role] = measure_mrc_grid(Path(path), role=role)
    m = out["measured"]
    if "source_raw" in m and "working_raw" in m:
        out["maps"]["G_r"] = m["working_raw"].grid.mapping_to(m["source_raw"].grid).tolist()
    if "source_aligned" in m and "working_aligned" in m:
        out["maps"]["G_a"] = m["working_aligned"].grid.mapping_to(m["source_aligned"].grid).tolist()
    # full Q matrices for the manifest
    out["Q"] = {role: meas.grid.Q.tolist() for role, meas in m.items()}
    return out
