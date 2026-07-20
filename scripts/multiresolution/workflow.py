#!/usr/bin/env python3
"""Orchestration for the multiresolution working-data workflow.

Validates the binning request (empirically-grounded divisibility gate), builds
all explicit grids, drives real IMOD reductions, converts the source ``.xf`` to
the working grid, generates the working aligned stack and local IMOD
reconstruction command files, optionally produces a binvol preview, and emits a
fully-resolved manifest carrying every grid matrix and source<->working map.

Source data is never modified. The CLI must REJECT unsupported combinations
(non-integer / non-{2,4,8} / non-divisible / anisotropic projection binning)
rather than silently approximate them (see
``MULTIRESOLUTION_NEWSTACK_CHARACTERIZATION.json``).

IMAGEBINNED strategy: the canonical choice is ``explicit_working_geometry`` --
all reconstruction geometry is expressed in the working grid and
``IMAGEBINNED 1`` is used, which structurally avoids the double-scaling hazard
of feeding source-scale geometry together with ``IMAGEBINNED B``. See
``docs/interoperability/WORKING_IMOD_RECONSTRUCTION.md``.
"""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import homogeneous_to_xf, read_xf, write_xf, xf_to_homogeneous  # noqa: E402

from . import transfer as T
from .grid2d import Grid2D, integer_binned_grid
from .grid3d import Grid3D, preview_grid_from

SUPPORTED_FACTORS = (2, 4, 8)
GEOMETRY_STRATEGIES = ("explicit_working_geometry", "source_geometry_with_imagebinned")


class MultiresError(ValueError):
    """Raised for unsupported / unsafe multiresolution requests (never approximate)."""


def validate_request(factor, source_shape_xy, *, projection_method="antialiased",
                     anisotropic=False, axis_permutation=False, reflect=False):
    f = factor
    if isinstance(f, float) and not f.is_integer():
        raise MultiresError(f"binning factor must be an integer; got {factor}")
    f = int(f)
    if f not in SUPPORTED_FACTORS:
        raise MultiresError(f"unsupported binning factor {factor}; allowed: {SUPPORTED_FACTORS}")
    if anisotropic:
        raise MultiresError("anisotropic projection binning is not supported (isotropic X/Y only)")
    if axis_permutation or reflect:
        raise MultiresError("axis permutation / reflection during reduction is not supported")
    nx, ny = int(source_shape_xy[0]), int(source_shape_xy[1])
    if nx % f != 0 or ny % f != 0:
        raise MultiresError(
            f"source dimensions {nx}x{ny} are not divisible by {f}. Real newstack centring "
            f"matches the (B-1)/2 grid model to <0.05 px only for divisible dimensions; "
            f"non-divisible reductions deviate (~0.3 px). Crop the source to a multiple of {f} "
            f"or choose a different factor. This is rejected, not approximated."
        )
    if projection_method not in ("antialiased", "binned"):
        raise MultiresError(f"unknown projection_method {projection_method!r}")
    return f


@dataclass
class MultiresPlan:
    factor: int
    source_raw: Grid2D
    source_aligned: Grid2D
    working_raw: Grid2D
    working_aligned: Grid2D
    G_r: np.ndarray
    G_a: np.ndarray
    geometry_strategy: str = "explicit_working_geometry"
    warnings: list = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "factor": self.factor,
            "geometry_strategy": self.geometry_strategy,
            "grids": {
                "source_raw": self.source_raw.to_dict(),
                "source_aligned": self.source_aligned.to_dict(),
                "working_raw": self.working_raw.to_dict(),
                "working_aligned": self.working_aligned.to_dict(),
            },
            "maps": {
                "G_r_working_raw_to_source_raw": self.G_r.tolist(),
                "G_a_working_aligned_to_source_aligned": self.G_a.tolist(),
                "G_source_raw_to_working_raw": np.linalg.inv(self.G_r).tolist(),
            },
            "warnings": self.warnings,
        }


def build_plan(factor, source_raw: Grid2D, source_aligned: Grid2D | None = None,
               *, working_raw_shape=None, working_aligned_shape=None,
               geometry_strategy="explicit_working_geometry") -> MultiresPlan:
    """Build the grid plan. Measured working shapes (from real newstack headers)
    should be supplied; otherwise the nominal ``n // factor`` is used."""
    f = validate_request(factor, source_raw.shape_xy)
    if geometry_strategy not in GEOMETRY_STRATEGIES:
        raise MultiresError(f"unknown geometry_strategy {geometry_strategy!r}")
    source_aligned = source_aligned or Grid2D.axis_aligned(
        "source_aligned", source_raw.shape_xy, source_raw.pixel_size_xy_A, role="source_aligned")
    wr = integer_binned_grid(source_raw, f, out_shape_xy=working_raw_shape,
                             name="working_raw", role="working_raw")
    wa = integer_binned_grid(source_aligned, f, out_shape_xy=working_aligned_shape,
                             name="working_aligned", role="working_aligned")
    return MultiresPlan(
        factor=f, source_raw=source_raw, source_aligned=source_aligned,
        working_raw=wr, working_aligned=wa,
        G_r=wr.mapping_to(source_raw), G_a=wa.mapping_to(source_aligned),
        geometry_strategy=geometry_strategy,
    )


def convert_source_xf_to_working(source_xf: Path, plan: MultiresPlan, out_xf: Path) -> None:
    """Convert every source raw->aligned .xf row to the working grid via
    ``H0_working = inv(G_a) @ H0_source @ G_r`` (NOT translation scaling)."""
    A0, d0 = read_xf(source_xf)
    Aw, dw = [], []
    for i in range(len(A0)):
        H0 = xf_to_homogeneous(A0[i], d0[i], plan.source_raw.shape_xy, plan.source_aligned.shape_xy)
        H0w = T.h0_working(H0, plan.G_r, plan.G_a)
        a, d = homogeneous_to_xf(H0w, plan.working_raw.shape_xy, plan.working_aligned.shape_xy)
        Aw.append(a); dw.append(d)
    write_xf(out_xf, np.stack(Aw), np.stack(dw))


# --- real-IMOD command construction (text; executed by the generated scripts) ---

def newstack_working_raw_cmd(source_st: Path, out_st: Path, factor: int, method="antialiased") -> list[str]:
    flag = ["-shrink", str(float(factor))] if method == "antialiased" else ["-bin", str(factor)]
    return ["newstack", "-input", str(source_st), "-output", str(out_st), *flag, "-float", "0"]


def newstack_working_aligned_onepass_cmd(source_st: Path, source_xf: Path, out_st: Path, factor: int) -> list[str]:
    """Canonical one-pass: apply the source .xf and shrink in a single newstack
    (one interpolation; geometrically verified against the grid model)."""
    return ["newstack", "-input", str(source_st), "-output", str(out_st),
            "-xform", str(source_xf), "-shrink", str(float(factor)), "-float", "0"]


def working_z_sampling(physical_thickness_A: float, working_pixel_A: float,
                       isotropic: bool = True, z_pixel_A: float | None = None):
    """Compute the working Z voxel count from physical thickness (NOT from the
    projection binning factor). Returns (nz, pz, physical_extent).

    With ``isotropic`` the Z voxel equals the working detector pixel; otherwise
    an explicit ``z_pixel_A`` may be supplied.
    """
    pz = working_pixel_A if isotropic else float(z_pixel_A if z_pixel_A else working_pixel_A)
    nz = int(round(physical_thickness_A / pz))
    return nz, pz, nz * pz


def tilt_working_com(*, in_stack: str, out_rec: str, tilt_file: str, working: Grid2D,
                     nz: int, geometry_strategy="explicit_working_geometry",
                     image_binned_source: int = 1) -> str:
    """Generate a local tilt.com for the working reconstruction.

    ``explicit_working_geometry``: FULLIMAGE and THICKNESS are in WORKING pixels,
    ``IMAGEBINNED 1`` (no internal scaling -> no double scaling).
    ``source_geometry_with_imagebinned``: kept for comparison only; FULLIMAGE and
    THICKNESS in SOURCE pixels with ``IMAGEBINNED B``.
    """
    nx, ny = working.shape_xy
    if geometry_strategy == "explicit_working_geometry":
        imagebinned = 1
        fullimage = f"{nx} {ny}"
        thickness = nz
    else:
        f = image_binned_source
        imagebinned = f
        fullimage = f"{nx * f} {ny * f}"
        thickness = nz * f
    return (
        f"# Local working reconstruction tilt.com (geometry_strategy={geometry_strategy}).\n"
        f"# Generated; never edit or overwrite the source tilt.com.\n"
        f"$tilt\n"
        f"InputProjections {in_stack}\n"
        f"OutputFile {out_rec}\n"
        f"TILTFILE {tilt_file}\n"
        f"FULLIMAGE {fullimage}\n"
        f"THICKNESS {thickness}\n"
        f"IMAGEBINNED {imagebinned}\n"
        f"RADIAL 0.35 0.05\n"
        f"$if (-e ./savework) ./savework\n"
    )


def reconstruction_run_script(tilt_com: str, out_rec: str, module_mode="auto", imod_module="imod") -> str:
    q = shlex.quote
    return f"""#!/usr/bin/env bash
set -euo pipefail
# Generated working reconstruction runner. Uses local command files only;
# never overwrites the source .ali/.rec.
if [ "{module_mode}" != "never" ] && ! command -v submfg >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then module load {q(imod_module)}; fi
fi
command -v submfg >/dev/null 2>&1 || {{ echo "ERROR: submfg not on PATH" >&2; exit 127; }}
[ -e {q(out_rec)} ] && {{ echo "ERROR: refusing to overwrite {out_rec}" >&2; exit 1; }}
submfg {q(tilt_com)}
echo "Wrote working reconstruction: {out_rec}"
"""


def binvol_preview_cmd(working_rec: Path, out_rec: Path, xb=2, yb=2, zb=1) -> list[str]:
    return ["binvol", "-x", str(xb), "-y", str(yb), "-z", str(zb), str(working_rec), str(out_rec)]
