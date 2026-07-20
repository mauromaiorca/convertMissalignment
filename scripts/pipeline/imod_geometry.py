#!/usr/bin/env python3
"""Authoritative target-reconstruction geometry resolution (§4).

Reuses the parsing logic in ``01_extract_etomo_params.py`` (loaded as a library)
rather than reimplementing it. The target volume is resolved by an explicit
precedence and is NEVER the raw-detector voxel count at the aligned/output pixel.

Precedence (§4):
1. explicit CLI/TOML override (shape + pixel);
2. source reconstruction MRC header (X,Y,Z + voxel) — header-only, no data read;
3. parsed ``tilt.com`` THICKNESS/IMAGEBINNED + aligned detector shape;
4. aligned detector shape + parsed reconstruction thickness;
5. hard failure.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional


class GeometryResolutionError(ValueError):
    pass


def _load_extract01():
    """Load 01_extract_etomo_params.py as a library (filename is not a module name)."""
    p = Path(__file__).resolve().parents[1] / "01_extract_etomo_params.py"
    spec = importlib.util.spec_from_file_location("extract01_lib", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass decorator needs this registered
    spec.loader.exec_module(mod)
    return mod


def _read_header_nxyz_voxel(path: Path):
    """Header-only read of an MRC: (nx, ny, nz), voxel_size_A. No data array load."""
    import mrcfile
    with mrcfile.open(path, permissive=True, header_only=True) as h:
        nxyz = (int(h.header.nx), int(h.header.ny), int(h.header.nz))
        vox = float(h.voxel_size.x)
    return nxyz, vox


def resolve_target_geometry(*, reconstruction_path: Optional[str],
                            tilt_com_path: Optional[str], newst_com_path: Optional[str],
                            imod_dir: Optional[str], mdoc_path: Optional[str],
                            aligned_shape_xyz: Optional[list], aligned_pixel_A: Optional[float],
                            raw_pixel_A: Optional[float],
                            override_shape_xyz: Optional[list] = None,
                            override_pixel_A: Optional[float] = None,
                            tol: float = 1e-3) -> dict:
    """Return IMOD reconstruction MRC storage geometry. ``shape_xyz`` is MRC
    X,Y,Z with reconstruction thickness in MRC Y. Raises on failure."""
    ext = _load_extract01()
    scalars = {}
    if imod_dir:
        try:
            scalars = ext.parse_imod_scalars(Path(imod_dir),
                                             Path(mdoc_path) if mdoc_path else None)
        except Exception:
            scalars = {}

    shape = None
    pixel = None
    source = None

    # 1. explicit override
    if override_shape_xyz and override_pixel_A:
        shape = [int(x) for x in override_shape_xyz]
        pixel = float(override_pixel_A)
        source = "explicit override (CLI/TOML)"

    # 2. reconstruction MRC header (header-only)
    if shape is None and reconstruction_path and Path(reconstruction_path).is_file():
        try:
            nxyz, vox = _read_header_nxyz_voxel(Path(reconstruction_path))
            shape = list(nxyz)
            pixel = float(override_pixel_A or vox or aligned_pixel_A or 0.0) or vox
            source = f"reconstruction header {Path(reconstruction_path).name}"
        except Exception as exc:
            raise GeometryResolutionError(
                f"could not read reconstruction header {reconstruction_path}: {exc}")

    # 3/4. aligned shape + tilt.com THICKNESS (binned by IMAGEBINNED)
    if shape is None and aligned_shape_xyz:
        thickness = scalars.get("thickness_unbinned_px_from_tilt")
        binned = scalars.get("image_binned_from_tilt")
        if thickness is not None:
            z = ext.safe_round_int(thickness / binned) if binned else int(thickness)
            shape = [int(aligned_shape_xyz[0]), int(z), int(aligned_shape_xyz[1])]
            pixel = float(override_pixel_A or aligned_pixel_A or 0.0)
            source = "aligned detector shape + tilt.com THICKNESS"

    if shape is None or not pixel:
        raise GeometryResolutionError(
            "target reconstruction geometry could not be resolved (no override, no readable "
            "reconstruction header, and no tilt.com THICKNESS). Provide a reconstruction stack, "
            "tilt.com, or an explicit [geometry.target_volume] shape+pixel. Refusing to fall back "
            "to raw/aligned detector voxel counts.")

    physical = [round(s * pixel, 4) for s in shape]
    result = {
        "shape_xyz": [int(s) for s in shape],
        "shape_frame": "imod_reconstruction_mrc_xyz__y_is_thickness",
        "pixel_size_A": float(pixel),
        "physical_size_A": physical,
        "source": source,
    }
    assert_physical_invariant(result, tol=tol)
    return result


def assert_physical_invariant(target: dict, *, tol: float = 1e-3) -> None:
    """physical_size_A == shape_xyz * pixel_size_A (§4)."""
    shape = target["shape_xyz"]; pix = target["pixel_size_A"]; phys = target["physical_size_A"]
    for s, p in zip(shape, phys):
        if abs(s * pix - p) > max(tol, tol * abs(s * pix)):
            raise GeometryResolutionError(
                f"target physical invariant violated: {phys} != {shape} x {pix}")
