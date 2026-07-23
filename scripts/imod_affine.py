#!/usr/bin/env python3
"""Mathematics for converting IMOD 2-D affine transforms to/from Warp geometry.

Coordinate conventions
----------------------
An IMOD ``.xf`` row is interpreted as a *forward* centered transform from the
raw input image to the aligned output image::

    a_centered = A @ r_centered + d

where ``A`` is formed from the first four values in row-major order and ``d``
is the final two values in pixels.  The absolute-coordinate version is::

    a_abs = c_out + A @ (r_abs - c_in) + d

Warp/MissAlignment samples the raw image from a projected coordinate expressed
in the aligned frame.  Consequently, the raw path needs the inverse map::

    r_centered_A = B @ a_centered_A + b

with physical coordinates in Angstroms.  For raw/aligned pixel sizes ``p_r``
and ``p_a``::

    B = (p_r / p_a) * inv(A)
    b = -p_r * inv(A) @ d

The module is intentionally independent of warpylib so that the mathematical
round-trip tests can run with only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

CenterConvention = Literal["imod", "pixel-center", "size-half"]

# Bump when the Warp TiltAxisAngle / OFFSET representation changes. v3: per-view axis is
# extracted from Warp's OFFICIAL .xf layout -- the rotation built from VecX=(A11,A21),
# VecY=(A12,A22) (i.e. A.T), fed to EulerFromMatrix(...).Z. For the near-conformal tomo2
# matrices this equals degrees(atan2(A12, A11)) (~+95.5), the SAME .xf convention (A.T) the
# offset conversion already uses. v2 read the axis from the raw IMOD layout atan2(A21, A11)
# (~-95.5) and v1 added +180 (~+84.5); both are the wrong side of 90 deg and reverse the
# tilt-axis direction (turning `/` into `\`). OFFSET is baked into Angles with LevelAngleY = 0.
# A marker without this version (fixed 84.1, the +180 branch, or the -95.5 IMOD-layout branch)
# is stale.
WARP_AXIS_ANGLE_CONVENTION_VERSION = 3


@dataclass(frozen=True)
class AffineDiagnostics:
    rotation_deg: float
    scale_major: float
    scale_minor: float
    anisotropy_ratio: float
    determinant: float
    shear_offdiag: float
    orthogonality_error: float
    condition_number: float


def _as_shape_xy(shape_xy: Sequence[int | float]) -> np.ndarray:
    shape = np.asarray(shape_xy, dtype=float)
    if shape.shape != (2,) or np.any(shape <= 0):
        raise ValueError(f"expected positive (X,Y) shape, got {shape_xy!r}")
    return shape


def image_center_xy(
    shape_xy: Sequence[int | float], convention: CenterConvention = "imod"
) -> np.ndarray:
    """Return the image centre in 0-based pixel coordinates.

    ``imod`` (the workflow default) uses ``((nx-1)/2, (ny-1)/2)``.  This was
    verified empirically against the installed IMOD ``newstack`` (Phase 4,
    IMOD 5.1.9): newstack applies a ``.xf`` about the geometric image centre,
    which is ``(nx+1)/2`` in IMOD's 1-based pixel coordinates, i.e.
    ``(nx-1)/2`` in 0-based coordinates.  A 25-degree calibration rotation
    matched this convention to < 0.005 px RMS while ``nx/2`` was off by
    ~0.3 px; see ``tests/test_imod_center_convention.py``.

    ``pixel-center`` is an explicit alias of the same ``(nx-1)/2`` convention.
    ``size-half`` returns ``nx/2``; it does NOT match newstack and is retained
    only for diagnostics and backward comparison.
    """

    shape = _as_shape_xy(shape_xy)
    if convention in ("imod", "pixel-center"):
        return (shape - 1.0) / 2.0
    if convention == "size-half":
        return shape / 2.0
    raise ValueError(f"unknown centre convention: {convention}")


def read_xf(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an IMOD ``.xf`` file as ``A[N,2,2]`` and ``d[N,2]``."""

    table = np.loadtxt(Path(path), dtype=float, ndmin=2)
    if table.ndim != 2 or table.shape[1] < 6:
        raise ValueError(f"{path}: expected at least 6 columns, got {table.shape}")
    table = table[:, :6]
    matrices = table[:, :4].reshape((-1, 2, 2))
    shifts = table[:, 4:6].copy()
    determinants = np.linalg.det(matrices)
    if np.any(~np.isfinite(table)):
        raise ValueError(f"{path}: non-finite transform values")
    if np.any(np.abs(determinants) < 1e-10):
        bad = np.where(np.abs(determinants) < 1e-10)[0].tolist()
        raise ValueError(f"{path}: singular/near-singular transforms at rows {bad}")
    return matrices, shifts


def write_xf(path: str | Path, matrices: np.ndarray, shifts: np.ndarray) -> None:
    """Write matrices and shifts in IMOD six-column format."""

    matrices = np.asarray(matrices, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if shifts.ndim == 1:
        shifts = shifts[None, ...]
    if matrices.shape[1:] != (2, 2) or shifts.shape != (len(matrices), 2):
        raise ValueError(
            f"incompatible shapes: matrices={matrices.shape}, shifts={shifts.shape}"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for matrix, shift in zip(matrices, shifts, strict=True):
            values = [*matrix.reshape(-1), *shift]
            handle.write(
                f"{values[0]:12.7f}{values[1]:12.7f}"
                f"{values[2]:12.7f}{values[3]:12.7f}"
                f"{values[4]:12.3f}{values[5]:12.3f}\n"
            )


def forward_points_pixels(
    points_xy: np.ndarray,
    matrix: np.ndarray,
    shift_xy: np.ndarray,
    input_shape_xy: Sequence[int | float],
    output_shape_xy: Sequence[int | float] | None = None,
    center_convention: CenterConvention = "imod",
) -> np.ndarray:
    """Apply an IMOD raw→aligned transform to absolute pixel coordinates."""

    points = np.asarray(points_xy, dtype=float)
    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    shift = np.asarray(shift_xy, dtype=float).reshape(2)
    if output_shape_xy is None:
        output_shape_xy = input_shape_xy
    c_in = image_center_xy(input_shape_xy, center_convention)
    c_out = image_center_xy(output_shape_xy, center_convention)
    return (points - c_in) @ matrix.T + shift + c_out


def inverse_points_pixels(
    points_xy: np.ndarray,
    matrix: np.ndarray,
    shift_xy: np.ndarray,
    input_shape_xy: Sequence[int | float],
    output_shape_xy: Sequence[int | float] | None = None,
    center_convention: CenterConvention = "imod",
) -> np.ndarray:
    """Map aligned absolute pixel coordinates back to raw input coordinates."""

    points = np.asarray(points_xy, dtype=float)
    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    shift = np.asarray(shift_xy, dtype=float).reshape(2)
    if output_shape_xy is None:
        output_shape_xy = input_shape_xy
    c_in = image_center_xy(input_shape_xy, center_convention)
    c_out = image_center_xy(output_shape_xy, center_convention)
    inv = np.linalg.inv(matrix)
    return (points - c_out - shift) @ inv.T + c_in


def inverse_physical_map(
    matrix: np.ndarray,
    shift_xy_px: np.ndarray,
    raw_pixel_size_A: float,
    aligned_pixel_size_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``B,b`` for aligned-centred Å → raw-centred Å."""

    if raw_pixel_size_A <= 0 or aligned_pixel_size_A <= 0:
        raise ValueError("pixel sizes must be positive")
    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    shift = np.asarray(shift_xy_px, dtype=float).reshape(2)
    inv = np.linalg.inv(matrix)
    b = -float(raw_pixel_size_A) * (inv @ shift)
    b_matrix = (float(raw_pixel_size_A) / float(aligned_pixel_size_A)) * inv
    return b_matrix, b


def forward_physical_map(
    matrix: np.ndarray,
    shift_xy_px: np.ndarray,
    raw_pixel_size_A: float,
    aligned_pixel_size_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``M,t`` for raw-centred Å → aligned-centred Å."""

    if raw_pixel_size_A <= 0 or aligned_pixel_size_A <= 0:
        raise ValueError("pixel sizes must be positive")
    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    shift = np.asarray(shift_xy_px, dtype=float).reshape(2)
    m = (float(aligned_pixel_size_A) / float(raw_pixel_size_A)) * matrix
    t = float(aligned_pixel_size_A) * shift
    return m, t


def transform_axis_angle_raw_to_aligned(
    raw_angle_deg: float, matrix: np.ndarray
) -> float:
    """Transform an in-plane raw-frame axis direction into the aligned frame.

    The angle is treated as a standard mathematical image-plane direction with
    vector ``(cos(phi), sin(phi))``.  The same numerical convention is used on
    output; a cluster integration test validates the sign/zero convention
    against the installed warpylib/IMOD versions.
    """

    angle = np.deg2rad(float(raw_angle_deg))
    vector_raw = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    vector_aligned = np.asarray(matrix, dtype=float).reshape(2, 2) @ vector_raw
    norm = np.linalg.norm(vector_aligned)
    if norm < 1e-12:
        raise ValueError("axis direction collapsed by transform")
    vector_aligned /= norm
    result = np.rad2deg(np.arctan2(vector_aligned[1], vector_aligned[0]))
    # Keep angles in a stable [-180, 180) interval.
    return float((result + 180.0) % 360.0 - 180.0)


def movement_at_raw_absolute_physical(
    raw_absolute_xy_A: np.ndarray,
    inverse_matrix_physical: np.ndarray,
    inverse_shift_physical_A: np.ndarray,
    raw_image_dimensions_physical_A: Sequence[float],
) -> np.ndarray:
    """Movement that makes Warp sample the full inverse IMOD affine.

    Warp computes ``z = raw_center + aligned_centered + offset`` and then samples
    at ``z - movement(z)``.  With ``offset=b`` the required movement is::

        movement(z) = (I - B) @ (z - raw_center - b)
    """

    z = np.asarray(raw_absolute_xy_A, dtype=float)
    b_matrix = np.asarray(inverse_matrix_physical, dtype=float).reshape(2, 2)
    b = np.asarray(inverse_shift_physical_A, dtype=float).reshape(2)
    raw_dims = _as_shape_xy(raw_image_dimensions_physical_A)
    raw_center = raw_dims / 2.0
    q = z - raw_center - b
    return q @ (np.eye(2) - b_matrix).T


def build_movement_grid_values(
    matrices: np.ndarray,
    shifts_xy_px: np.ndarray,
    raw_shape_xy: Sequence[int],
    raw_pixel_size_A: float,
    aligned_pixel_size_A: float,
    grid_shape_xy: Sequence[int] = (5, 5),
    grid_image_shape_xy: Sequence[int] | None = None,
    grid_image_pixel_size_A: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Warp/einspline movement-grid values for the full inverse affine.

    Returns ``movement_x``, ``movement_y`` flattened in Warp's
    ``[(tilt*Y+y)*X+x]`` layout, plus per-tilt offsets ``b[N,2]`` in Å.
    """

    matrices = np.asarray(matrices, dtype=float)
    shifts = np.asarray(shifts_xy_px, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if shifts.ndim == 1:
        shifts = shifts[None, ...]
    if matrices.shape[1:] != (2, 2) or shifts.shape != (len(matrices), 2):
        raise ValueError("invalid matrix/shift array shapes")
    nxg, nyg = (int(grid_shape_xy[0]), int(grid_shape_xy[1]))
    if nxg < 2 or nyg < 2:
        raise ValueError("movement grid must have at least 2 nodes in X and Y")

    raw_shape = _as_shape_xy(grid_image_shape_xy or raw_shape_xy)
    image_pixel = float(grid_image_pixel_size_A or raw_pixel_size_A)
    raw_dims_A = raw_shape * image_pixel
    values_x = np.zeros(nxg * nyg * len(matrices), dtype=np.float32)
    values_y = np.zeros_like(values_x)
    offsets = np.zeros((len(matrices), 2), dtype=float)

    index = 0
    for tilt, (matrix, shift) in enumerate(zip(matrices, shifts, strict=True)):
        inverse_matrix, inverse_shift = inverse_physical_map(
            matrix, shift, raw_pixel_size_A, aligned_pixel_size_A
        )
        offsets[tilt] = inverse_shift
        for gy in range(nyg):
            fy = gy / (nyg - 1)
            for gx in range(nxg):
                fx = gx / (nxg - 1)
                z_abs = np.array([fx * raw_dims_A[0], fy * raw_dims_A[1]])
                movement = movement_at_raw_absolute_physical(
                    z_abs, inverse_matrix, inverse_shift, raw_dims_A
                )
                values_x[index] = movement[0]
                values_y[index] = movement[1]
                index += 1
    return values_x, values_y, offsets


def evaluate_inverse_affine_from_warp_components(
    aligned_centered_xy_A: np.ndarray,
    offset_xy_A: np.ndarray,
    movement_function,
    raw_image_dimensions_physical_A: Sequence[float],
) -> np.ndarray:
    """Evaluate Warp's offset/movement order for independent tests."""

    q = np.asarray(aligned_centered_xy_A, dtype=float)
    offset = np.asarray(offset_xy_A, dtype=float).reshape(2)
    center = _as_shape_xy(raw_image_dimensions_physical_A) / 2.0
    z_abs = q + center + offset
    movement = np.asarray(movement_function(z_abs), dtype=float)
    return z_abs - movement - center


def xf_to_homogeneous(
    matrix: np.ndarray,
    shift_xy: np.ndarray,
    input_shape_xy: Sequence[int | float],
    output_shape_xy: Sequence[int | float],
    center_convention: CenterConvention = "imod",
) -> np.ndarray:
    """Convert a centred IMOD transform into a 3×3 absolute-pixel matrix."""

    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    shift = np.asarray(shift_xy, dtype=float).reshape(2)
    c_in = image_center_xy(input_shape_xy, center_convention)
    c_out = image_center_xy(output_shape_xy, center_convention)
    translation = c_out + shift - matrix @ c_in
    result = np.eye(3, dtype=float)
    result[:2, :2] = matrix
    result[:2, 2] = translation
    return result


def homogeneous_to_xf(
    homogeneous: np.ndarray,
    input_shape_xy: Sequence[int | float],
    output_shape_xy: Sequence[int | float],
    center_convention: CenterConvention = "imod",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an absolute-pixel 3×3 affine matrix to centred IMOD form."""

    h = np.asarray(homogeneous, dtype=float).reshape(3, 3)
    if not np.allclose(h[2], [0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("matrix is not a 2-D affine homogeneous transform")
    matrix = h[:2, :2].copy()
    c_in = image_center_xy(input_shape_xy, center_convention)
    c_out = image_center_xy(output_shape_xy, center_convention)
    shift = h[:2, 2] - c_out + matrix @ c_in
    return matrix, shift


def compose_xf(
    first_matrix: np.ndarray,
    first_shift: np.ndarray,
    second_matrix: np.ndarray,
    second_shift: np.ndarray,
    input_shape_xy: Sequence[int | float],
    intermediate_shape_xy: Sequence[int | float],
    output_shape_xy: Sequence[int | float],
    center_convention: CenterConvention = "imod",
) -> tuple[np.ndarray, np.ndarray]:
    """Compose raw→intermediate and intermediate→output transforms."""

    h_first = xf_to_homogeneous(
        first_matrix,
        first_shift,
        input_shape_xy,
        intermediate_shape_xy,
        center_convention,
    )
    h_second = xf_to_homogeneous(
        second_matrix,
        second_shift,
        intermediate_shape_xy,
        output_shape_xy,
        center_convention,
    )
    return homogeneous_to_xf(
        h_second @ h_first,
        input_shape_xy,
        output_shape_xy,
        center_convention,
    )


def fit_affine(
    input_points_xy: np.ndarray, output_points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit ``output = A @ input + d``.

    Returns ``A``, ``d`` and per-point residual vectors.
    """

    x = np.asarray(input_points_xy, dtype=float)
    y = np.asarray(output_points_xy, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(x) < 3:
        raise ValueError("input/output point arrays must both have shape (N,2), N>=3")
    design = np.column_stack([x, np.ones(len(x))])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    matrix = coefficients[:2, :].T
    shift = coefficients[2, :]
    predicted = x @ matrix.T + shift
    residuals = y - predicted
    return matrix, shift, residuals


def residual_statistics(residuals_xy: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(residuals_xy, dtype=float)
    norms = np.linalg.norm(residuals, axis=-1)
    return {
        "rms": float(np.sqrt(np.mean(norms**2))),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p95": float(np.percentile(norms, 95)),
        "max": float(np.max(norms)),
        "rms_x": float(np.sqrt(np.mean(residuals[..., 0] ** 2))),
        "rms_y": float(np.sqrt(np.mean(residuals[..., 1] ** 2))),
    }


def diagnose_matrix(matrix: np.ndarray) -> AffineDiagnostics:
    """Return rotation/scale/shear diagnostics using the polar decomposition."""

    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    u, singular_values, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        singular_values[-1] *= -1
        rotation = u @ vt
    stretch = rotation.T @ matrix
    rotation_deg = np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
    abs_singular = np.abs(singular_values)
    major = float(np.max(abs_singular))
    minor = float(np.min(abs_singular))
    return AffineDiagnostics(
        rotation_deg=float(rotation_deg),
        scale_major=major,
        scale_minor=minor,
        anisotropy_ratio=float(major / minor) if minor > 0 else float("inf"),
        determinant=float(np.linalg.det(matrix)),
        shear_offdiag=float(0.5 * (abs(stretch[0, 1]) + abs(stretch[1, 0]))),
        orthogonality_error=float(np.linalg.norm(matrix.T @ matrix - np.eye(2))),
        condition_number=float(np.linalg.cond(matrix)),
    )


def imod_xf_rotation_angle_deg(matrix: np.ndarray) -> float:
    """In-plane rotation of an IMOD ``.xf`` linear matrix via polar decomposition (deg).

    Uses the raw IMOD row layout ``A = [[A11, A12], [A21, A22]]`` -> ``atan2(A21, A11)``.
    Scale-unbiased: the isotropic scale does not bias the angle. This is the project's
    validated extraction (same SVD/atan2 as :func:`diagnose_matrix`)."""
    return float(diagnose_matrix(matrix).rotation_deg)


def warp_axis_layout_matrix(matrix: np.ndarray) -> np.ndarray:
    """Warp's OFFICIAL ``.xf`` axis-extraction layout: ``A.T`` = ``[[A11, A21], [A12, A22]]``.

    Warp's ``TiltSeries.ImportAlignments`` builds its rotation from the vectors
    ``VecX = (A11, A21)`` and ``VecY = (A12, A22)`` and feeds that matrix to
    ``EulerFromMatrix(...).Z``. Given the IMOD row matrix ``A = [[A11, A12], [A21, A22]]`` that
    arrangement is exactly ``A.T``. Extracting the axis from THIS layout (not the raw IMOD
    layout) is what puts the tomo2 axis on the +95.5 side of 90 deg -- and it uses the same
    ``A.T`` convention the ``.xf`` offset conversion already uses, so axis and offset are
    finally derived from one canonical matrix."""
    a = np.asarray(matrix, dtype=float).reshape(2, 2)
    return np.ascontiguousarray(a.T)


def warp_axis_angle_from_xf_layout(matrix: np.ndarray) -> float:
    """Warp ``TiltAxisAngle`` (deg) from the OFFICIAL ``A.T`` layout, scale-unbiased.

    Polar rotation of :func:`warp_axis_layout_matrix`; for the near-conformal tomo2 matrices
    this equals ``degrees(atan2(A12, A11))`` (e.g. +95.478 for the first real row). The simple
    ``atan2`` form is confirmed to agree with this polar form for all rows in the regression
    tests; the installed Warp ``EulerFromMatrix`` is the authority validated cluster-side."""
    return imod_xf_rotation_angle_deg(warp_axis_layout_matrix(matrix))


def warp_tilt_axis_angle_from_xf(
    matrix: np.ndarray,
    *,
    angle_sign: int = -1,
    reference_angle_deg: float = 84.1,
) -> tuple[float, float, float]:
    """Per-view Warp ``TiltAxisAngle`` from the source ``.xf``, in Warp's OFFICIAL layout.

    The axis is ``EulerFromMatrix`` of the ``A.T`` layout (``VecX=(A11,A21)``,
    ``VecY=(A12,A22)``), assigned straight to ``TiltAxisAngles`` -- matching Warp's official
    ``TiltSeries.ImportAlignments``. For tomo2 this gives ~+95.5. There is NO +180 adjustment and
    NO branch normalisation to the align.com estimate: the +180 branch (~+84.5) and the raw
    IMOD-layout polar branch (~-95.5) are BOTH the wrong side of 90 deg and reverse the tilt-axis
    direction (turning `/` into `\\`). ``angle_sign``/``reference_angle_deg`` are retained for
    provenance only. Returns ``(warp_axis_deg, imod_layout_axis_deg, adjustment_deg=0)`` where the
    second element is the raw IMOD-layout polar angle (~-95.5), recorded for provenance."""
    warp_axis = warp_axis_angle_from_xf_layout(matrix)
    imod_layout_axis = imod_xf_rotation_angle_deg(matrix)
    return float(warp_axis), float(imod_layout_axis), 0.0


def regular_grid_points(
    shape_xy: Sequence[int | float], nx: int = 17, ny: int = 13
) -> np.ndarray:
    """Absolute pixel coordinates spanning an image, including its boundaries."""

    shape = _as_shape_xy(shape_xy)
    if nx < 2 or ny < 2:
        raise ValueError("grid dimensions must be >=2")
    xs = np.linspace(0.0, shape[0], nx)
    ys = np.linspace(0.0, shape[1], ny)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


def rows_from_affines(matrices: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None]
    if shifts.ndim == 1:
        shifts = shifts[None]
    return np.column_stack([matrices.reshape(len(matrices), 4), shifts])


__all__ = [
    "AffineDiagnostics",
    "build_movement_grid_values",
    "compose_xf",
    "diagnose_matrix",
    "evaluate_inverse_affine_from_warp_components",
    "fit_affine",
    "forward_physical_map",
    "forward_points_pixels",
    "homogeneous_to_xf",
    "image_center_xy",
    "imod_xf_rotation_angle_deg",
    "inverse_physical_map",
    "warp_axis_layout_matrix",
    "warp_axis_angle_from_xf_layout",
    "warp_tilt_axis_angle_from_xf",
    "inverse_points_pixels",
    "movement_at_raw_absolute_physical",
    "read_xf",
    "regular_grid_points",
    "residual_statistics",
    "rows_from_affines",
    "transform_axis_angle_raw_to_aligned",
    "write_xf",
    "xf_to_homogeneous",
]
