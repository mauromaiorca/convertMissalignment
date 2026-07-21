#!/usr/bin/env python3
"""Canonical revised-IMOD-alignment geometry (source-aware round trip).

    original IMOD  ->  Warp representation  ->  MissAlignment refinement  ->  revised IMOD

Both MissAlignment result backends (``constrained_json`` and ``warp_xml``) converge
into ONE typed object here, :class:`ImodAlignmentRevision`, BEFORE any IMOD file is
written. The writers and reports consume only that object, so there is a single
geometry convention.

Composition (the one canonical rule)::

    H_final_i = DeltaH_i @ H_original_i

  * ``H_original_i``  original IMOD raw->aligned transform (the imported ``.xf`` row)
  * ``DeltaH_i``      MissAlignment correction expressed in the aligned-image frame
                      (an aligned->aligned map)
  * ``H_final_i``     revised raw->aligned transform written to ``<series>.xf``

All matrix composition uses the repository's validated IMOD centre convention
(``imod_affine.image_center_xy(..., "imod") == (n-1)/2``, empirically matched to
``newstack``) via :func:`imod_affine.compose_xf`. The residual ``DeltaH`` is exported
separately (``<series>.residual.xf``); it is diagnostic and must never be referenced
as the complete raw->aligned transform.

Representability: each per-tilt refined detector mapping is sampled on a grid and an
affine is least-squares fitted. If the fit residual exceeds tolerance the tilt is
``non_affine`` and, under the ``fail`` policy, the export refuses rather than writing a
misleading ``.xf``.

Pure NumPy: no warpylib/torch/IMOD/WarpTools, so every geometry decision is testable
off-cluster.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import (  # noqa: E402
    compose_xf, diagnose_matrix, fit_affine, forward_points_pixels,
    image_center_xy, regular_grid_points, residual_statistics,
)

# Bump when the revised-alignment writer/report contract changes shape or semantics.
REVISION_CONTRACT_VERSION = 1

REPRESENTABILITY_CLASSES = ("exact_affine", "affine_within_tolerance", "non_affine")


# --------------------------------------------------------------------------- #
# policy (mirrors [export.imod_revision])
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RevisionPolicy:
    """Resolved [export.imod_revision] policy that governs the writers."""

    enabled: bool = True
    output_root: str = "exported_data/imod"
    mode: str = "compose_with_original"
    non_affine_policy: str = "fail"                  # 'fail' | 'warn' | 'allow'
    affine_fit_rms_tolerance_px: float = 0.10
    affine_fit_max_tolerance_px: float = 0.25
    angle_representation: str = "preserve_original_decomposition"
    global_positioning_policy: str = "preserve_unless_refined"
    shift_recovery_policy: str = "provenance_or_constrained_fit"
    thickness_policy: str = "preserve_original"
    write_residual_xf: bool = True
    run_imod_reconstruction_validation: bool = True
    overwrite_source: bool = False

    @classmethod
    def from_config(cls, table: Optional[dict]) -> "RevisionPolicy":
        t = dict(table or {})
        base = cls()
        return cls(
            enabled=bool(t.get("enabled", base.enabled)),
            output_root=str(t.get("output_root", base.output_root)),
            mode=str(t.get("mode", base.mode)),
            non_affine_policy=str(t.get("non_affine_policy", base.non_affine_policy)),
            affine_fit_rms_tolerance_px=float(
                t.get("affine_fit_rms_tolerance_px", base.affine_fit_rms_tolerance_px)),
            affine_fit_max_tolerance_px=float(
                t.get("affine_fit_max_tolerance_px", base.affine_fit_max_tolerance_px)),
            angle_representation=str(t.get("angle_representation", base.angle_representation)),
            global_positioning_policy=str(
                t.get("global_positioning_policy", base.global_positioning_policy)),
            shift_recovery_policy=str(t.get("shift_recovery_policy", base.shift_recovery_policy)),
            thickness_policy=str(t.get("thickness_policy", base.thickness_policy)),
            write_residual_xf=bool(t.get("write_residual_xf", base.write_residual_xf)),
            run_imod_reconstruction_validation=bool(
                t.get("run_imod_reconstruction_validation",
                      base.run_imod_reconstruction_validation)),
            overwrite_source=bool(t.get("overwrite_source", base.overwrite_source)),
        )

    def to_manifest(self) -> dict:
        return {
            "enabled": self.enabled, "output_root": self.output_root, "mode": self.mode,
            "non_affine_policy": self.non_affine_policy,
            "affine_fit_rms_tolerance_px": self.affine_fit_rms_tolerance_px,
            "affine_fit_max_tolerance_px": self.affine_fit_max_tolerance_px,
            "angle_representation": self.angle_representation,
            "global_positioning_policy": self.global_positioning_policy,
            "shift_recovery_policy": self.shift_recovery_policy,
            "thickness_policy": self.thickness_policy,
            "write_residual_xf": self.write_residual_xf,
            "run_imod_reconstruction_validation": self.run_imod_reconstruction_validation,
            "overwrite_source": self.overwrite_source,
        }

    def policy_hash_fields(self) -> dict:
        """The policy fields that must invalidate the export cache when changed."""
        return {
            "mode": self.mode, "non_affine_policy": self.non_affine_policy,
            "affine_fit_rms_tolerance_px": round(self.affine_fit_rms_tolerance_px, 6),
            "affine_fit_max_tolerance_px": round(self.affine_fit_max_tolerance_px, 6),
            "angle_representation": self.angle_representation,
            "global_positioning_policy": self.global_positioning_policy,
            "shift_recovery_policy": self.shift_recovery_policy,
            "thickness_policy": self.thickness_policy,
            "write_residual_xf": self.write_residual_xf,
        }


class RevisionError(ValueError):
    """Raised when the refined geometry cannot be represented within policy."""


# --------------------------------------------------------------------------- #
# typed geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Affine2D:
    """A centred IMOD 2-D affine: aligned = matrix @ (raw - c_in) + shift + c_out."""

    matrix: np.ndarray          # (2, 2)
    shift: np.ndarray           # (2,)

    def __post_init__(self):
        object.__setattr__(self, "matrix", np.asarray(self.matrix, dtype=float).reshape(2, 2))
        object.__setattr__(self, "shift", np.asarray(self.shift, dtype=float).reshape(2))

    @classmethod
    def identity(cls) -> "Affine2D":
        return cls(np.eye(2), np.zeros(2))

    @classmethod
    def from_row(cls, row: Sequence[float]) -> "Affine2D":
        row = np.asarray(row, dtype=float).reshape(-1)
        if row.size < 6:
            raise ValueError("an IMOD .xf row needs 6 values")
        return cls(row[:4].reshape(2, 2), row[4:6])

    def to_row(self) -> list[float]:
        return [*self.matrix.reshape(-1).tolist(), *self.shift.tolist()]

    @property
    def is_translation_only(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2), atol=1e-9))


@dataclass(frozen=True)
class OriginalImodGeometry:
    """The imported original IMOD alignment for one series."""

    series: str
    raw_shape_xy: tuple[int, int]
    aligned_shape_xy: tuple[int, int]
    raw_pixel_size_A: float
    aligned_pixel_size_A: float
    original_transforms: list[Affine2D]           # per-tilt raw->aligned .xf
    original_tilt_angles_deg: list[float]         # raw .tlt (before OFFSET)
    x_axis_tilt_per_view_deg: Optional[list[float]] = None   # .xtilt, if present


@dataclass(frozen=True)
class RefinedWarpGeometry:
    """The MissAlignment/Warp refinement, as per-tilt sampled detector correspondences.

    For every tilt, ``sample_points_xy`` are absolute aligned-frame detector pixel
    coordinates and ``refined_points_xy`` are where the refinement maps them. The
    (points -> refined points) map is the aligned-frame correction ``DeltaH``. Both
    result backends produce this by sampling their representation; keeping the canonical
    input as point correspondences makes the affine fit and the non-affine test uniform.
    """

    backend: str                                  # 'constrained_json' | 'warp_xml'
    sample_points_xy: list[np.ndarray]            # per tilt (M, 2)
    refined_points_xy: list[np.ndarray]           # per tilt (M, 2)
    included: list[bool]
    acquisition_index: Optional[list[int]] = None
    revised_tilt_angles_deg: Optional[list[float]] = None
    revised_x_axis_tilt_per_view_deg: Optional[list[float]] = None


@dataclass(frozen=True)
class RepresentabilityReport:
    tilt_class: list[str]                          # per tilt in REPRESENTABILITY_CLASSES
    rms_residual_px: list[float]
    max_residual_px: list[float]
    rms_residual_A: list[float]
    max_residual_A: list[float]
    sampling_grid_xy: tuple[int, int]
    image_dimensions_xy: tuple[int, int]

    @property
    def worst_class(self) -> str:
        for cls in ("non_affine", "affine_within_tolerance", "exact_affine"):
            if cls in self.tilt_class:
                return cls
        return "exact_affine"

    @property
    def has_non_affine(self) -> bool:
        return "non_affine" in self.tilt_class

    def to_manifest(self) -> dict:
        return {
            "worst_class": self.worst_class,
            "per_tilt_class": list(self.tilt_class),
            "rms_residual_px": [round(v, 6) for v in self.rms_residual_px],
            "max_residual_px": [round(v, 6) for v in self.max_residual_px],
            "rms_residual_A": [round(v, 6) for v in self.rms_residual_A],
            "max_residual_A": [round(v, 6) for v in self.max_residual_A],
            "sampling_grid_xy": list(self.sampling_grid_xy),
            "image_dimensions_xy": list(self.image_dimensions_xy),
        }


@dataclass(frozen=True)
class ImodAlignmentRevision:
    """The single typed revision object every writer/report consumes."""

    original_geometry: OriginalImodGeometry
    refined_geometry: RefinedWarpGeometry

    residual_transforms: list[Affine2D]            # DeltaH per tilt (aligned frame)
    final_transforms: list[Affine2D]               # H_final per tilt (raw->aligned)

    original_tilt_angles_deg: list[float]
    revised_tilt_angles_deg: list[float]

    original_positioning: dict                     # [geometry.imod_positioning] table
    revised_positioning: dict

    representability: RepresentabilityReport
    provenance: dict = field(default_factory=dict)

    @property
    def series(self) -> str:
        return self.original_geometry.series

    @property
    def n_tilts(self) -> int:
        return len(self.final_transforms)


# --------------------------------------------------------------------------- #
# composition + representability
# --------------------------------------------------------------------------- #
def compose_final_transform(original: Affine2D, delta: Affine2D, *,
                            raw_shape_xy, aligned_shape_xy) -> Affine2D:
    """H_final = DeltaH @ H_original, in the IMOD centre convention.

    ``original`` maps raw->aligned; ``delta`` maps aligned->aligned. The composed
    transform maps raw->aligned. Uses imod_affine.compose_xf, which returns
    homogeneous_to_xf(H_delta @ H_original) about the validated ('imod') centre.
    """
    matrix, shift = compose_xf(
        original.matrix, original.shift,      # first: raw -> aligned
        delta.matrix, delta.shift,            # second: aligned -> aligned
        input_shape_xy=raw_shape_xy,
        intermediate_shape_xy=aligned_shape_xy,
        output_shape_xy=aligned_shape_xy,
        center_convention="imod",
    )
    return Affine2D(matrix, shift)


def fit_delta_and_classify(sample_points_xy: np.ndarray, refined_points_xy: np.ndarray, *,
                           aligned_shape_xy, aligned_pixel_size_A: float,
                           policy: RevisionPolicy) -> tuple[Affine2D, dict]:
    """Fit an aligned-frame affine ``DeltaH`` to (sample -> refined) and classify it.

    Returns the fitted ``Affine2D`` (absolute-pixel affine converted to a centred IMOD
    transform) and a diagnostics dict with residual stats + representability class.
    """
    sample = np.asarray(sample_points_xy, dtype=float).reshape(-1, 2)
    refined = np.asarray(refined_points_xy, dtype=float).reshape(-1, 2)
    if sample.shape != refined.shape or len(sample) < 3:
        raise RevisionError("need >=3 matched sample/refined points to fit DeltaH")

    # Absolute-pixel affine: refined = A_abs @ sample + d_abs (aligned->aligned).
    a_abs, d_abs, residuals = fit_affine(sample, refined)
    stats = residual_statistics(residuals)
    rms_px, max_px = stats["rms"], stats["max"]

    if rms_px <= 1e-9 and max_px <= 1e-9:
        cls = "exact_affine"
    elif rms_px <= policy.affine_fit_rms_tolerance_px and max_px <= policy.affine_fit_max_tolerance_px:
        cls = "affine_within_tolerance"
    else:
        cls = "non_affine"

    # Convert the absolute-pixel affine (same input/output = aligned frame) into a
    # centred IMOD transform so it composes with the original .xf. For an equal in/out
    # shape and the 'imod' centre, the centred shift is d_abs + (A_abs - I) @ c.
    c = image_center_xy(aligned_shape_xy, "imod")
    centred_shift = d_abs + (a_abs - np.eye(2)) @ c
    delta = Affine2D(a_abs, centred_shift)

    diagnostics = {
        "class": cls,
        "rms_residual_px": float(rms_px), "max_residual_px": float(max_px),
        "rms_residual_A": float(rms_px * aligned_pixel_size_A),
        "max_residual_A": float(max_px * aligned_pixel_size_A),
    }
    return delta, diagnostics


def build_revision(original: OriginalImodGeometry, refined: RefinedWarpGeometry, *,
                   policy: RevisionPolicy, sampling_grid_xy: tuple[int, int] = (7, 7),
                   original_positioning: Optional[dict] = None,
                   revised_positioning: Optional[dict] = None,
                   provenance: Optional[dict] = None) -> ImodAlignmentRevision:
    """Converge a backend's refined geometry into the canonical revision object.

    Fits per-tilt ``DeltaH``, classifies representability, composes ``H_final =
    DeltaH @ H_original`` and preserves the angle decomposition. Under
    ``non_affine_policy == 'fail'`` a single non-affine tilt aborts the export rather
    than writing a misleading ``.xf``.
    """
    n = len(original.original_transforms)
    if not (len(refined.sample_points_xy) == len(refined.refined_points_xy) == n):
        raise RevisionError(
            f"tilt count mismatch: {n} original transforms vs "
            f"{len(refined.sample_points_xy)} refined mappings")

    residual_transforms: list[Affine2D] = []
    final_transforms: list[Affine2D] = []
    tilt_class: list[str] = []
    rms_px: list[float] = []
    max_px: list[float] = []
    rms_A: list[float] = []
    max_A: list[float] = []

    for i in range(n):
        delta, diag = fit_delta_and_classify(
            refined.sample_points_xy[i], refined.refined_points_xy[i],
            aligned_shape_xy=original.aligned_shape_xy,
            aligned_pixel_size_A=original.aligned_pixel_size_A, policy=policy)
        final = compose_final_transform(
            original.original_transforms[i], delta,
            raw_shape_xy=original.raw_shape_xy, aligned_shape_xy=original.aligned_shape_xy)
        residual_transforms.append(delta)
        final_transforms.append(final)
        tilt_class.append(diag["class"])
        rms_px.append(diag["rms_residual_px"]); max_px.append(diag["max_residual_px"])
        rms_A.append(diag["rms_residual_A"]); max_A.append(diag["max_residual_A"])

    representability = RepresentabilityReport(
        tilt_class=tilt_class, rms_residual_px=rms_px, max_residual_px=max_px,
        rms_residual_A=rms_A, max_residual_A=max_A,
        sampling_grid_xy=sampling_grid_xy, image_dimensions_xy=original.aligned_shape_xy)

    if representability.has_non_affine and policy.non_affine_policy == "fail":
        bad = [i for i, c in enumerate(tilt_class) if c == "non_affine"]
        raise RevisionError(
            f"{len(bad)} tilt(s) {bad} are non-affine beyond tolerance "
            f"(rms>{policy.affine_fit_rms_tolerance_px}px or max>"
            f"{policy.affine_fit_max_tolerance_px}px); non_affine_policy='fail' refuses to "
            "write a misleading .xf. Use a grid-preserving exporter or relax the policy.")

    revised_angles = (list(refined.revised_tilt_angles_deg)
                      if refined.revised_tilt_angles_deg is not None
                      else list(original.original_tilt_angles_deg))

    return ImodAlignmentRevision(
        original_geometry=original, refined_geometry=refined,
        residual_transforms=residual_transforms, final_transforms=final_transforms,
        original_tilt_angles_deg=list(original.original_tilt_angles_deg),
        revised_tilt_angles_deg=revised_angles,
        original_positioning=dict(original_positioning or {}),
        revised_positioning=dict(revised_positioning or {}),
        representability=representability, provenance=dict(provenance or {}))


def converge_revision(original: OriginalImodGeometry, deltas: list[Affine2D], *,
                      policy: RevisionPolicy, backend: str,
                      representability_stats: Optional[list[dict]] = None,
                      included: Optional[list[bool]] = None,
                      acquisition_index: Optional[list[int]] = None,
                      revised_tilt_angles_deg: Optional[list[float]] = None,
                      revised_x_axis_tilt_per_view_deg: Optional[list[float]] = None,
                      sampling_grid_xy: tuple[int, int] = (7, 7),
                      original_positioning: Optional[dict] = None,
                      revised_positioning: Optional[dict] = None,
                      provenance: Optional[dict] = None) -> ImodAlignmentRevision:
    """Converge a backend that already produced per-tilt aligned-frame ``DeltaH`` affines.

    Both result backends land here after fitting: ``constrained_json`` (closed-form affine,
    exact) and ``warp_xml`` (grid fit, whose per-tilt residual stats are passed via
    ``representability_stats`` = [{"rms_residual_px", "max_residual_px"}, ...]). The
    canonical ``H_final = DeltaH @ H_original`` composition still happens here, so every
    backend shares one geometry convention. The point-correspondence path
    :func:`build_revision` remains for callers that want the fit measured directly.
    """
    n = len(original.original_transforms)
    if len(deltas) != n:
        raise RevisionError(f"tilt count mismatch: {n} originals vs {len(deltas)} deltas")
    included = list(included) if included is not None else [True] * n
    stats = representability_stats or [{"rms_residual_px": 0.0, "max_residual_px": 0.0}] * n

    final_transforms, tilt_class = [], []
    rms_px, max_px, rms_A, max_A = [], [], [], []
    px2A = float(original.aligned_pixel_size_A)
    for i in range(n):
        final_transforms.append(compose_final_transform(
            original.original_transforms[i], deltas[i],
            raw_shape_xy=original.raw_shape_xy, aligned_shape_xy=original.aligned_shape_xy))
        r = float(stats[i].get("rms_residual_px", 0.0))
        m = float(stats[i].get("max_residual_px", 0.0))
        if r <= 1e-9 and m <= 1e-9:
            cls = "exact_affine"
        elif r <= policy.affine_fit_rms_tolerance_px and m <= policy.affine_fit_max_tolerance_px:
            cls = "affine_within_tolerance"
        else:
            cls = "non_affine"
        tilt_class.append(cls)
        rms_px.append(r); max_px.append(m); rms_A.append(r * px2A); max_A.append(m * px2A)

    representability = RepresentabilityReport(
        tilt_class=tilt_class, rms_residual_px=rms_px, max_residual_px=max_px,
        rms_residual_A=rms_A, max_residual_A=max_A,
        sampling_grid_xy=sampling_grid_xy, image_dimensions_xy=original.aligned_shape_xy)
    if representability.has_non_affine and policy.non_affine_policy == "fail":
        bad = [i for i, c in enumerate(tilt_class) if c == "non_affine"]
        raise RevisionError(
            f"{len(bad)} tilt(s) {bad} are non-affine beyond tolerance; "
            "non_affine_policy='fail' refuses to write a misleading .xf")

    refined = RefinedWarpGeometry(
        backend=backend,
        sample_points_xy=[np.empty((0, 2))] * n, refined_points_xy=[np.empty((0, 2))] * n,
        included=included, acquisition_index=acquisition_index,
        revised_tilt_angles_deg=revised_tilt_angles_deg,
        revised_x_axis_tilt_per_view_deg=revised_x_axis_tilt_per_view_deg)
    revised_angles = (list(revised_tilt_angles_deg) if revised_tilt_angles_deg is not None
                      else list(original.original_tilt_angles_deg))
    return ImodAlignmentRevision(
        original_geometry=original, refined_geometry=refined,
        residual_transforms=list(deltas), final_transforms=final_transforms,
        original_tilt_angles_deg=list(original.original_tilt_angles_deg),
        revised_tilt_angles_deg=revised_angles,
        original_positioning=dict(original_positioning or {}),
        revised_positioning=dict(revised_positioning or {}),
        representability=representability, provenance=dict(provenance or {}))


def sample_affine_correspondences(delta: Affine2D, shape_xy, *, nx=7, ny=7) -> tuple:
    """Sample (grid, delta(grid)) so a pre-fitted aligned-frame affine can enter
    build_revision through the same point-correspondence path as a real backend."""
    grid = regular_grid_points(shape_xy, nx=nx, ny=ny)
    refined = forward_points_pixels(grid, delta.matrix, delta.shift, shape_xy, shape_xy, "imod")
    return grid, refined


# --------------------------------------------------------------------------- #
# per-tilt physical-effect metrics (detector-grid displacement, not just coeffs)
# --------------------------------------------------------------------------- #
def _grid_points(shape_xy, nx=9, ny=7) -> np.ndarray:
    return regular_grid_points(shape_xy, nx=nx, ny=ny)


def tilt_change_metrics(original: Affine2D, final: Affine2D, delta: Affine2D, *,
                        raw_shape_xy, aligned_shape_xy, aligned_pixel_size_A: float,
                        grid_nx: int = 9, grid_ny: int = 7) -> dict:
    """Measure the physical effect of the correction by sampling detector points.

    Displacement is where a raw grid point lands under the FINAL vs the ORIGINAL
    raw->aligned transform (aligned-frame pixels, converted to Angstrom). Also returns
    the DeltaH matrix diagnostics (rotation/scale/shear/determinant).
    """
    raw_pts = _grid_points(raw_shape_xy, grid_nx, grid_ny)
    orig_out = forward_points_pixels(raw_pts, original.matrix, original.shift,
                                     raw_shape_xy, aligned_shape_xy, "imod")
    final_out = forward_points_pixels(raw_pts, final.matrix, final.shift,
                                      raw_shape_xy, aligned_shape_xy, "imod")
    disp = final_out - orig_out                       # aligned-frame pixels
    norms = np.linalg.norm(disp, axis=1)
    px2A = float(aligned_pixel_size_A)

    # centre displacement
    c_raw = image_center_xy(raw_shape_xy, "imod")[None, :]
    c_orig = forward_points_pixels(c_raw, original.matrix, original.shift,
                                   raw_shape_xy, aligned_shape_xy, "imod")[0]
    c_final = forward_points_pixels(c_raw, final.matrix, final.shift,
                                    raw_shape_xy, aligned_shape_xy, "imod")[0]
    centre_disp = c_final - c_orig

    # corners of the raw image
    W, H = float(raw_shape_xy[0]), float(raw_shape_xy[1])
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=float)
    corner_o = forward_points_pixels(corners, original.matrix, original.shift,
                                     raw_shape_xy, aligned_shape_xy, "imod")
    corner_f = forward_points_pixels(corners, final.matrix, final.shift,
                                     raw_shape_xy, aligned_shape_xy, "imod")
    corner_disp = np.linalg.norm(corner_f - corner_o, axis=1)

    diag = diagnose_matrix(delta.matrix)
    return {
        "centre_displacement_px": [float(centre_disp[0]), float(centre_disp[1])],
        "centre_displacement_A": [float(centre_disp[0] * px2A), float(centre_disp[1] * px2A)],
        "mean_displacement_px": float(np.mean(norms)),
        "rms_displacement_px": float(np.sqrt(np.mean(norms ** 2))),
        "p95_displacement_px": float(np.percentile(norms, 95)),
        "max_displacement_px": float(np.max(norms)),
        "mean_displacement_A": float(np.mean(norms) * px2A),
        "rms_displacement_A": float(np.sqrt(np.mean(norms ** 2)) * px2A),
        "max_displacement_A": float(np.max(norms) * px2A),
        "corner_displacement_px": [float(v) for v in corner_disp],
        "residual_rotation_deg": float(diag.rotation_deg),
        "scale_x": float(delta.matrix[0, 0]),
        "scale_y": float(delta.matrix[1, 1]),
        "isotropic_scale": float(np.sqrt(abs(diag.determinant))),
        "shear": float(diag.shear_offdiag),
        "determinant": float(diag.determinant),
    }
