"""Canonical IMOD tomogram-positioning geometry from ``tilt.com``.

IMOD's ``tilt`` program positions the reconstruction with four parameters that the
previous pipeline parsed only partially (``THICKNESS``) or not at all:

    THICKNESS  <n>              reconstruction thickness, unbinned pixels
    OFFSET     <deg>            global tilt-angle offset added to every view
    XAXISTILT  <deg>            tilt of the reconstruction about the X (tilt) axis
    SHIFT      <sx> <sz>        reconstruction shift in X and Z, unbinned pixels

This module is the single authority for parsing, representing, converting and
hashing those values. It has **no** ``warpylib``/``torch`` dependency, so the
parsing, the canonical structure and the numerical IMOD projection oracle can be
tested off-cluster. The Warp *application* (level angles / tomogram shift) lives in
``etomo_to_warp.py`` and consumes the resolved structure produced here.

Coordinate conventions used throughout (documented, not assumed):

  IMOD reconstruction/specimen frame (right-handed):
      X = tilt axis
      Y = in-plane, perpendicular to the tilt axis on the specimen
      Z = beam / thickness direction
  A view at signed tilt angle ``theta`` is the specimen rotated about the tilt
  axis... in IMOD's ``tilt`` the projection geometry is equivalent to rotating the
  specimen about the X (tilt) axis is NOT correct: ``tilt`` tilts about the axis
  that in the *aligned stack* is vertical. We model the forward projection so that
  the detector coordinate perpendicular to the tilt axis is
      u(theta) = x_perp * cos(theta) + z * sin(theta)
  and the coordinate along the tilt axis is v = (tilt-axis coordinate). This is the
  standard single-axis weighted-back-projection forward model and is what the
  oracle below encodes and tests.

Sign conventions are validated numerically by the oracle and its tests, never
inferred from field names. See ``IMOD_TO_WARP_SIGN_NOTES``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Bump when the parsed/resolved positioning contract changes shape or semantics.
# v2: IMOD SHIFT is transformed into the Warp volume frame with the signed IMOD_MRC_TO_WARP
# orientation matrix (was the unsigned [sx, 0, sz] construction); this invalidates all
# conversion/reconstruction caches so sign-mismatched SHIFTs are regenerated.
POSITIONING_CONTRACT_VERSION = 2

# The four tilt.com fields this module owns.
POSITIONING_FIELDS = ("THICKNESS", "OFFSET", "XAXISTILT", "SHIFT")

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

IMOD_TO_WARP_SIGN_NOTES = (
    "OFFSET -> Warp global tilt-angle offset (LevelAngleY): effective angle = raw + OFFSET, "
    "applied exactly once. XAXISTILT -> Warp LevelAngleX; the sign is selected by the numeric "
    "oracle against the IMOD projection and must be re-confirmed against the installed warpylib "
    "Euler convention on the cluster. SHIFT (sx, sz) is a 3-D object-space translation "
    "(reconstruction X and Z=thickness), converted to Angstrom with the UNBINNED IMOD pixel "
    "size and mapped through the repository volume-frame contract; it is NOT a constant image "
    "offset (it projects as sx*cos(theta)+sz*sin(theta))."
)

# --------------------------------------------------------------------------- #
# IMOD -> Warp tilt-angle sign (the ONE canonical value; do not duplicate)
# --------------------------------------------------------------------------- #
# Warp's standard ``ts_import`` normally writes each tilt angle with sign -1 (``--dont_invert``
# retains the source sign and flips geometric handedness). Our converter writes Warp XML
# directly, bypassing that sign conversion, so we apply it explicitly and exactly once to BOTH
# the per-view angles and the OFFSET (LevelAngleY). ``-1`` is the production, IMOD-compatible
# default. Because the value is +-1 it is its own inverse.
IMOD_TO_WARP_TILT_ANGLE_SIGN = -1


def validate_tilt_angle_sign(sign) -> int:
    """Return ``sign`` as an int, requiring exactly -1 or +1."""
    s = int(sign)
    if s not in (-1, 1):
        raise ValueError(f"imod_to_warp_tilt_angle_sign must be -1 or +1, got {sign!r}")
    return s


def imod_angles_to_warp(imod_angles, sign=IMOD_TO_WARP_TILT_ANGLE_SIGN):
    """warp_angles = sign * imod_raw_angles (element-wise, applied exactly once)."""
    s = validate_tilt_angle_sign(sign)
    return [s * float(a) for a in imod_angles]


def warp_angles_to_imod(warp_angles, sign=IMOD_TO_WARP_TILT_ANGLE_SIGN):
    """imod_raw_angle = sign * warp_angle (the exact inverse; +-1 is its own inverse)."""
    s = validate_tilt_angle_sign(sign)
    return [s * float(a) for a in warp_angles]


def imod_offset_to_warp_level_angle_y(offset_deg, sign=IMOD_TO_WARP_TILT_ANGLE_SIGN):
    """LevelAngleY = sign * OFFSET (same sign as the angles, applied exactly once)."""
    return validate_tilt_angle_sign(sign) * float(offset_deg)


def warp_level_angle_y_to_imod_offset(level_angle_y, sign=IMOD_TO_WARP_TILT_ANGLE_SIGN):
    """OFFSET = sign * LevelAngleY (the exact inverse)."""
    return validate_tilt_angle_sign(sign) * float(level_angle_y)


def tilt_view_order_identity(n_views: int) -> dict:
    """The direct-stack view-order contract: Warp rows == source stack sections (identity)."""
    order = list(range(int(n_views)))
    return {
        "policy": "source_stack_order",
        "mapping": "identity",
        "warp_to_source": order,
        "source_to_warp": order,
    }


def tilt_angle_convention_manifest(sign, *, validation_status="pending_reconstruction_comparison") -> dict:
    """The angle-sign contract recorded in conversion/validation/export manifests."""
    s = validate_tilt_angle_sign(sign)
    return {
        "imod_to_warp_sign": s,
        "operation": "elementwise_negation" if s == -1 else "identity",
        "offset_uses_same_sign": True,
        "validation_status": validation_status,
    }


# --------------------------------------------------------------------------- #
# parsing (tilt.com is authoritative)
# --------------------------------------------------------------------------- #
def _active_lines(text: str) -> list[str]:
    """Non-comment, non-empty lines of an IMOD/PIP command file.

    IMOD command files start with a ``$program`` line and use ``#`` for comments.
    Blank lines and comment lines are ignored; a trailing ``#`` comment on an
    otherwise active line is stripped.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("$"):
            continue
        out.append(line)
    return out


def parse_scalar(text: str, key: str) -> Optional[float]:
    """Return the value of the LAST active ``KEY value`` / ``KEY = value`` entry.

    Case-insensitive, tolerant of leading whitespace. IMOD/PIP uses the last
    occurrence of a repeated parameter, so we scan active lines and keep the last
    match (deterministic; covered by a duplicate-entries test).
    """
    pat = re.compile(rf"(?i)^\s*{re.escape(key)}\s*(?:=)?\s*({_NUMBER})\b")
    value: Optional[float] = None
    for line in _active_lines(text):
        m = pat.match(line)
        if m:
            value = float(m.group(1))
    return value


def parse_pair_floats(text: str, key: str) -> Optional[tuple[float, float]]:
    """Return the LAST active ``KEY f1 f2`` entry as a float pair (e.g. SHIFT).

    A dedicated float parser: unlike the integer FULLIMAGE parser, SHIFT values
    are floating point (``SHIFT 0.0 -8.1``).
    """
    pat = re.compile(rf"(?i)^\s*{re.escape(key)}\s*(?:=)?\s*({_NUMBER})\s+({_NUMBER})\b")
    value: Optional[tuple[float, float]] = None
    for line in _active_lines(text):
        m = pat.match(line)
        if m:
            value = (float(m.group(1)), float(m.group(2)))
    return value


# --------------------------------------------------------------------------- #
# canonical structure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImodPositioning:
    """Resolved IMOD positioning. Optional parameters resolve to zero when applied,
    but absence and an explicit zero stay distinguishable via ``present_fields``."""

    tilt_angle_offset_deg: float = 0.0        # OFFSET
    x_axis_tilt_deg: float = 0.0              # XAXISTILT
    shift_x_unbinned_px: float = 0.0         # SHIFT[0]
    shift_z_unbinned_px: float = 0.0         # SHIFT[1]
    unbinned_pixel_size_A: Optional[float] = None
    thickness_unbinned_px: Optional[int] = None
    source_path: Optional[str] = None
    source_kind: str = "none"                # 'tilt.com' | 'tilt.log' | 'override' | 'none'
    present_fields: tuple = ()               # subset of POSITIONING_FIELDS actually found
    overridden: tuple = ()                   # fields whose value came from an explicit override
    # The ONE canonical IMOD->Warp tilt-angle sign, applied to BOTH angles and OFFSET.
    imod_to_warp_tilt_angle_sign: int = IMOD_TO_WARP_TILT_ANGLE_SIGN

    # -- physical shifts (require the UNBINNED IMOD pixel size) --------------
    @property
    def shift_x_A(self) -> Optional[float]:
        if self.unbinned_pixel_size_A is None:
            return None
        return self.shift_x_unbinned_px * self.unbinned_pixel_size_A

    @property
    def shift_z_A(self) -> Optional[float]:
        if self.unbinned_pixel_size_A is None:
            return None
        return self.shift_z_unbinned_px * self.unbinned_pixel_size_A

    @property
    def has_nonzero_shift(self) -> bool:
        return self.shift_x_unbinned_px != 0.0 or self.shift_z_unbinned_px != 0.0

    def require_pixel_size_for_shift(self) -> None:
        """Fail loudly rather than guess the physical scale of a real shift."""
        if self.has_nonzero_shift and self.unbinned_pixel_size_A is None:
            raise ValueError(
                "tilt.com SHIFT is non-zero but the unbinned IMOD pixel size could not be "
                "resolved (align.com UnbinnedPixelSize). Refusing to guess the physical scale; "
                "set [geometry.imod_positioning].unbinned_pixel_size_A explicitly.")

    # -- manifest / provenance ----------------------------------------------
    def to_manifest(self) -> dict:
        return {
            "contract_version": POSITIONING_CONTRACT_VERSION,
            "source": self.source_path,
            "source_kind": self.source_kind,
            "present_fields": list(self.present_fields),
            "overridden": list(self.overridden),
            "thickness_unbinned_px": self.thickness_unbinned_px,
            "tilt_angle_offset_deg": self.tilt_angle_offset_deg,
            "x_axis_tilt_deg": self.x_axis_tilt_deg,
            "shift_unbinned_px": [self.shift_x_unbinned_px, self.shift_z_unbinned_px],
            "unbinned_pixel_size_A": self.unbinned_pixel_size_A,
            "shift_A": [self.shift_x_A, self.shift_z_A],
            "imod_to_warp_tilt_angle_sign": self.imod_to_warp_tilt_angle_sign,
            "units": {
                "angles": "degrees",
                "shift_unbinned_px": "unbinned IMOD pixels",
                "shift_A": "angstrom",
                "thickness": "unbinned IMOD pixels",
            },
            "sign_conventions": IMOD_TO_WARP_SIGN_NOTES,
            "positioning_hash": self.positioning_hash(),
        }

    def to_toml_table(self) -> dict:
        """The ``[geometry.imod_positioning]`` table written into project_settings.toml."""
        table = {
            "contract_version": POSITIONING_CONTRACT_VERSION,
            "tilt_angle_offset_deg": self.tilt_angle_offset_deg,
            "x_axis_tilt_deg": self.x_axis_tilt_deg,
            "shift_x_unbinned_px": self.shift_x_unbinned_px,
            "shift_z_unbinned_px": self.shift_z_unbinned_px,
            "imod_to_warp_tilt_angle_sign": self.imod_to_warp_tilt_angle_sign,
            "source_kind": self.source_kind,
            "present_fields": list(self.present_fields),
        }
        if self.unbinned_pixel_size_A is not None:
            table["unbinned_pixel_size_A"] = self.unbinned_pixel_size_A
        if self.thickness_unbinned_px is not None:
            table["thickness_unbinned_px"] = self.thickness_unbinned_px
        if self.source_path is not None:
            table["source_path"] = self.source_path
        if self.overridden:
            table["overridden"] = list(self.overridden)
        return table

    def positioning_hash(self) -> str:
        """Stable hash of every value that must force a Warp reconversion when it changes.

        Deliberately excludes ``source_path``/``present_fields`` (provenance only) and
        includes the pixel size used for SHIFT and the contract version.
        """
        payload = {
            "v": POSITIONING_CONTRACT_VERSION,
            "offset": _round(self.tilt_angle_offset_deg),
            "xaxis": _round(self.x_axis_tilt_deg),
            "shift_x_px": _round(self.shift_x_unbinned_px),
            "shift_z_px": _round(self.shift_z_unbinned_px),
            "pixel_A": _round(self.unbinned_pixel_size_A) if self.unbinned_pixel_size_A is not None else None,
            "thickness": self.thickness_unbinned_px,
            "tilt_angle_sign": self.imod_to_warp_tilt_angle_sign,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _round(value: Optional[float], places: int = 9) -> Optional[float]:
    return None if value is None else round(float(value), places)


def from_toml_table(table: Optional[dict]) -> ImodPositioning:
    """Load a resolved config's ``[geometry.imod_positioning]`` table.

    A missing table (older projects) resolves to all-zero for backward-compatible
    *input loading* only; the caller decides whether an existing conversion marker
    is still current (see ``positioning_hash``).
    """
    if not table:
        return ImodPositioning()
    shift_x = float(table.get("shift_x_unbinned_px", 0.0) or 0.0)
    shift_z = float(table.get("shift_z_unbinned_px", 0.0) or 0.0)
    thickness = table.get("thickness_unbinned_px")
    return ImodPositioning(
        tilt_angle_offset_deg=float(table.get("tilt_angle_offset_deg", 0.0) or 0.0),
        x_axis_tilt_deg=float(table.get("x_axis_tilt_deg", 0.0) or 0.0),
        shift_x_unbinned_px=shift_x,
        shift_z_unbinned_px=shift_z,
        unbinned_pixel_size_A=table.get("unbinned_pixel_size_A"),
        thickness_unbinned_px=int(thickness) if thickness is not None else None,
        source_path=table.get("source_path"),
        source_kind=str(table.get("source_kind", "config")),
        present_fields=tuple(table.get("present_fields", ()) or ()),
        overridden=tuple(table.get("overridden", ()) or ()),
        imod_to_warp_tilt_angle_sign=validate_tilt_angle_sign(
            table.get("imod_to_warp_tilt_angle_sign", IMOD_TO_WARP_TILT_ANGLE_SIGN)),
    )


def parse_imod_positioning(
    tilt_com_path: Optional[Path],
    *,
    unbinned_pixel_size_A: Optional[float] = None,
    thickness_unbinned_px: Optional[int] = None,
    tilt_log_path: Optional[Path] = None,
    overrides: Optional[dict] = None,
) -> ImodPositioning:
    """Parse OFFSET/XAXISTILT/SHIFT from ``tilt.com`` (authoritative).

    ``tilt.log`` is consulted ONLY as an explicitly-recorded per-field fallback when
    ``tilt.com`` is missing or lacks the field. ``overrides`` (explicit project
    setting / CLI) win over both. Precedence per field:
        override > tilt.com > tilt.log > documented zero default.
    """
    overrides = overrides or {}
    present: list[str] = []
    overridden: list[str] = []
    source_kind = "none"
    source_path: Optional[str] = None

    com_text = tilt_com_path.read_text(errors="replace") if (tilt_com_path and Path(tilt_com_path).is_file()) else ""
    log_text = tilt_log_path.read_text(errors="replace") if (tilt_log_path and Path(tilt_log_path).is_file()) else ""
    if com_text:
        source_kind, source_path = "tilt.com", str(tilt_com_path)

    def resolve_scalar(field_key: str, override_key: str) -> float:
        nonlocal source_kind
        if overrides.get(override_key) is not None:
            overridden.append(field_key)
            present.append(field_key)
            return float(overrides[override_key])
        value = parse_scalar(com_text, field_key)
        if value is None and log_text:
            value = parse_scalar(log_text, field_key)
            if value is not None and source_kind == "none":
                source_kind = "tilt.log"
        if value is not None:
            present.append(field_key)
            return value
        return 0.0

    offset = resolve_scalar("OFFSET", "tilt_angle_offset_deg")
    xaxis = resolve_scalar("XAXISTILT", "x_axis_tilt_deg")

    shift_override = overrides.get("shift_unbinned_px")
    if shift_override is not None:
        shift = (float(shift_override[0]), float(shift_override[1]))
        overridden.append("SHIFT")
        present.append("SHIFT")
    else:
        shift = parse_pair_floats(com_text, "SHIFT")
        if shift is None and log_text:
            shift = parse_pair_floats(log_text, "SHIFT")
            if shift is not None and source_kind in ("none", "tilt.com"):
                source_kind = "tilt.log"
        if shift is not None:
            present.append("SHIFT")
        else:
            shift = (0.0, 0.0)

    thickness = thickness_unbinned_px
    if overrides.get("thickness_unbinned_px") is not None:
        thickness = int(overrides["thickness_unbinned_px"])
        overridden.append("THICKNESS")
    if thickness is None:
        parsed_thickness = parse_scalar(com_text, "THICKNESS")
        thickness = int(parsed_thickness) if parsed_thickness is not None else None
    if thickness is not None:
        present.append("THICKNESS")

    pixel = overrides.get("unbinned_pixel_size_A", unbinned_pixel_size_A)
    tilt_angle_sign = validate_tilt_angle_sign(
        overrides.get("imod_to_warp_tilt_angle_sign", IMOD_TO_WARP_TILT_ANGLE_SIGN))

    result = ImodPositioning(
        tilt_angle_offset_deg=offset,
        x_axis_tilt_deg=xaxis,
        shift_x_unbinned_px=shift[0],
        shift_z_unbinned_px=shift[1],
        unbinned_pixel_size_A=pixel,
        thickness_unbinned_px=thickness,
        source_path=source_path,
        source_kind=source_kind,
        present_fields=tuple(dict.fromkeys(present)),   # dedupe, keep order
        overridden=tuple(dict.fromkeys(overridden)),
        imod_to_warp_tilt_angle_sign=tilt_angle_sign,
    )
    result.require_pixel_size_for_shift()
    return result


# --------------------------------------------------------------------------- #
# named IMOD -> Warp conversion functions (documented conventions)
# --------------------------------------------------------------------------- #
def imod_offset_to_warp(raw_tilt_angles_deg: Iterable[float], offset_deg: float) -> dict:
    """OFFSET -> Warp global tilt-angle offset.

    source frame : IMOD tilt angles (per view), degrees
    dest frame   : Warp per-view rotation angle; Warp applies angle + LevelAngleY
    convention   : effective_angle_i = raw_angle_i + OFFSET, applied EXACTLY ONCE
    representation: 'level_angle_y' (the raw .tlt values stay in ts.angles)
    units        : degrees
    """
    raw = [float(a) for a in raw_tilt_angles_deg]
    effective = [a + float(offset_deg) for a in raw]
    return {
        "raw_tilt_angles_deg": raw,
        "tilt_angle_offset_deg": float(offset_deg),
        "effective_tilt_angles_deg": effective,
        "warp_representation": "level_angle_y",
        "raw_range_deg": [min(raw), max(raw)] if raw else None,
        "effective_range_deg": [min(effective), max(effective)] if effective else None,
    }


def imod_xaxis_tilt_to_warp(x_axis_tilt_deg: float, *, sign: int) -> dict:
    """XAXISTILT -> Warp LevelAngleX.

    source frame : IMOD reconstruction X-axis (tilt-axis) tilt, degrees
    dest frame   : Warp LevelAngleX (RotateX in the Warp Euler chain)
    convention   : LevelAngleX = sign * XAXISTILT, where ``sign`` in {+1, -1} is
                   selected numerically by the projection oracle (never by name)
    units        : degrees
    """
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    return {
        "x_axis_tilt_deg": float(x_axis_tilt_deg),
        "warp_level_angle_x_deg": sign * float(x_axis_tilt_deg),
        "sign": sign,
        "warp_representation": "level_angle_x",
    }


# Volume-frame contract (uses scripts/geometry/volume_frames.py). The IMOD reconstruction
# SHIFT (X, Z=thickness) is a 3-D VECTOR: build it in NATIVE IMOD-MRC axis order
# [X, Y=thickness, Z=detector] as [sx_A, sz_A, 0] (SHIFT Z is thickness -> IMOD-MRC Y), then
# transform ONCE with the signed IMOD_MRC_TO_WARP orientation. This is NOT [sx, 0, sz].
def imod_reconstruction_shift_to_warp(
    shift_x_unbinned_px: float,
    shift_z_unbinned_px: float,
    unbinned_pixel_size_A: float,
    *,
    tilt_angle_sign: int = IMOD_TO_WARP_TILT_ANGLE_SIGN,
) -> dict:
    """SHIFT -> Warp object-space translation (Angstrom), via the signed frame matrix.

    IMOD-MRC vector = [sx_A, sz_A, 0] (X, Y=thickness, Z=detector); Warp = M @ that, with
    M = imod_mrc_to_warp_orientation(tilt_angle_sign). For sx=0, sz=-8.1, pixel=2.2, sign=-1:
    IMOD-MRC = [0, -17.82, 0] -> Warp = [0, 0, +17.82] Angstrom (was [0, 0, -17.82]).
    """
    if unbinned_pixel_size_A is None:
        raise ValueError("SHIFT physical mapping requires the unbinned IMOD pixel size")
    import numpy as np
    from geometry.volume_frames import imod_mrc_to_warp_orientation

    pixel = float(unbinned_pixel_size_A)
    shift_imod_mrc_A = np.array(
        [float(shift_x_unbinned_px) * pixel, float(shift_z_unbinned_px) * pixel, 0.0],
        dtype=np.float64)
    orientation = imod_mrc_to_warp_orientation(tilt_angle_sign)
    shift_warp_A = orientation @ shift_imod_mrc_A
    return {
        "warp_object_shift_A": [float(v) for v in shift_warp_A],   # (X, Y, Z=thickness)
        "imod_shift_vector_A": [float(v) for v in shift_imod_mrc_A],
        "orientation_matrix_imod_mrc_to_warp": orientation.tolist(),
        "orientation_determinant": int(round(float(np.linalg.det(orientation)))),
        "reconstruction_shift_unbinned_px": [float(shift_x_unbinned_px), float(shift_z_unbinned_px)],
        "unbinned_pixel_size_A": pixel,
        "tilt_angle_sign": int(tilt_angle_sign),
        "warp_representation": "object_space_translation_xyz_A_signed_frame",
    }


def warp_shift_to_imod_reconstruction(
    warp_object_shift_A,
    unbinned_pixel_size_A: float,
    *,
    tilt_angle_sign: int = IMOD_TO_WARP_TILT_ANGLE_SIGN,
) -> dict:
    """Exact inverse of :func:`imod_reconstruction_shift_to_warp`.

    IMOD-MRC vector = WARP_TO_IMOD_MRC @ warp; recover SHIFT X from component 0 and SHIFT Z
    from component 1 (NOT component 2), divided by the unbinned pixel size.
    """
    if unbinned_pixel_size_A is None:
        raise ValueError("SHIFT inverse mapping requires the unbinned IMOD pixel size")
    import numpy as np
    from geometry.volume_frames import warp_to_imod_mrc_orientation

    pixel = float(unbinned_pixel_size_A)
    warp = np.asarray(warp_object_shift_A, dtype=np.float64).reshape(3)
    shift_imod_mrc_A = warp_to_imod_mrc_orientation(tilt_angle_sign) @ warp
    return {
        "shift_x_unbinned_px": float(shift_imod_mrc_A[0] / pixel),
        "shift_z_unbinned_px": float(shift_imod_mrc_A[1] / pixel),   # component 1, not 2
        "imod_shift_vector_A": [float(v) for v in shift_imod_mrc_A],
        "tilt_angle_sign": int(tilt_angle_sign),
    }


# --------------------------------------------------------------------------- #
# numerical IMOD projection oracle (numpy) — establishes signs/axes by geometry
# --------------------------------------------------------------------------- #
def imod_detector_projection(point_xyz, tilt_deg, *, offset_deg=0.0, x_axis_tilt_deg=0.0,
                             shift_xz_px=(0.0, 0.0)):
    """Forward-project one specimen point to the aligned-stack detector, IMOD convention.

    point_xyz          : (x, y, z) in reconstruction/specimen coordinates
                         x = tilt axis, y = in-plane perpendicular, z = beam/thickness
    returns            : (u, v) detector coordinates
                         u = perpendicular to the tilt axis, v = along the tilt axis

    Model (documented above):
      1. XAXISTILT tilts the specimen about the X (tilt) axis by ``x_axis_tilt_deg``,
         coupling y and z.
      2. SHIFT translates the reconstruction by (sx, 0, sz).
      3. The view tilt is (raw tilt + OFFSET) about the tilt axis; the detector
         coordinate perpendicular to the axis is u = x'*sin(theta)+... see code.
    """
    import numpy as np

    x, y, z = (float(c) for c in point_xyz)
    sx, sz = (float(s) for s in shift_xz_px)
    a = np.deg2rad(float(x_axis_tilt_deg))
    theta = np.deg2rad(float(tilt_deg) + float(offset_deg))

    # 1. X-axis tilt: rotate (y, z) about the X axis by ``a``.
    y1 = y * np.cos(a) - z * np.sin(a)
    z1 = y * np.sin(a) + z * np.cos(a)
    x1 = x
    # 2. reconstruction shift (X, Z).
    x1 += sx
    z1 += sz
    # 3. project at the (offset-corrected) tilt angle about the tilt axis.
    #    u is perpendicular to the tilt axis; v is along the tilt axis.
    u = x1 * np.cos(theta) + z1 * np.sin(theta)
    v = y1
    return float(u), float(v)
