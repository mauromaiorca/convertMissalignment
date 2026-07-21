"""Canonical WarpTools reconstruction tiling / locale / contract configuration.

Every ``ts_reconstruct`` call site (pre_conversion_reconstruction.py,
warptools_reconstruction.py, and any future path) must resolve its tiling through
``resolve_tiling`` and pass ``ReconstructionTiling.to_args()``, so the padded-context
size, cache contract and resource preflight are identical everywhere.

Padding semantics (mirrors the installed Warp ``ts_reconstruct`` source):

    int SizeSub       = SubVolumeSize
    int SizeSubPadded = (int)(SizeSub * SubVolumePadding) * 2
    Projector(new int3(SizeSubPadded), 1)          # cubic, isotropic
    subtomo        : int3(SizeSubPadded)           # reconstructed with 3-D context
    subtomoCropped : int3(SizeSub)                 # central crop copied to output

So ``subvolume_padding`` is a single ISOTROPIC XYZ padding *factor*; the padded
reconstruction side is ``int(size * padding) * 2`` in X, Y AND Z. It provides padded
reconstruction *context*, NOT true final-volume overlap: block centres are strided by
``SubVolumeSize`` and the cropped central blocks are copied directly into the output,
so the actual output overlap is ZERO (non-overlapping central crop, direct copy).
The boundary artefact can look stronger in XZ than XY because of tomographic
anisotropy, but the padding acts equally in all three axes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

RECONSTRUCTION_CONTRACT_VERSION = 1

DEFAULT_SUBVOLUME_SIZE = 64
DEFAULT_SUBVOLUME_PADDING = 6
MIN_SUBVOLUME_PADDING = 6          # this project requires >= 6; lower is rejected, not clamped

# Warp default padding, used only to report the memory ratio (never as a fallback).
WARP_DEFAULT_PADDING = 3

PADDING_AXES = "XYZ_isotropic"
ASSEMBLY_METHOD = "non_overlapping_central_crop_direct_copy"
ACTUAL_OUTPUT_OVERLAP_PX = 0

# XML numeric fields that must not contain locale comma decimals.
_LOCALE_SENSITIVE_XML_FIELDS = (
    "Angles", "AngleX", "AngleY", "LevelAngleX", "LevelAngleY",
    "AxisAngle", "TiltAxisAngle", "AxisOffsetX", "AxisOffsetY",
    "PixelSize", "Dimensions", "VolumeDimensionsAngstrom", "GridMovement",
    "GridVolumeWarp", "Offset", "Dose",
)


class ReconstructionConfigError(ValueError):
    """Raised for an invalid or out-of-policy reconstruction tiling configuration."""


def _validate_positive_int(value, name: str) -> int:
    # A bool is an int subclass in Python; reject it explicitly.
    if isinstance(value, bool):
        raise ReconstructionConfigError(f"{name} must be an integer, not a boolean ({value!r})")
    if isinstance(value, float):
        if not value.is_integer():
            raise ReconstructionConfigError(f"{name} must be a whole integer, got {value!r}")
        value = int(value)
    if not isinstance(value, int):
        raise ReconstructionConfigError(f"{name} must be an integer, got {type(value).__name__} {value!r}")
    if value <= 0:
        raise ReconstructionConfigError(f"{name} must be a positive integer, got {value}")
    return value


@dataclass(frozen=True)
class ReconstructionTiling:
    subvolume_size: int = DEFAULT_SUBVOLUME_SIZE
    subvolume_padding: int = DEFAULT_SUBVOLUME_PADDING

    @property
    def padded_side(self) -> int:
        """Padded cubic reconstruction side in pixels: int(size * padding) * 2 (Warp source)."""
        return int(self.subvolume_size * self.subvolume_padding) * 2

    @property
    def padded_voxel_count(self) -> int:
        side = self.padded_side
        return side * side * side

    def ratio_vs_padding(self, other_padding: int = WARP_DEFAULT_PADDING) -> float:
        """Cubic-allocation ratio versus another padding factor (e.g. Warp default 3)."""
        other = ReconstructionTiling(self.subvolume_size, other_padding)
        return self.padded_voxel_count / other.padded_voxel_count

    def to_args(self) -> list[str]:
        """The explicit ts_reconstruct flags; never rely on Warp defaults."""
        return [
            "--subvolume_size", str(self.subvolume_size),
            "--subvolume_padding", str(self.subvolume_padding),
        ]

    def resolved_interpretation(self) -> dict:
        return {
            "subvolume_size_px": self.subvolume_size,
            "subvolume_padding_factor": self.subvolume_padding,
            "padding_axes": PADDING_AXES,
            "padded_reconstruction_side_px": self.padded_side,
            "padded_voxel_count": self.padded_voxel_count,
            "actual_output_overlap_px": ACTUAL_OUTPUT_OVERLAP_PX,
            "assembly_method": ASSEMBLY_METHOD,
            "contract_version": RECONSTRUCTION_CONTRACT_VERSION,
        }


def build_ts_reconstruct_command(
    executable: str,
    *,
    settings,
    input_data,
    output_angpix: float,
    device_list: str,
    perdevice: int,
    tiling: "ReconstructionTiling",
    dont_invert: bool = True,
    normalize: bool = False,
    dont_mask: bool = True,
) -> list[str]:
    """The single canonical ``ts_reconstruct`` base command shared by every call site.

    Emits ``--subvolume_size`` and ``--subvolume_padding`` explicitly (never relying on
    Warp defaults). Per-call ``--input_processing``/``--output_processing`` are appended
    by the caller. ``normalize=False`` keeps ``--dont_normalize`` (the current policy).
    """
    cmd = [
        executable, "ts_reconstruct",
        "--settings", str(settings),
        "--input_data", str(input_data),
        "--angpix", f"{float(output_angpix):.12g}",
        "--device_list", str(device_list),
        "--perdevice", str(int(perdevice)),
        *tiling.to_args(),
    ]
    if dont_invert:
        cmd.append("--dont_invert")
    if not normalize:
        cmd.append("--dont_normalize")
    if dont_mask:
        cmd.append("--dont_mask")
    return cmd


def resolve_tiling(table: Optional[dict]) -> ReconstructionTiling:
    """Resolve [reconstruction.warptools] {subvolume_size, subvolume_padding}.

    Applies the project policy: size defaults to 64, padding defaults to 6 and must be
    >= 6. A lower padding is REJECTED with a clear error, never silently replaced with
    the Warp default. Values must be finite positive integers (booleans rejected).
    """
    table = table or {}
    size = _validate_positive_int(table.get("subvolume_size", DEFAULT_SUBVOLUME_SIZE), "subvolume_size")
    padding = _validate_positive_int(table.get("subvolume_padding", DEFAULT_SUBVOLUME_PADDING), "subvolume_padding")
    if padding < MIN_SUBVOLUME_PADDING:
        raise ReconstructionConfigError(
            f"subvolume_padding={padding} is below the required minimum {MIN_SUBVOLUME_PADDING} "
            f"for this project. Set it to >= {MIN_SUBVOLUME_PADDING} (default {DEFAULT_SUBVOLUME_PADDING}); "
            "the Warp default is NOT substituted.")
    return ReconstructionTiling(subvolume_size=size, subvolume_padding=padding)


# --------------------------------------------------------------------------- #
# locale safety (F)
# --------------------------------------------------------------------------- #
def warptools_env(base: Optional[dict] = None) -> dict:
    """A copy of the environment with a deterministic numeric locale merged in.

    Merges LC_ALL=C and LANG=C without deleting the rest of the environment, so
    WarpTools never emits comma decimal separators.
    """
    env = dict(os.environ if base is None else base)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def xml_comma_decimal_fields(xml_text: str, fields=_LOCALE_SENSITIVE_XML_FIELDS) -> list[str]:
    """Return the numeric XML fields that contain locale comma decimals (e.g. ``1,23``).

    A comma decimal is a digit-comma-digit inside an attribute/element value. Returns
    the offending field names; an empty list means the metadata are locale-clean.
    """
    offenders: list[str] = []
    for field in fields:
        f = re.escape(field)
        # Covers the common Warp/XML forms:
        #   Field="1,23"                     (attribute named after the field)
        #   <Param Name="Field" Value="1,23">/ <Param Name="Field">1,23</Param>
        #   <Field ...>1,23</Field>          (element named after the field)
        pat = re.compile(
            rf'{f}\s*=\s*"[^"]*\d,\d'
            rf'|Name\s*=\s*"{f}"[^<]*?\d,\d'
            rf'|<{f}\b[^<]*?\d,\d',
            re.IGNORECASE)
        if pat.search(xml_text):
            offenders.append(field)
    return offenders


# --------------------------------------------------------------------------- #
# resource preflight (E)
# --------------------------------------------------------------------------- #
def resource_preflight(tiling: ReconstructionTiling, *, n_tilts: int, device_list: str,
                       free_vram_mb: Optional[float] = None) -> dict:
    """Report cubic allocation size; NOT an exact VRAM prediction."""
    report = {
        "subvolume_size": tiling.subvolume_size,
        "subvolume_padding": tiling.subvolume_padding,
        "padded_side_px": tiling.padded_side,
        "padded_voxel_count": tiling.padded_voxel_count,
        "ratio_vs_warp_default_padding_3": round(tiling.ratio_vs_padding(WARP_DEFAULT_PADDING), 4),
        "ratio_vs_padding_4": round(tiling.ratio_vs_padding(4), 4),
        "n_tilts": int(n_tilts),
        "device_list": device_list,
        "free_vram_mb": free_vram_mb,
        "note": (
            "Cubic voxel count is NOT an exact VRAM prediction: Warp also allocates "
            "projectors, Fourier arrays, CTF arrays, sample weights and per-tilt image "
            "buffers. On CUDA OOM / cuFFT / allocation failure, reduce concurrency first "
            "(--perdevice 1) before reducing padding."),
    }
    return report


# --------------------------------------------------------------------------- #
# reconstruction cache contract (H)
# --------------------------------------------------------------------------- #
def reconstruction_contract_hash(
    tiling: ReconstructionTiling,
    *,
    output_angpix: Optional[float],
    normalize: bool,
    warptools_version: Optional[str],
    numeric_locale: str = "C",
    tilt_angle_sign: Optional[int] = None,
) -> str:
    """Stable hash forcing reconstruction reuse only when every geometry-affecting
    input matches: tiling, requested angpix, normalisation policy, WarpTools version,
    numeric locale and the IMOD->Warp tilt-angle sign (so a sign +1 reconstruction is
    never reused for a sign -1 request)."""
    payload = {
        "v": RECONSTRUCTION_CONTRACT_VERSION,
        "subvolume_size": tiling.subvolume_size,
        "subvolume_padding": tiling.subvolume_padding,
        "output_angpix": None if output_angpix is None else round(float(output_angpix), 9),
        "normalize": bool(normalize),
        "warptools_version": warptools_version,
        "numeric_locale": numeric_locale,
        "tilt_angle_sign": None if tilt_angle_sign is None else int(tilt_angle_sign),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reconstruction_identity(tiling: ReconstructionTiling) -> str:
    """A parameter-specific output/manifest id so diagnostic reconstructions with
    different tiling cannot be confused, e.g. ``reconstruction_s64_p6``."""
    return f"reconstruction_s{tiling.subvolume_size}_p{tiling.subvolume_padding}"


# --------------------------------------------------------------------------- #
# pixel-size consistency (G)
# --------------------------------------------------------------------------- #
def pixel_size_consistency_report(
    *,
    unbinned_pixel_size_A: Optional[float],
    image_binned: Optional[int],
    aligned_pixel_size_A: Optional[float],
    warp_input_angpix_A: Optional[float],
    requested_output_angpix_A: Optional[float],
    xml_volume_physical_A=None,
    output_voxel_size_A: Optional[float] = None,
    tolerance_frac: float = 0.01,
) -> dict:
    """Keep every pixel-size quantity separate and validate the declared scaling.

    Binning is expected, so the output pixel size is NOT required to equal the unbinned
    acquisition pixel size. A mismatch between the requested --angpix and the produced
    MRC voxel size is a DISTINCT error, never a padding artefact.
    """
    report = {
        "unbinned_pixel_size_A": unbinned_pixel_size_A,
        "image_binned": image_binned,
        "aligned_pixel_size_A": aligned_pixel_size_A,
        "warp_input_angpix_A": warp_input_angpix_A,
        "requested_output_angpix_A": requested_output_angpix_A,
        "xml_volume_physical_A": list(xml_volume_physical_A) if xml_volume_physical_A else None,
        "output_voxel_size_A": output_voxel_size_A,
        "tolerance_frac": tolerance_frac,
        "problems": [],
        "output_voxel_matches_request": None,
    }
    req = requested_output_angpix_A
    got = output_voxel_size_A
    if req is not None and got is not None:
        if req <= 0 or not math.isfinite(req):
            report["problems"].append(f"requested output angpix invalid: {req}")
        else:
            ok = abs(float(got) - float(req)) / abs(float(req)) <= tolerance_frac
            report["output_voxel_matches_request"] = bool(ok)
            if not ok:
                report["problems"].append(
                    f"output MRC voxel size {got} A disagrees with requested --angpix {req} A "
                    f"beyond {tolerance_frac:.1%} (pixel-size error, NOT a block-padding artefact)")
    return report
