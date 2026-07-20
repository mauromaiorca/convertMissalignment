#!/usr/bin/env python3
"""External IMOD CTF phase-flipping orchestration.

Implements the convention validated in ``CTF_TRANSFORM_CONVENTION_VALIDATION.md``
against real ``ctfphaseflip`` 5.1.9: CTF is applied to an ALIGNED stack with no
per-image transform; ``PixelSize``/``UnbinnedPixelSize`` are in nanometres and
their ratio encodes the binning; ``InterpolationWidth`` is required and
``MaximumStripWidth`` must NOT be 0. Source command files are never modified --
they are copied locally and patched with the reused
``patch_imod_scripts.patch_standard_input_file``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patch_imod_scripts import directive_key, patch_standard_input_file  # noqa: E402

CTF_MODES = ("off", "working", "final", "both")


class CtfError(ValueError):
    pass


def validate_ctf_mode(mode: str, condition: str, has_aligned_stack: bool) -> None:
    if mode not in CTF_MODES:
        raise CtfError(f"ctf.mode={mode!r} not in {CTF_MODES}")
    if mode in ("working", "both"):
        if condition != "ali_identity":
            raise CtfError(
                f"ctf.mode={mode!r} requires condition=ali_identity (working CTF needs an aligned "
                f"working stack); got condition={condition!r}. Raw conditions with working CTF are "
                "not supported -- this fails clearly rather than guessing.")
        if not has_aligned_stack:
            raise CtfError(f"ctf.mode={mode!r} requires a valid aligned working stack")


@dataclass
class CtfInputs:
    command_file: str | None = None
    defocus_file: str | None = None
    angle_file: str | None = None
    aligned_stack: str | None = None
    raw_stack: str | None = None
    selection_reasons: dict[str, str] = field(default_factory=dict)


def _unique(cands: list[Path], label: str, reasons: dict) -> Path | None:
    files = sorted({p.resolve() for p in cands if p.is_file()})
    if not files:
        return None
    if len(files) > 1:
        raise CtfError(
            f"ambiguous {label}: {[str(f) for f in files]}. Disambiguate via [ctf] in the TOML "
            "or a CLI override; refusing to guess.")
    reasons[label] = f"unique match: {files[0].name}"
    return files[0]


def discover_ctf_inputs(data_dir: Path, basename: str, overrides: dict[str, str] | None = None) -> CtfInputs:
    overrides = overrides or {}
    data_dir = Path(data_dir)
    reasons: dict[str, str] = {}

    def resolve(key, label, patterns):
        if overrides.get(key):
            p = Path(overrides[key])
            p = p if p.is_absolute() else data_dir / p
            if not p.is_file():
                raise CtfError(f"{label} override not found: {p}")
            reasons[label] = f"explicit override: {overrides[key]}"
            return str(p.resolve())
        cands = []
        for pat in patterns:
            cands += list(data_dir.glob(pat)) + list(data_dir.rglob(pat))
        u = _unique(cands, label, reasons)
        return str(u) if u else None

    return CtfInputs(
        command_file=resolve("command_file", "ctfcorrection.com", ["ctfcorrection.com"]),
        defocus_file=resolve("defocus_file", ".defocus", [f"{basename}.defocus", "*.defocus"]),
        angle_file=resolve("angle_file", ".tlt", [f"{basename}.tlt", f"{basename}_ali.tlt"]),
        aligned_stack=resolve("aligned_stack", "_ali", [f"{basename}_ali.mrc", f"{basename}.ali"]),
        raw_stack=resolve("raw_stack", "raw", [f"{basename}.mrc", f"{basename}.st"]),
        selection_reasons=reasons,
    )


def angstrom_to_nm(a: float) -> float:
    return float(a) / 10.0


def build_ctfphaseflip_cmd(*, input_stack: Path, output_stack: Path, angle_file: Path,
                           defocus_file: Path, pixel_size_A: float, unbinned_pixel_A: float,
                           voltage_kv: int = 300, cs_mm: float = 2.7, amp_contrast: float = 0.07,
                           defocus_tol_nm: int = 200, interpolation_width: int = 20,
                           axis_angle_deg: float = 0.0, use_gpu: int = 0) -> list[str]:
    """Direct ctfphaseflip command for an ALIGNED stack (no transform).

    Pixel sizes are converted to nm. ``MaximumStripWidth`` is intentionally
    omitted (dynamic) -- setting it to 0 hangs the program.
    """
    return [
        "ctfphaseflip",
        "-InputStack", str(input_stack), "-OutputFileName", str(output_stack),
        "-AngleFile", str(angle_file), "-DefocusFile", str(defocus_file),
        "-PixelSize", f"{angstrom_to_nm(pixel_size_A):.6f}",
        "-UnbinnedPixelSize", f"{angstrom_to_nm(unbinned_pixel_A):.6f}",
        "-DefocusTol", str(int(defocus_tol_nm)),
        "-InterpolationWidth", str(int(interpolation_width)),
        "-Voltage", str(int(voltage_kv)), "-SphericalAberration", f"{cs_mm}",
        "-AmplitudeContrast", f"{amp_contrast}", "-AxisAngle", f"{axis_angle_deg}",
        "-UseGPU", str(int(use_gpu)),
    ]


def parse_ctf_com_params(source_com: Path) -> dict[str, Any]:
    """Read project CTF parameters from a real ``ctfcorrection.com`` (defect #9).

    Returns the measured ``voltage_kv``, ``cs_mm``, ``amplitude_contrast`` and
    ``axis_angle_deg`` parsed from the file's ``ctfphaseflip`` directives, plus
    ``defocus_file``/``angle_file``/``pixel_size_nm``/``unbinned_pixel_nm`` when
    present, and a ``found`` map recording which keys were located. Values not
    present in the file are returned as ``None`` -- the caller decides whether a
    config default may fill a gap (and records that it did). The source file is
    only read, never modified.
    """
    source_com = Path(source_com)
    if not source_com.is_file():
        raise CtfError(f"source ctfcorrection.com not found: {source_com}")
    vals: dict[str, str] = {}
    for ln in source_com.read_text(errors="replace").splitlines():
        key = directive_key(ln)
        if key is None:
            continue
        parts = ln.strip().split(maxsplit=1)
        vals[key] = parts[1].strip() if len(parts) > 1 else ""

    def _num(key, cast):
        if key in vals and vals[key] != "":
            try:
                return cast(vals[key].split()[0])
            except (ValueError, IndexError):
                raise CtfError(f"{source_com}: unparseable {key} value {vals[key]!r}")
        return None

    voltage = _num("voltage", float)
    cs = _num("sphericalaberration", float)
    amp = _num("amplitudecontrast", float)
    axis = _num("axisangle", float)
    pix_nm = _num("pixelsize", float)
    unb_nm = _num("unbinnedpixelsize", float)
    out = {
        "voltage_kv": int(voltage) if voltage is not None else None,
        "cs_mm": cs, "amplitude_contrast": amp, "axis_angle_deg": axis,
        "pixel_size_nm": pix_nm, "unbinned_pixel_nm": unb_nm,
        "defocus_file": vals.get("defocusfile") or None,
        "angle_file": vals.get("anglefile") or None,
        "has_transform_file": "transformfile" in vals,
        "found": {k: (k in vals and vals[k] != "") for k in
                  ("voltage", "sphericalaberration", "amplitudecontrast", "axisangle",
                   "pixelsize", "unbinnedpixelsize", "defocusfile", "anglefile")},
        "source_com": str(source_com),
    }
    return out


def patch_ctf_com(source_com: Path, local_com: Path, replacements: dict[str, str]) -> dict[str, Any]:
    """Copy the source ctfcorrection.com locally and patch it (source untouched).

    Returns a patch report with before/after values. Reuses
    ``patch_imod_scripts.patch_standard_input_file`` (preserves unknown entries,
    case-insensitive keys). It never sets ``MaximumStripWidth`` to 0.
    """
    source_com = Path(source_com)
    if not source_com.is_file():
        raise CtfError(f"source ctfcorrection.com not found: {source_com}")
    # Reject raw-stack CTF (per-image TransformFile) in the first implementation.
    before_lines = source_com.read_text(errors="replace").splitlines()
    keys = {directive_key(ln) for ln in before_lines}
    keys = {k.lower() for k in keys if k}
    if "transformfile" in keys:
        raise CtfError(
            "source ctfcorrection.com specifies a TransformFile (raw-stack CTF). The first "
            "implementation supports aligned-stack CTF only; this fails clearly.")
    if "maximumstripwidth" in (k.lower() for k in replacements):
        raise CtfError("refusing to patch MaximumStripWidth (0 hangs ctfphaseflip)")
    local_com.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_com, local_com)
    before = {directive_key(ln): ln for ln in before_lines if directive_key(ln)}
    patch_standard_input_file(local_com, replacements, append_missing=True)
    after_lines = local_com.read_text().splitlines()
    after = {directive_key(ln): ln for ln in after_lines if directive_key(ln)}
    return {
        "source_com": str(source_com), "local_com": str(local_com),
        "patched_keys": list(replacements.keys()),
        "before": {k: before.get(k) for k in replacements},
        "after": {k: after.get(k) for k in replacements},
        "source_unmodified": source_com.read_text(errors="replace").splitlines() == before_lines,
    }


def assert_uncorrected_input(ctf_state: str) -> None:
    """Refuse to phase-flip an already CTF-corrected stack (double-CTF guard)."""
    if ctf_state == "phase_flipped":
        raise CtfError("refusing to CTF-correct an already phase-flipped stack (double CTF). "
                       "CTF must be applied to an uncorrected stack.")


def run_ctfphaseflip(cmd: list[str], *, env=None) -> subprocess.CompletedProcess:
    e = dict(os.environ if env is None else env)
    e.setdefault("IMOD_DIR", "/Applications/IMOD")
    return subprocess.run(cmd, env=e, text=True, capture_output=True)


def validate_ctf_output(input_stack: Path, output_stack: Path) -> dict[str, Any]:
    import numpy as np
    import mrcfile
    with mrcfile.open(input_stack, permissive=True) as h:
        ishape = h.data.shape; ipix = float(h.voxel_size.x)
    with mrcfile.open(output_stack, permissive=True) as h:
        odata = np.asarray(h.data); opix = float(h.voxel_size.x)
    report = {
        "dims_preserved": tuple(odata.shape) == tuple(ishape),
        "tilt_count_preserved": odata.shape[0] == ishape[0],
        "pixel_preserved": abs(opix - ipix) < 1e-3,
        "all_finite": bool(np.isfinite(odata).all()),
        "input_shape": list(ishape), "output_shape": list(odata.shape),
        "pixel_A": opix,
    }
    report["ok"] = all([report["dims_preserved"], report["tilt_count_preserved"],
                        report["pixel_preserved"], report["all_finite"]])
    return report
