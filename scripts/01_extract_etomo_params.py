#!/usr/bin/env python3
"""
Extract parameters from an IMOD/eTomo tilt-series layout for MissAlignment.

This script does NOT convert data and does NOT run MissAlignment. It only reads
small metadata files and MRC headers, then writes a JSON file and a readable
summary report that can be used by the conversion and execution scripts.

Example
-------
conda activate missalign
python 01_extract_etomo_params.py \
  --etomo-dir ./lam8_ts_004 \
  --out-dir ./missalign_params_lam8 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

try:
    import mrcfile
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: mrcfile is not installed. Activate your missalign environment or run:\n"
        "  python -m pip install mrcfile\n"
    ) from exc


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class MRCInfo:
    path: str
    exists: bool
    shape_zyx: list[int] | None = None
    header_nxyz: list[int] | None = None
    voxel_size_A: list[float] | None = None
    dtype: str | None = None
    cella: list[float] | None = None
    origin: list[float] | None = None
    error: str | None = None


# ----------------------------- basic helpers -----------------------------


def norm_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)




def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, PermissionError):
        return False


def safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, PermissionError):
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except (OSError, PermissionError):
        return False


def safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except (OSError, PermissionError):
        return []


def safe_glob(path: Path, pattern: str) -> list[Path]:
    try:
        return list(path.glob(pattern))
    except (OSError, PermissionError):
        return []


def safe_readable_file(path: Path) -> bool:
    return safe_is_file(path) and os.access(path, os.R_OK)

def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def count_nonempty_lines(path: Path | None) -> int | None:
    if path is None or not safe_exists(path):
        return None
    n = 0
    try:
        with path.open("r", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    n += 1
    except (OSError, PermissionError):
        return None
    return n


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if safe_readable_file(path):
            return path
    return None


def parse_key_value_number(text: str, keys: Iterable[str]) -> float | None:
    """Parse IMOD-like key/value numeric parameters from text."""
    for key in keys:
        # Accept: RotationAngle 84.0, RotationAngle = 84.0, RotationAngle\t84.0
        pat = re.compile(rf"(?im)^\s*{re.escape(key)}\s*(?:=)?\s*({NUMBER_RE})\b")
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def parse_key_value_string(text: str, keys: Iterable[str]) -> str | None:
    for key in keys:
        pat = re.compile(rf"(?im)^\s*{re.escape(key)}\s*(?:=)?\s*([^\s#]+)")
        m = pat.search(text)
        if m:
            return m.group(1).strip().strip('"')
    return None


def parse_pair_ints(text: str, keys: Iterable[str]) -> list[int] | None:
    for key in keys:
        pat = re.compile(rf"(?im)^\s*{re.escape(key)}\s*(?:=)?\s*(\d+)\s+(\d+)\b")
        m = pat.search(text)
        if m:
            return [int(m.group(1)), int(m.group(2))]
    return None


def parse_tilt_axis_angle(imod_dir: Path, data_dir: Path, basename: str, mdoc_path: Path | None = None) -> tuple[float | None, str | None]:
    """Prefer IMOD align.com/log RotationAngle, then SerialEM metadata."""
    preferred_files = [
        imod_dir / "align.com",
        imod_dir / "align.log",
        imod_dir / "newst.log",
        imod_dir / "tilt.log",
    ]

    for path in preferred_files[:2]:
        value = parse_key_value_number(read_text(path), ["RotationAngle"])
        if value is not None:
            return value, f"{path}: RotationAngle"

    pat = re.compile(rf"(?i)Tilt\s+axis\s+angle\s*=\s*({NUMBER_RE})")
    fallback_files = preferred_files[2:]
    if mdoc_path is not None:
        fallback_files.append(mdoc_path)
    else:
        fallback_files.extend([
            data_dir / f"{basename}.mrc.mdoc",
            data_dir / f"{basename}.mdoc",
            imod_dir / f"{basename}.mrc.mdoc",
            imod_dir / f"{basename}.mdoc",
        ])
    for path in fallback_files:
        m = pat.search(read_text(path))
        if m:
            return float(m.group(1)), f"{path}: Tilt axis angle"

    return None, None

def strip_known_imod_suffixes(stem: str) -> str:
    """Return a likely eTomo series basename from an IMOD-generated filename stem."""
    suffixes = [
        "_full_rec", "_even_rec", "_odd_rec", "_rec", "_ali", "_preali",
        "_fid", "_nogaps", "_orig", ".rawtlt", ".tltxf", ".prexf", ".prexg",
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem


def detect_series_basename(data_dir: Path) -> str:
    """Detect the real IMOD series basename when directory names differ."""
    env = os.environ.get("MISSALIGN_SERIES_BASENAME") or os.environ.get("SERIES_BASENAME")
    if env:
        return env.strip()

    name = data_dir.name
    for suffix in ("_Imod", "_IMOD", "_imod"):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    exact_nested = sorted(
        [p for p in safe_iterdir(data_dir) if p.is_dir() and p.name.lower().endswith("_imod")],
        key=lambda p: p.name,
    )
    for p in exact_nested:
        candidate = re.sub(r"(?i)_imod$", "", p.name)
        if safe_exists(data_dir / f"{candidate}.mrc") or safe_exists(data_dir / f"{candidate}.st"):
            return candidate

    search_dirs = [data_dir] + exact_nested
    for root in search_dirs:
        texts = [read_text(root / "newst.com"), read_text(root / "tilt.com"), read_text(root / "align.com")]
        for text in texts:
            for keys in (["InputFile", "OutputFile", "TransformFile"], ["TiltFile", "TILTFILE", "OutputTiltFile"]):
                val = parse_key_value_string(text, keys)
                if not val:
                    continue
                stem = strip_known_imod_suffixes(Path(val).stem)
                for base in search_dirs:
                    if any(safe_exists(base / f"{stem}{ext}") for ext in [".mrc", ".st", ".tlt", ".xf", ".rawtlt"]):
                        return stem

    for f in sorted(safe_glob(data_dir, "*.mrc")) + sorted(safe_glob(data_dir, "*.st")):
        if f.name.endswith("~"):
            continue
        stem = strip_known_imod_suffixes(f.stem)
        if safe_exists(data_dir / f"{stem}.rawtlt") or safe_exists(data_dir / f"{stem}.mrc.mdoc"):
            return stem
    return data_dir.name


def imod_dir_score(path: Path, basename: str) -> int:
    score = 0
    for name in ("align.com", "newst.com", "tilt.com"):
        if safe_is_file(path / name):
            score += 10
    for name in (f"{basename}.xf", f"{basename}.tlt", f"{basename}.ali", f"{basename}_ali.mrc", f"{basename}.rec"):
        if safe_exists(path / name):
            score += 5
    score += min(sum(1 for _ in safe_glob(path, "*.xf")), 2)
    score += min(sum(1 for _ in safe_glob(path, "*.tlt")), 2)
    return score


def detect_imod_dir(data_dir: Path, basename: str) -> Path:
    """Find the directory containing eTomo command/alignment files."""
    env = os.environ.get("MISSALIGN_IMOD_DIR")
    if env:
        path = Path(env)
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not safe_is_dir(path):
            raise SystemExit(f"ERROR: MISSALIGN_IMOD_DIR is not a directory: {path}")
        return path

    # Prefer an exact nested IMOD directory before any scoring.  In many
    # projects the raw stack and .rawtlt live in data_dir, while .xf/.tlt/.com
    # files live in <basename>_Imod.  Scoring can be unreliable when some
    # auxiliary files are not readable, so an exact directory name is
    # authoritative.
    for name in (f"{basename}_Imod", f"{basename}_IMOD", f"{basename}_imod"):
        exact = data_dir / name
        if safe_is_dir(exact):
            return exact.resolve()

    candidates = [data_dir]
    for name in ("Imod", "IMOD", "imod"):
        p = data_dir / name
        if safe_is_dir(p) and p not in candidates:
            candidates.append(p)
    for p in sorted(safe_iterdir(data_dir)):
        if safe_is_dir(p) and p not in candidates and p.name.lower().endswith("_imod"):
            candidates.append(p)

    scored = [(imod_dir_score(p, basename), p) for p in candidates]
    scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    best_score, best = scored[0]
    if best_score == 0:
        return data_dir
    return best.resolve()


def resolve_named_path(name: str | None, roots: Iterable[Path]) -> Path | None:
    if not name:
        return None
    p = Path(name)
    if p.is_absolute():
        return p if safe_readable_file(p) else None
    for root in roots:
        candidate = root / p
        if safe_readable_file(candidate):
            return candidate
    return None

def env_path(name: str, base: Path | None = None) -> Path | None:
    val = os.environ.get(name)
    if not val:
        return None
    p = Path(val)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


def parse_xyz_env(name: str) -> list[int] | None:
    val = os.environ.get(name)
    if not val:
        return None
    parts = [x for x in re.split(r"[xX,;\s]+", val.strip()) if x]
    if len(parts) != 3:
        raise SystemExit(f"ERROR: {name} must be three integers, e.g. 1024x1440x440; got {val!r}")
    return [int(x) for x in parts]


def parse_float_env(name: str) -> float | None:
    val = os.environ.get(name)
    if not val:
        return None
    return float(val)


def parse_int_env(name: str) -> int | None:
    val = os.environ.get(name)
    if not val:
        return None
    return int(val)

def mrc_info(path: Path | None) -> MRCInfo | None:
    if path is None:
        return None
    info = MRCInfo(path=norm_path(path), exists=safe_exists(path))
    if not safe_exists(path):
        return info
    try:
        with mrcfile.open(path, permissive=True) as m:
            info.shape_zyx = [int(x) for x in m.data.shape]
            info.header_nxyz = [int(m.header.nx), int(m.header.ny), int(m.header.nz)]
            info.voxel_size_A = [float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)]
            info.dtype = str(m.data.dtype)
            info.cella = [float(m.header.cella.x), float(m.header.cella.y), float(m.header.cella.z)]
            info.origin = [float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)]
    except Exception as exc:
        info.error = repr(exc)
    return info


def safe_round_int(value: float) -> int:
    return int(round(float(value)))




def unique_readable_glob(roots: Iterable[Path], patterns: Iterable[str], *, label: str) -> Path | None:
    """Return the sole readable glob candidate, or fail clearly when ambiguous."""
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for pattern in patterns:
            for candidate in sorted(safe_glob(root, pattern)):
                if not safe_readable_file(candidate):
                    continue
                key = str(candidate.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(candidate)
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    names = "\n".join(f"    - {x}" for x in found)
    raise SystemExit(
        f"ERROR: multiple readable candidates found for {label}; set the corresponding "
        f"value explicitly in project_settings.toml:\n{names}"
    )

def choose_stack_files(data_dir: Path, imod_dir: Path, basename: str) -> dict[str, Path | None]:
    raw_roots = [data_dir, imod_dir]
    derived_roots = [imod_dir, data_dir]

    def candidates(roots: list[Path], names: list[str]) -> list[Path]:
        return [root / name for root in roots for name in names]

    raw_stack = first_existing(candidates(raw_roots, [f"{basename}.mrc", f"{basename}.st"]))
    aligned_stack = first_existing(candidates(derived_roots, [
        f"{basename}_ali.mrc", f"{basename}_ali.st", f"{basename}.ali",
    ]))
    if aligned_stack is None:
        aligned_stack = unique_readable_glob(
            [imod_dir], ["*.ali", "*_ali.mrc", "*_ali.st"], label="aligned stack"
        )

    prealigned_stack = first_existing(candidates(derived_roots, [
        f"{basename}_preali.mrc", f"{basename}_preali.st", f"{basename}.preali",
    ]))
    if prealigned_stack is None:
        prealigned_stack = unique_readable_glob(
            [imod_dir], ["*.preali", "*_preali.mrc", "*_preali.st"], label="prealigned stack"
        )

    # A reconstruction stack is used only to infer target geometry.  Explicit
    # --target-volume-xyz makes it unnecessary, and an explicit reconstruction
    # override is authoritative.  Prefer the non-half-map reconstruction when
    # several full/even/odd files are present.
    explicit_rec = env_path("MISSALIGN_RECONSTRUCTION_STACK", data_dir)
    manual_xyz = os.environ.get("MISSALIGN_TARGET_VOLUME_XYZ", "").strip()
    if explicit_rec is not None:
        rec_stack = explicit_rec
    elif manual_xyz:
        rec_stack = None
    else:
        preferred_rec_names = [
            f"{basename}_rec.mrc", f"{basename}.rec", f"{basename}_full_rec.mrc",
            f"{basename}_rec.st", f"{basename}_full_rec.st",
        ]
        rec_stack = first_existing(candidates(derived_roots, preferred_rec_names))
        if rec_stack is None:
            # Do not choose even/odd half reconstructions automatically.  If a
            # single non-half candidate remains, use it; otherwise report the
            # ambiguity so the top-level setup can offer a numbered choice.
            all_rec = []
            seen = set()
            for root in derived_roots:
                for pattern in ("*.rec", "*_rec.mrc", "*_full_rec.mrc", "*_rec.st", "*_full_rec.st"):
                    for candidate in sorted(safe_glob(root, pattern)):
                        if not safe_readable_file(candidate):
                            continue
                        key = str(candidate.resolve())
                        if key in seen:
                            continue
                        seen.add(key)
                        all_rec.append(candidate)
            non_half = [x for x in all_rec if not re.search(r"(?:^|_)(?:even|odd)(?:_|\.)", x.name, re.I)]
            pool = non_half or all_rec
            if len(pool) == 1:
                rec_stack = pool[0]
            elif len(pool) > 1:
                names = "\n".join(f"    - {x}" for x in pool)
                raise SystemExit(
                    "ERROR: multiple readable candidates found for reconstruction stack; "
                    "use --reconstruction-stack FILE or --xyz X,Y,Z:\n" + names
                )

    return {
        "raw_stack": raw_stack,
        "aligned_stack": aligned_stack,
        "prealigned_stack": prealigned_stack,
        "reconstruction_stack": rec_stack,
    }

def choose_metadata_files(data_dir: Path, imod_dir: Path, basename: str) -> dict[str, Path | None]:
    roots = [imod_dir, data_dir]
    align_com = read_text(imod_dir / "align.com")
    newst_com = read_text(imod_dir / "newst.com")
    tilt_com = read_text(imod_dir / "tilt.com")

    output_tilt_name = parse_key_value_string(align_com, ["OutputTiltFile"])
    transform_name = parse_key_value_string(newst_com, ["TransformFile"])
    tiltfile_from_tilt = parse_key_value_string(tilt_com, ["TILTFILE", "TiltFile"])

    final_tilt = first_existing([
        p for p in [
            resolve_named_path(output_tilt_name, roots),
            resolve_named_path(tiltfile_from_tilt, roots),
            imod_dir / f"{basename}.tlt",
            imod_dir / f"{basename}_fid.tlt",
            data_dir / f"{basename}.tlt",
            data_dir / f"{basename}.rawtlt",
            imod_dir / f"{basename}.rawtlt",
        ] if p is not None
    ])
    if final_tilt is None:
        final_tilt = unique_readable_glob([imod_dir], ["*.tlt", "*.rawtlt"], label="final tilt file")
    raw_tilt = first_existing([
        data_dir / f"{basename}.rawtlt",
        imod_dir / f"{basename}.rawtlt",
        data_dir / f"{basename}.tlt",
        imod_dir / f"{basename}.tlt",
    ])
    final_xf = first_existing([
        p for p in [
            resolve_named_path(transform_name, roots),
            imod_dir / f"{basename}.xf",
            imod_dir / f"{basename}_fid.xf",
            imod_dir / f"{basename}.tltxf",
            data_dir / f"{basename}.xf",
            data_dir / f"{basename}.tltxf",
        ] if p is not None
    ])
    if final_xf is None:
        final_xf = unique_readable_glob([imod_dir], ["*.xf", "*.tltxf"], label="final transform file")
    tltxf = first_existing([
        imod_dir / f"{basename}.tltxf",
        imod_dir / f"{basename}_fid.xf",
        data_dir / f"{basename}.tltxf",
    ])
    mdoc = first_existing([
        data_dir / f"{basename}.mrc.mdoc",
        data_dir / f"{basename}.mdoc",
        imod_dir / f"{basename}.mrc.mdoc",
        imod_dir / f"{basename}.mdoc",
    ])

    return {
        "final_tilt": final_tilt,
        "raw_tilt": raw_tilt,
        "final_xf": final_xf,
        "tltxf": tltxf,
        "mdoc": mdoc,
    }

def parse_imod_scalars(imod_dir: Path, mdoc_path: Path | None) -> dict[str, Any]:
    align_text = read_text(imod_dir / "align.com") + "\n" + read_text(imod_dir / "align.log")
    tilt_text = read_text(imod_dir / "tilt.com") + "\n" + read_text(imod_dir / "tilt.log")
    mdoc_text = read_text(mdoc_path) if mdoc_path is not None else ""

    unbinned_pixel_nm = parse_key_value_number(align_text, ["UnbinnedPixelSize"])
    image_binned = parse_key_value_number(tilt_text, ["IMAGEBINNED", "ImageBinned"])
    thickness = parse_key_value_number(tilt_text, ["THICKNESS", "Thickness"])
    fullimage_yx = parse_pair_ints(tilt_text, ["FULLIMAGE", "FullImage"])

    m = re.search(rf"(?im)^\s*PixelSpacing\s*=\s*({NUMBER_RE})", mdoc_text)
    mdoc_pixel_A = float(m.group(1)) if m else None

    # IMOD tomogram-positioning (OFFSET / XAXISTILT / SHIFT). tilt.com is authoritative:
    # parsed from tilt.com ALONE by the canonical geometry.imod_positioning module, never
    # from the concatenated tilt.com+tilt.log text (which could accept a stale log entry).
    # tilt.log is used only as an explicitly-recorded per-field fallback.
    unbinned_pixel_A = unbinned_pixel_nm * 10.0 if unbinned_pixel_nm is not None else mdoc_pixel_A
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from geometry.imod_positioning import parse_imod_positioning

    positioning = parse_imod_positioning(
        imod_dir / "tilt.com",
        unbinned_pixel_size_A=unbinned_pixel_A,
        thickness_unbinned_px=int(thickness) if thickness is not None else None,
        tilt_log_path=imod_dir / "tilt.log",
    )

    return {
        "unbinned_pixel_size_nm_from_align": unbinned_pixel_nm,
        "unbinned_pixel_size_A_from_align": unbinned_pixel_nm * 10.0 if unbinned_pixel_nm is not None else None,
        "image_binned_from_tilt": int(image_binned) if image_binned is not None else None,
        "thickness_unbinned_px_from_tilt": int(thickness) if thickness is not None else None,
        "fullimage_yx_from_tilt": fullimage_yx,
        "pixel_spacing_A_from_mdoc": mdoc_pixel_A,
        # canonical IMOD positioning: full manifest and the resolved TOML table
        "imod_positioning": positioning.to_manifest(),
        "imod_positioning_table": positioning.to_toml_table(),
    }

def build_params(etomo_dir: Path) -> dict[str, Any]:
    data_dir = etomo_dir.resolve()
    basename = detect_series_basename(data_dir)
    imod_dir = detect_imod_dir(data_dir, basename)
    warnings: list[str] = []

    stacks = choose_stack_files(data_dir, imod_dir, basename)
    metadata_files = choose_metadata_files(data_dir, imod_dir, basename)

    # Optional manual overrides. These are mainly used by wrapper scripts when an
    # eTomo directory has non-standard names or incomplete command files. Paths may
    # be absolute or relative to --etomo-dir.
    stack_overrides = {
        "raw_stack": env_path("MISSALIGN_RAW_STACK", data_dir),
        "aligned_stack": env_path("MISSALIGN_ALIGNED_STACK", data_dir),
        "prealigned_stack": env_path("MISSALIGN_PREALIGNED_STACK", data_dir),
        "reconstruction_stack": env_path("MISSALIGN_RECONSTRUCTION_STACK", data_dir),
    }
    for key, value in stack_overrides.items():
        if value is not None:
            stacks[key] = value

    metadata_overrides = {
        "final_tilt": env_path("MISSALIGN_FINAL_TILT_FILE", data_dir),
        "raw_tilt": env_path("MISSALIGN_RAW_TILT_FILE", data_dir),
        "final_xf": env_path("MISSALIGN_FINAL_XF_FILE", data_dir),
        "tltxf": env_path("MISSALIGN_TLTXF_FILE", data_dir),
        "mdoc": env_path("MISSALIGN_MDOC_FILE", data_dir),
    }
    for key, value in metadata_overrides.items():
        if value is not None:
            metadata_files[key] = value

    scalars = parse_imod_scalars(imod_dir, metadata_files.get("mdoc"))
    tilt_axis, tilt_axis_source = parse_tilt_axis_angle(
        imod_dir, data_dir, basename, metadata_files.get("mdoc")
    )
    # Optional explicit override (project_settings.toml tilt_axis_angle_deg /
    # setup_missalign_project.py --tilt-axis-angle, threaded through the shell
    # front-end as MISSALIGN_TILT_AXIS_ANGLE).  Use an explicit None test so a
    # legitimate 0-degree tilt axis is honoured rather than treated as unset.
    tilt_axis_override = parse_float_env("MISSALIGN_TILT_AXIS_ANGLE")
    if tilt_axis_override is not None:
        tilt_axis = tilt_axis_override
        tilt_axis_source = "override (MISSALIGN_TILT_AXIS_ANGLE / --tilt-axis-angle)"

    infos: dict[str, Any] = {}
    for key, path in stacks.items():
        infos[key] = asdict(mrc_info(path)) if path is not None else None

    raw_info = mrc_info(stacks["raw_stack"])
    ali_info = mrc_info(stacks["aligned_stack"])
    rec_info = mrc_info(stacks["reconstruction_stack"])

    raw_pixel = None
    if raw_info and raw_info.voxel_size_A:
        raw_pixel = float(raw_info.voxel_size_A[0])
    elif scalars["pixel_spacing_A_from_mdoc"] is not None:
        raw_pixel = float(scalars["pixel_spacing_A_from_mdoc"])
    elif scalars["unbinned_pixel_size_A_from_align"] is not None:
        raw_pixel = float(scalars["unbinned_pixel_size_A_from_align"])

    aligned_pixel = None
    if ali_info and ali_info.voxel_size_A:
        aligned_pixel = float(ali_info.voxel_size_A[0])

    target_pixel = None
    if rec_info and rec_info.voxel_size_A:
        target_pixel = float(rec_info.voxel_size_A[0])
    elif aligned_pixel is not None:
        target_pixel = aligned_pixel
    elif raw_pixel is not None:
        b = scalars.get("image_binned_from_tilt")
        target_pixel = raw_pixel * b if b else raw_pixel

    # Optional pixel-size overrides. Values are in Angstrom/pixel.
    raw_pixel = parse_float_env("MISSALIGN_RAW_PIXEL_SIZE_A") or raw_pixel
    aligned_pixel = parse_float_env("MISSALIGN_ALIGNED_PIXEL_SIZE_A") or aligned_pixel
    target_pixel = parse_float_env("MISSALIGN_TARGET_PIXEL_SIZE_A") or target_pixel

    target_volume_xyz = None
    if rec_info and rec_info.header_nxyz:
        # MRC header order is X,Y,Z; this is also what etomo_to_warp.py expects.
        target_volume_xyz = [int(x) for x in rec_info.header_nxyz]
    elif ali_info and ali_info.header_nxyz:
        z = None
        if scalars.get("thickness_unbinned_px_from_tilt") and scalars.get("image_binned_from_tilt"):
            z = safe_round_int(scalars["thickness_unbinned_px_from_tilt"] / scalars["image_binned_from_tilt"])
        target_volume_xyz = [int(ali_info.header_nxyz[0]), int(ali_info.header_nxyz[1]), int(z or ali_info.header_nxyz[2])]

    raw_volume_xyz = None
    ali_volume_xyz = None
    if target_volume_xyz and target_pixel and raw_pixel:
        physical = [target_volume_xyz[i] * target_pixel for i in range(3)]
        raw_volume_xyz = [safe_round_int(x / raw_pixel) for x in physical]
    if target_volume_xyz and target_pixel and aligned_pixel:
        physical = [target_volume_xyz[i] * target_pixel for i in range(3)]
        ali_volume_xyz = [safe_round_int(x / aligned_pixel) for x in physical]

    # Optional volume-shape overrides. X,Y,Z order, matching etomo_to_warp.py.
    manual_target_xyz = parse_xyz_env("MISSALIGN_TARGET_VOLUME_XYZ")
    manual_raw_xyz = parse_xyz_env("MISSALIGN_RAW_VOLUME_XYZ")
    manual_ali_xyz = parse_xyz_env("MISSALIGN_ALIGNED_VOLUME_XYZ")
    if manual_target_xyz is not None:
        target_volume_xyz = manual_target_xyz
    if manual_raw_xyz is not None:
        raw_volume_xyz = manual_raw_xyz
    elif manual_target_xyz is not None and target_pixel and raw_pixel:
        physical = [target_volume_xyz[i] * target_pixel for i in range(3)]
        raw_volume_xyz = [safe_round_int(x / raw_pixel) for x in physical]
    if manual_ali_xyz is not None:
        ali_volume_xyz = manual_ali_xyz
    elif manual_target_xyz is not None and target_pixel and aligned_pixel:
        physical = [target_volume_xyz[i] * target_pixel for i in range(3)]
        ali_volume_xyz = [safe_round_int(x / aligned_pixel) for x in physical]

    # counts
    counts = {
        "final_tilt_angles": count_nonempty_lines(metadata_files["final_tilt"]),
        "raw_tilt_angles": count_nonempty_lines(metadata_files["raw_tilt"]),
        "final_xf_rows": count_nonempty_lines(metadata_files["final_xf"]),
        "tltxf_rows": count_nonempty_lines(metadata_files["tltxf"]),
    }
    if raw_info and raw_info.shape_zyx:
        counts["raw_stack_tilts"] = int(raw_info.shape_zyx[0])
    if ali_info and ali_info.shape_zyx:
        counts["aligned_stack_tilts"] = int(ali_info.shape_zyx[0])
    manual_raw_tilts = parse_int_env("MISSALIGN_RAW_STACK_TILTS")
    if manual_raw_tilts is not None:
        counts["raw_stack_tilts"] = manual_raw_tilts

    for label in ["final_tilt_angles", "final_xf_rows", "raw_stack_tilts"]:
        if counts.get(label) is None:
            warnings.append(f"Could not determine {label}.")
    if counts.get("final_tilt_angles") and counts.get("final_xf_rows") and counts["final_tilt_angles"] != counts["final_xf_rows"]:
        warnings.append("final tilt-file line count differs from final .xf row count.")
    if counts.get("raw_stack_tilts") and counts.get("final_tilt_angles") and counts["raw_stack_tilts"] != counts["final_tilt_angles"]:
        warnings.append("raw stack tilt count differs from final tilt-file line count.")
    if counts.get("aligned_stack_tilts") and counts.get("final_tilt_angles") and counts["aligned_stack_tilts"] != counts["final_tilt_angles"]:
        warnings.append("aligned stack tilt count differs from final tilt-file line count.")
    if tilt_axis is None:
        warnings.append("Could not determine tilt-axis angle; provide --tilt-axis-angle later.")
    if target_volume_xyz is None:
        warnings.append("Could not determine target volume shape; provide it manually.")

    files = {k: norm_path(v) if v else None for k, v in {**stacks, **metadata_files}.items()}

    conditions = {
        "raw_xf": {
            "stack": files["raw_stack"],
            "tilt_file": files["final_tilt"],
            "xf_file": files["final_xf"],
            "source_xf_file": files["final_xf"],
            "volume_shape_xyz": raw_volume_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "translation",
            "axis_frame": "raw",
            "description": "backward-compatible alias: raw stack + translation-only IMOD metadata",
        },
        "raw_xf_translation": {
            "stack": files["raw_stack"],
            "tilt_file": files["final_tilt"],
            "xf_file": files["final_xf"],
            "source_xf_file": files["final_xf"],
            "volume_shape_xyz": raw_volume_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "translation",
            "axis_frame": "raw",
            "description": "raw stack + translation component derived from final IMOD .xf",
        },
        "raw_xf_affine_fixed": {
            "stack": files["raw_stack"],
            "tilt_file": files["final_tilt"],
            "xf_file": files["final_xf"],
            "source_xf_file": files["final_xf"],
            "volume_shape_xyz": raw_volume_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "full-affine",
            "axis_frame": "aligned",
            "description": "raw stack + complete fixed IMOD affine represented by offsets and movement grids",
        },
        "ali_identity": {
            "stack": files["aligned_stack"],
            "tilt_file": files["final_tilt"],
            "xf_file": "IDENTITY",
            "source_xf_file": files["final_xf"],
            "volume_shape_xyz": ali_volume_xyz or target_volume_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "identity",
            "axis_frame": "aligned",
            "description": "aligned stack + identity metadata; stack may be generated automatically with IMOD newstack",
        },
        "raw_identity": {
            "stack": files["raw_stack"],
            "tilt_file": files["final_tilt"],
            "xf_file": "IDENTITY",
            "source_xf_file": files["final_xf"],
            "volume_shape_xyz": raw_volume_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "identity",
            "axis_frame": "raw",
            "description": "raw stack + identity transforms, control without XY alignment",
        },
    }

    params = {
        "schema_version": 2,
        "series_name": basename,
        "etomo_dir": norm_path(data_dir),
        "imod_dir": norm_path(imod_dir),
        "files": files,
        "mrc_headers": infos,
        "counts": counts,
        "imod_parameters": scalars,
        "geometry": {
            "tilt_axis_angle_deg": tilt_axis,
            "tilt_axis_angle_source": tilt_axis_source,
            "raw_pixel_size_A": raw_pixel,
            "aligned_pixel_size_A": aligned_pixel,
            "target_output_pixel_size_A": target_pixel,
            "target_volume_shape_xyz": target_volume_xyz,
            "raw_volume_shape_xyz_for_converter": raw_volume_xyz,
            "aligned_volume_shape_xyz_for_converter": ali_volume_xyz or target_volume_xyz,
            # canonical IMOD tilt.com positioning table (mirrors imod_parameters), so the
            # legacy 02_convert path can propagate it without digging into imod_parameters.
            "imod_positioning": scalars.get("imod_positioning_table"),
        },
        "conditions": conditions,
        "warnings": warnings,
    }
    return params


def write_summary(params: dict[str, Any], path: Path) -> None:
    g = params["geometry"]
    files = params["files"]
    counts = params["counts"]
    cond = params["conditions"]

    def base(p: str | None) -> str:
        if p is None:
            return "MISSING"
        return Path(p).name if p != "IDENTITY" else "IDENTITY"

    lines = []
    lines.append(f"series_name:                 {params['series_name']}")
    lines.append(f"etomo_dir:                   {params['etomo_dir']}")
    lines.append(f"imod_dir:                    {params.get('imod_dir', params['etomo_dir'])}")
    lines.append("")
    lines.append(f"tilt-axis angle:             {g['tilt_axis_angle_deg']} deg")
    lines.append(f"tilt-axis angle source:      {g['tilt_axis_angle_source']}")
    lines.append(f"raw stack:                   {base(files.get('raw_stack'))}")
    lines.append(f"aligned stack:               {base(files.get('aligned_stack'))}")
    lines.append(f"final tilt file:             {base(files.get('final_tilt'))}")
    lines.append(f"final transform file:        {base(files.get('final_xf'))}")
    lines.append("")
    lines.append(f"raw pixel size:              {g['raw_pixel_size_A']} Å/px")
    lines.append(f"aligned pixel size:          {g['aligned_pixel_size_A']} Å/px")
    lines.append(f"target output pixel size:    {g['target_output_pixel_size_A']} Å/px")
    lines.append(f"target volume xyz:           {g['target_volume_shape_xyz']}")
    lines.append(f"raw volume xyz for converter:{g['raw_volume_shape_xyz_for_converter']}")
    lines.append(f"ali volume xyz for converter:{g['aligned_volume_shape_xyz_for_converter']}")
    lines.append("")
    lines.append("counts:")
    for key, value in counts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("conditions:")
    for name, c in cond.items():
        lines.append(f"  {name}:")
        lines.append(f"    stack: {base(c['stack'])}")
        lines.append(f"    tilt_file: {base(c['tilt_file'])}")
        lines.append(f"    xf_file: {base(c['xf_file'])}")
        lines.append(f"    source_xf_file: {base(c.get('source_xf_file'))}")
        lines.append(f"    alignment_mode: {c.get('alignment_mode')}")
        lines.append(f"    axis_frame: {c.get('axis_frame')}")
        lines.append(f"    volume_shape_xyz: {c['volume_shape_xyz']}")
        lines.append(f"    output_pixel_size_A: {c['output_pixel_size_A']}")
    if params.get("warnings"):
        lines.append("")
        lines.append("warnings:")
        for w in params["warnings"]:
            lines.append(f"  - {w}")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract eTomo/IMOD parameters for MissAlignment conversion.")
    parser.add_argument("--etomo-dir", required=True, type=Path, help="Input eTomo/IMOD tilt-series directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory where params JSON/report will be written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--basename", default=None, help="Override IMOD series basename, e.g. 64x_Vero_02.")
    parser.add_argument("--imod-dir", default=None, help="Directory containing eTomo command/.xf/.tlt files; absolute or relative to --etomo-dir.")
    parser.add_argument("--raw-stack", default=None, help="Override raw stack path, absolute or relative to --etomo-dir.")
    parser.add_argument("--aligned-stack", default=None, help="Override aligned stack path.")
    parser.add_argument("--reconstruction-stack", default=None, help="Override reconstruction stack path used to get target volume shape.")
    parser.add_argument("--raw-tilt-file", default=None, help="Override raw tilt file path.")
    parser.add_argument("--final-tilt-file", default=None, help="Override final tilt file path.")
    parser.add_argument("--final-xf-file", default=None, help="Override final IMOD .xf path.")
    parser.add_argument("--target-volume-xyz", default=None, help="Override target volume X,Y,Z, e.g. 1024x1440x440.")
    parser.add_argument("--raw-volume-xyz", default=None, help="Override raw converter volume X,Y,Z.")
    parser.add_argument("--aligned-volume-xyz", default=None, help="Override aligned converter volume X,Y,Z.")
    parser.add_argument("--target-pixel-size-A", default=None, help="Override target output pixel size in Angstrom/pixel.")
    parser.add_argument("--raw-pixel-size-A", default=None, help="Override raw pixel size in Angstrom/pixel.")
    parser.add_argument("--aligned-pixel-size-A", default=None, help="Override aligned pixel size in Angstrom/pixel.")
    parser.add_argument("--raw-stack-tilts", default=None, help="Override raw stack tilt count.")
    parser.add_argument("--tilt-axis-angle", default=None, help="Override tilt-axis angle in degrees (used when IMOD files lack a parseable RotationAngle).")
    args = parser.parse_args()

    cli_to_env = {
        "basename": "MISSALIGN_SERIES_BASENAME",
        "imod_dir": "MISSALIGN_IMOD_DIR",
        "raw_stack": "MISSALIGN_RAW_STACK",
        "aligned_stack": "MISSALIGN_ALIGNED_STACK",
        "reconstruction_stack": "MISSALIGN_RECONSTRUCTION_STACK",
        "raw_tilt_file": "MISSALIGN_RAW_TILT_FILE",
        "final_tilt_file": "MISSALIGN_FINAL_TILT_FILE",
        "final_xf_file": "MISSALIGN_FINAL_XF_FILE",
        "target_volume_xyz": "MISSALIGN_TARGET_VOLUME_XYZ",
        "raw_volume_xyz": "MISSALIGN_RAW_VOLUME_XYZ",
        "aligned_volume_xyz": "MISSALIGN_ALIGNED_VOLUME_XYZ",
        "target_pixel_size_A": "MISSALIGN_TARGET_PIXEL_SIZE_A",
        "raw_pixel_size_A": "MISSALIGN_RAW_PIXEL_SIZE_A",
        "aligned_pixel_size_A": "MISSALIGN_ALIGNED_PIXEL_SIZE_A",
        "raw_stack_tilts": "MISSALIGN_RAW_STACK_TILTS",
        "tilt_axis_angle": "MISSALIGN_TILT_AXIS_ANGLE",
    }
    for attr, env_name in cli_to_env.items():
        val = getattr(args, attr)
        if val is not None:
            os.environ[env_name] = str(val)

    etomo_dir = args.etomo_dir.resolve()
    if not safe_exists(etomo_dir) or not safe_is_dir(etomo_dir):
        raise SystemExit(f"ERROR: --etomo-dir is not a directory: {etomo_dir}")

    out_dir = args.out_dir.resolve()
    if safe_exists(out_dir) and any(safe_iterdir(out_dir)) and not args.overwrite:
        raise SystemExit(f"ERROR: output directory is not empty: {out_dir}\nUse --overwrite to reuse it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = build_params(etomo_dir)
    json_path = out_dir / "etomo_missalign_params.json"
    report_path = out_dir / "etomo_missalign_params.txt"

    json_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    write_summary(params, report_path)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")
    print("\nSummary:\n")
    print(report_path.read_text())

    if params.get("warnings"):
        print("Warnings were reported. Check the summary before conversion.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
