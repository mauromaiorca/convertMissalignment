#!/usr/bin/env python3
"""Generate/patch IMOD tilt.com for a SOURCE-resolution reconstruction.

Source geometry only: FULLIMAGE and THICKNESS in SOURCE pixels, IMAGEBINNED 1
(never working pixel size / IMAGEBINNED B). Source .com files are never edited in
place — a local copy is patched, preserving unknown directives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_tilt_com(*, in_stack: str, out_rec: str, tilt_file: str,
                   fullimage_xy: tuple, thickness: int,
                   xtilt_file: Optional[str] = None, radial=(0.35, 0.05)) -> str:
    nx, ny = int(fullimage_xy[0]), int(fullimage_xy[1])
    lines = [
        "# Source-resolution reconstruction tilt.com (IMAGEBINNED 1; source pixels).",
        "# Generated; never edit or overwrite the source tilt.com.",
        "$tilt",
        f"InputProjections {in_stack}",
        f"OutputFile {out_rec}",
        f"TILTFILE {tilt_file}",
        f"FULLIMAGE {nx} {ny}",
        f"THICKNESS {int(thickness)}",
        "IMAGEBINNED 1",
        f"RADIAL {radial[0]} {radial[1]}",
    ]
    if xtilt_file:
        lines.append(f"XTILTFILE {xtilt_file}")
    lines.append("$if (-e ./savework) ./savework")
    return "\n".join(lines) + "\n"


def patch_tilt_com(source_com: Path, local_com: Path, replacements: dict) -> dict:
    """Copy a source tilt.com locally and patch only validated directives.

    Preserves comments, ``$`` launchers, and unknown directives. Returns a report.
    """
    import shutil
    import sys
    source_com = Path(source_com)
    if not source_com.is_file():
        raise FileNotFoundError(f"source tilt.com not found: {source_com}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from patch_imod_scripts import directive_key, patch_standard_input_file
    before_lines = source_com.read_text(errors="replace").splitlines()
    before = {directive_key(ln): ln for ln in before_lines if directive_key(ln)}
    local_com.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_com, local_com)
    patch_standard_input_file(local_com, replacements, append_missing=True)
    after_lines = local_com.read_text().splitlines()
    after = {directive_key(ln): ln for ln in after_lines if directive_key(ln)}
    return {
        "source_com": str(source_com), "local_com": str(local_com),
        "patched": list(replacements),
        "before": {k: before.get(k) for k in replacements},
        "after": {k: after.get(k) for k in replacements},
        "source_unmodified": source_com.read_text(errors="replace").splitlines() == before_lines,
    }
