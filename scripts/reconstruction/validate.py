#!/usr/bin/env python3
"""Reconstruction prerequisite + output validation."""
from __future__ import annotations

from pathlib import Path

from .model import ReconstructionError, ReconstructionRequest


def _count_lines(p) -> int:
    return sum(1 for ln in Path(p).read_text().splitlines() if ln.strip())


def validate_prerequisites(req: ReconstructionRequest, *, measure_sections=None) -> dict:
    """Check section/.xf/.tlt consistency before running. ``measure_sections`` is
    injected (path -> n) so this stays import-light."""
    rep = {}
    nt = _count_lines(req.tilt_file)
    rep["tilt_rows"] = nt
    if req.xf_file:
        rep["xf_rows"] = _count_lines(req.xf_file)
        if rep["xf_rows"] != nt:
            raise ReconstructionError(f"xf rows {rep['xf_rows']} != tilt rows {nt}")
    if measure_sections:
        stack = req.aligned_stack if req.input_mode == "aligned_stack" else req.raw_stack
        if stack:
            rep["sections"] = measure_sections(stack)
            if rep["sections"] != nt:
                raise ReconstructionError(f"stack sections {rep['sections']} != tilt rows {nt}")
    return rep


def validate_output(rec_path: Path, *, measure_mrc=None) -> dict:
    """Validate a produced reconstruction header (shape/finite)."""
    rec_path = Path(rec_path)
    if not rec_path.is_file() or rec_path.stat().st_size == 0:
        raise ReconstructionError(f"reconstruction missing/empty: {rec_path}")
    rep = {"path": str(rec_path), "size": rec_path.stat().st_size}
    if measure_mrc:
        rep.update(measure_mrc(rec_path))
    return rep
