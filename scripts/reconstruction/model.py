#!/usr/bin/env python3
"""Reconstruction request/result types + validation (explicit inputs only)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

INPUT_MODES = ("raw_plus_xf", "aligned_stack")
EXECUTIONS = ("local", "slurm", "skip")
HALF_SPLITS = ("angle", "index")


class ReconstructionError(ValueError):
    pass


@dataclass
class ReconstructionRequest:
    output_dir: str
    aligned_stack: Optional[str] = None
    raw_stack: Optional[str] = None
    xf_file: Optional[str] = None
    tilt_file: Optional[str] = None
    xtilt_file: Optional[str] = None
    newst_com: Optional[str] = None
    tilt_com: Optional[str] = None
    input_mode: str = "aligned_stack"
    execution: str = "skip"
    halfmaps: bool = False
    half_split_mode: str = "angle"
    cluster_profile: str = "maxwell"
    fullimage_xy: Optional[tuple] = None        # SOURCE pixels (nx, ny)
    thickness: Optional[int] = None             # SOURCE pixels
    pixel_size_A: Optional[float] = None
    basename: str = "series"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconstructionResult:
    output_dir: str
    tilt_com: Optional[str] = None
    run_script: Optional[str] = None
    sbatch: Optional[str] = None
    output_rec: Optional[str] = None
    half_files: dict = field(default_factory=dict)
    executed: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_request(req: ReconstructionRequest) -> None:
    if req.input_mode not in INPUT_MODES:
        raise ReconstructionError(f"input_mode {req.input_mode!r} not in {INPUT_MODES}")
    if req.execution not in EXECUTIONS:
        raise ReconstructionError(f"execution {req.execution!r} not in {EXECUTIONS}")
    if req.half_split_mode not in HALF_SPLITS:
        raise ReconstructionError(f"half_split_mode {req.half_split_mode!r} not in {HALF_SPLITS}")
    if req.input_mode == "aligned_stack":
        if not req.aligned_stack:
            raise ReconstructionError("input_mode=aligned_stack requires aligned_stack")
    else:
        if not (req.raw_stack and req.xf_file):
            raise ReconstructionError("input_mode=raw_plus_xf requires raw_stack and xf_file")
    if not req.tilt_file:
        raise ReconstructionError("tilt_file is required")
    # explicit inputs must exist
    for label, p in (("aligned_stack", req.aligned_stack), ("raw_stack", req.raw_stack),
                     ("xf_file", req.xf_file), ("tilt_file", req.tilt_file),
                     ("xtilt_file", req.xtilt_file)):
        if p and not Path(p).is_file():
            raise ReconstructionError(f"{label} does not exist: {p}")
