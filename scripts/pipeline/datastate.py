#!/usr/bin/env python3
"""Explicit data-state model for the binning + CTF pipeline.

Every stack is a :class:`Stack` with validated state literals. CTF state is
NEVER inferred from a filename -- it is recorded explicitly by the step that
created the stack, together with the exact command. The manifest is the single
source of truth for which stack is selected for MissAlignment and for the final
reconstruction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ALIGNMENT_STATES = ("raw", "source_aligned", "working_aligned", "final_aligned")
CTF_STATES = ("uncorrected", "phase_flipped")
BINNING_STATES = ("source", "working", "preview")
INTENDED_USES = ("missalignment_input", "working_qc", "final_reconstruction",
                 "visualization_only", "input")

ROLES = (
    "source_raw", "source_aligned_uncorrected", "source_aligned_ctf",
    "working_raw", "working_aligned_uncorrected", "working_aligned_ctf",
    "working_selected", "working_reconstruction",
    "final_source_aligned_uncorrected", "final_source_aligned_ctf",
    "final_aligned_ctf",
    "preview",
)


@dataclass
class Stack:
    role: str
    path: str
    alignment_state: str
    ctf_state: str
    binning_state: str
    intended_use: str
    source_parent: str | None = None
    grid: dict | None = None
    interpolation_history: list[str] = field(default_factory=list)
    created_by_command: str | None = None
    allowed_for_missalignment: bool = False
    allowed_for_final_reconstruction: bool = False

    def __post_init__(self):
        self._check("role", self.role, ROLES)
        self._check("alignment_state", self.alignment_state, ALIGNMENT_STATES)
        self._check("ctf_state", self.ctf_state, CTF_STATES)
        self._check("binning_state", self.binning_state, BINNING_STATES)
        self._check("intended_use", self.intended_use, INTENDED_USES)
        # Safety invariants: a preview / visualization stack can never be used
        # for MissAlignment or the final reconstruction.
        if self.intended_use == "visualization_only" or self.binning_state == "preview":
            if self.allowed_for_missalignment or self.allowed_for_final_reconstruction:
                raise ValueError(f"{self.role}: visualization/preview stack cannot be allowed for "
                                 "missalignment or final reconstruction")
        # A working-binned stack can never be the final reconstruction input.
        if self.binning_state == "working" and self.allowed_for_final_reconstruction:
            raise ValueError(f"{self.role}: a working-binned stack cannot feed the final reconstruction")

    @staticmethod
    def _check(name, value, allowed):
        if value not in allowed:
            raise ValueError(f"{name}={value!r} not in {allowed}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_for_missalignment(uncorrected: Stack, ctf: Stack | None, ctf_mode: str) -> Stack:
    """Manifest-driven selection of the MissAlignment input stack.

    ``working``/``both`` select the CTF-corrected working stack; otherwise the
    uncorrected working stack. Selection never renames/overwrites a stack.
    """
    if ctf_mode in ("working", "both"):
        if ctf is None:
            raise ValueError("ctf.mode requires a working CTF stack but none was generated")
        chosen = ctf
    else:
        chosen = uncorrected
    return Stack(
        role="working_selected", path=chosen.path,
        alignment_state="working_aligned", ctf_state=chosen.ctf_state,
        binning_state="working", intended_use="missalignment_input",
        source_parent=chosen.role, grid=chosen.grid,
        interpolation_history=list(chosen.interpolation_history),
        created_by_command=f"select(ctf_mode={ctf_mode}) -> {chosen.role}",
        allowed_for_missalignment=True, allowed_for_final_reconstruction=False,
    )


def final_reconstruction_input(uncorrected: Stack, ctf: Stack | None, ctf_mode: str) -> Stack:
    """Select the FINAL reconstruction input (source resolution)."""
    if ctf_mode in ("final", "both"):
        if ctf is None:
            raise ValueError("ctf.mode final/both requires a final CTF stack")
        return ctf
    return uncorrected
