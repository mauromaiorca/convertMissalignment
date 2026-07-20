#!/usr/bin/env python3
"""Reusable IMOD reconstruction library (spec §16).

Extracted from the legacy ``setup_imod_recon.py`` so the canonical path takes
EXPLICIT inputs and never (a) searches arbitrarily for the latest MissAlignment
XML, (b) chooses a result without a manifest, or (c) exports the final ``.xf``
independently — those responsibilities belong to the Phase-3 ``finalize`` path
and the canonical result contract. ``setup_imod_recon.py`` remains as a thin
compatibility wrapper.
"""
from __future__ import annotations

from .command_files import build_tilt_com, patch_tilt_com
from .halfsets import split_halfsets
from .model import ReconstructionRequest, ReconstructionResult, validate_request
from .prepare import prepare_imod_reconstruction
from .validate import validate_output, validate_prerequisites

__all__ = [
    "ReconstructionRequest", "ReconstructionResult", "validate_request",
    "build_tilt_com", "patch_tilt_com", "split_halfsets",
    "prepare_imod_reconstruction", "validate_prerequisites", "validate_output",
]
