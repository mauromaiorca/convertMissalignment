#!/usr/bin/env python3
"""Constrained alignment integration for MissAlignment (reference module).

This is the code the fork wires into the MissAlignment dispatcher. It maps the
``alignment: translation|rigid|similarity`` iteration setting to the single
constrained optimizer (``alignment_models.optimize_constrained_2d``) driven by the
**real** MissAlignment reconstruction/projector/scoring/loss via the
``reconstruct_and_score`` hook (Option B detector-field materialization).

It imports the repo's ``alignment_models`` package as the math source of truth, so
the constrained DOF, gauges, scopes, regularization, warm starts, telemetry and the
canonical result contract are shared with the locally-tested implementation — only
the projector hook differs (production vs the local grid_sample reference).

Dispatch contract (call from the MissAlignment per-iteration loop):

    run_constrained_iteration(
        alignment_mode="rigid",          # translation|rigid|similarity
        tilt_series=...,                 # the real TiltSeries
        reconstruct_and_score=...,       # REAL projector+scoring+loss closure
        initial_parameters=...,          # warm-started from the previous stage
        ...,
    ) -> ConstrainedResult

The ``reconstruct_and_score(field, *, model, params, tilt_angles, recon)`` closure
MUST: update the per-tilt geometry from ``field`` (the differentiable detector
movement field), call ``reconstruct_subvolumes`` (or equivalent), evaluate the
scoring network, and return the differentiable image loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve the repo's alignment_models (math source of truth).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

from alignment_models.constraints import GaugeConfig                       # noqa: E402
from alignment_models.materialize import WARM_START_PRESETS                # noqa: E402
from alignment_models.optimize_constrained_2d import (                     # noqa: E402
    OptimizerSettings, ReconstructionSettings, SafetyBounds,
    optimize_constrained_2d)
from alignment_models.parameter_scope import ScopeConfig                   # noqa: E402
from alignment_models.regularization import RegularizationConfig           # noqa: E402
from alignment_models.registry import get_model                           # noqa: E402

SUPPORTED_ALIGNMENTS = ("translation", "rigid", "similarity")


def default_scopes(mode: str) -> ScopeConfig:
    if mode == "similarity":
        return ScopeConfig(translation="per_tilt", rotation="per_tilt", isotropic_scale="global")
    if mode == "rigid":
        return ScopeConfig(translation="per_tilt", rotation="per_tilt")
    return ScopeConfig(translation="per_tilt")


def default_gauge(mode: str) -> GaugeConfig:
    return GaugeConfig(anchor_tilt="closest_to_zero",
                       zero_mean_rotation=(mode in ("rigid", "similarity")),
                       zero_mean_log_scale=(mode == "similarity"))


def warm_start(prev_mode: str, prev_params, new_mode: str):
    """translation->rigid (copy tx,ty; phi=0); rigid->similarity (copy tx,ty,phi; log_scale=0)."""
    import torch
    p = torch.as_tensor(prev_params, dtype=torch.float64)
    n = p.shape[0]
    target = get_model(new_mode)
    out = target.identity_params(n).clone()
    ncopy = min(p.shape[1], out.shape[1])
    out[:, :ncopy] = p[:, :ncopy]
    return out


def run_constrained_iteration(
    *, alignment_mode: str, reconstruct_and_score, initial_parameters, tilt_angles,
    reconstruction_settings: ReconstructionSettings, result_dir=None, telemetry_dir=None,
    stage_label="stage0", seed=None, optimizer_settings: OptimizerSettings | None = None,
    regularization: RegularizationConfig | None = None, device="cuda", dtype=None,
):
    if alignment_mode not in SUPPORTED_ALIGNMENTS:
        raise ValueError(f"unsupported constrained alignment mode {alignment_mode!r}; "
                         f"expected one of {SUPPORTED_ALIGNMENTS}")
    model = get_model(alignment_mode)
    return optimize_constrained_2d(
        alignment_model=model, initial_parameters=initial_parameters,
        reconstruct_and_score=reconstruct_and_score, tilt_angles=tilt_angles,
        parameter_scopes=default_scopes(alignment_mode), gauge=default_gauge(alignment_mode),
        regularization=regularization or RegularizationConfig(),
        optimizer_settings=optimizer_settings or OptimizerSettings(steps=500, lr=1e-3),
        reconstruction_settings=reconstruction_settings, safety_bounds=SafetyBounds(),
        telemetry_dir=telemetry_dir, result_dir=result_dir, stage_label=stage_label,
        seed=seed, device=device, dtype=dtype)


def staged_schedule(mode: str):
    """The warm-started stage schedule per mode (matches 03_run_missalignment)."""
    return WARM_START_PRESETS()[mode]["iteration_settings"]
