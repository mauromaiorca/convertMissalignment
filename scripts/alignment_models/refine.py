#!/usr/bin/env python3
"""Local, autograd-based refinement engine for the constrained models.

This is a REAL executable refinement forward pass that runs entirely locally on
**coordinate** data (per-tilt source -> target correspondences in aligned
physical Angstrom). It exercises the full chain that a MissAlignment
integration would use, minus the image-based warpylib sampling:

    params --(scopes)--> constrained matrices --> predicted coords
          --(coordinate MSE + regularization)--> loss --(autograd)--> optimizer
          --(staged schedule, gauge fixing)--> residual params --> exact .xf

It does NOT touch images or warpylib. The image-based MissAlignment forward
pass (sampling a tilt series and back-propagating a reconstruction/projection
loss) requires the installed warpylib/MissAlignment and a GPU and is therefore
NOT exercised here -- see the execution-readiness classification in
``LOCAL_LOGICAL_VALIDATION_REPORT.md``.

The coordinate target can be produced from real IMOD ``newstack`` marker
positions, so a synthetic-but-real end-to-end (known transform -> newstack ->
measure -> refine -> exact .xf) is possible locally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import torch
from .constraints import GaugeConfig, apply_gauge, gauge_report
from .parameter_scope import ScopeConfig, apply_scopes
from .regularization import RegularizationConfig, regularization_loss
from .refinement_config import RefinementConfig
from .registry import embed_params, get_model, model_rank


@dataclass
class RefineResult:
    model: str
    params: Any                      # torch.Tensor (n_tilts, n_params), final model
    converged: bool
    final_data_rms_A: float
    stage_history: list = field(default_factory=list)
    gauge: dict = field(default_factory=dict)


def _data_rms(model, params, source, target, center):
    pred = model.apply_centered(params, source, center)
    return torch.sqrt(((pred - target) ** 2).sum(-1).mean())


def _scope_for_stage(scope: ScopeConfig, stage_model: str) -> ScopeConfig:
    """Drop scopes for components the stage model does not have (avoids errors)."""
    rank = model_rank(stage_model)
    return ScopeConfig(
        translation=scope.translation,
        rotation=scope.rotation if rank >= model_rank("rigid") else "per_tilt",
        isotropic_scale=scope.isotropic_scale if stage_model in ("similarity", "affine") else "per_tilt",
        anisotropic_scale=scope.anisotropic_scale if stage_model == "affine" else "per_tilt",
        shear=scope.shear if stage_model == "affine" else "per_tilt",
        spline_control_points=scope.spline_control_points,
    )


def refine(
    config: RefinementConfig,
    source_points_A,
    target_points_A,
    tilt_angles,
    aligned_shape_xy,
    pixel_size_A: float,
    *,
    dtype=None,
    iters_per_stage: int = 250,
    lr: float = 0.05,
    tol_A: float = 1e-4,
    verbose: bool = False,
) -> RefineResult:
    """Refine a constrained residual against per-tilt coordinate correspondences.

    ``source_points_A`` / ``target_points_A``: ``(n_tilts, M, 2)`` aligned
    physical coordinates (Angstrom). Returns the fitted residual in the final
    model's parameter space, gauge-fixed.
    """
    dtype = dtype or torch.float64
    from . import coordinate_frames as cf
    center = torch.tensor(cf.physical_center_xy(aligned_shape_xy, pixel_size_A), dtype=dtype)
    source = torch.as_tensor(np.asarray(source_points_A), dtype=dtype)
    target = torch.as_tensor(np.asarray(target_points_A), dtype=dtype)
    n_tilts = source.shape[0]
    angles = list(np.asarray(tilt_angles, dtype=float).reshape(-1))

    stages = config.resolved_stages()
    final_model_name = config.model
    # Start in the first stage's model space (identity).
    cur_model_name = stages[0].model
    params = get_model(cur_model_name, dtype=dtype).identity_params(n_tilts)

    history = []
    for si, stage in enumerate(stages):
        # Warm-start: embed current params into this stage's model space.
        if model_rank(stage.model) < model_rank(cur_model_name):
            raise ValueError("schedule must be non-decreasing in model rank")
        params = embed_params(params, cur_model_name, stage.model)
        cur_model_name = stage.model
        model = get_model(cur_model_name, dtype=dtype)
        scope = _scope_for_stage(config.scope, cur_model_name)

        p = params.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        last = None
        for it in range(max(1, iters_per_stage * stage.max_epochs // 5 if stage.max_epochs else iters_per_stage)):
            opt.zero_grad()
            q = apply_scopes(cur_model_name, p, scope, angles)
            pred = model.apply_centered(q, source, center)
            data = ((pred - target) ** 2).mean()
            reg = regularization_loss(cur_model_name, q, angles, config.regularization)
            loss = data + reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                q_mon = apply_scopes(cur_model_name, p, scope, angles)
                last = float(_data_rms(model, q_mon, source, target, center))
            if last < tol_A:
                break
        params = apply_scopes(cur_model_name, p, scope, angles).detach()
        history.append({"stage": si, "model": cur_model_name, "data_rms_A": last, "iters": it + 1})
        if verbose:
            print(f"  stage {si} ({cur_model_name}): data RMS = {last:.4g} A")

    # Final embed to the configured model and gauge-fix.
    params = embed_params(params, cur_model_name, final_model_name)
    final_model = get_model(final_model_name, dtype=dtype)
    params = apply_gauge(final_model_name, params, angles, config.gauge).detach()
    final_rms = float(_data_rms(final_model, params, source, target, center))
    return RefineResult(
        model=final_model_name,
        params=params,
        converged=final_rms < max(tol_A, 1e-3),
        final_data_rms_A=final_rms,
        stage_history=history,
        gauge=gauge_report(final_model_name, params),
    )
