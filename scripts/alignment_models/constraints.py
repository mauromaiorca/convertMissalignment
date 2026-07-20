#!/usr/bin/env python3
"""Gauge fixing for residual refinement.

A residual transform field is not fully identifiable: a global rotation, a
global isotropic scale, a global shear, and an overall translation can be
absorbed into the tilt-axis angle, the pixel size, the specimen geometry, or
the volume position. These constraints remove those unobservable global degrees
of freedom in a deterministic, documented way. They do **not** modify tilt
angles, tilt-axis angles, pixel sizes, or volume dimensions.

Enforcement (parameter-space projections; ``params`` is ``(n_tilts, n_params)``):

- ``anchor_tilt = "closest_to_zero"``: subtract the anchor tilt's translation
  from every tilt, so the tilt nearest 0 degrees has zero residual translation.
- ``zero_mean_rotation``: subtract ``mean(phi)`` from every ``phi``.
- ``zero_mean_log_scale``: subtract the mean isotropic log-scale. For
  ``similarity`` this is ``log_scale``; for ``affine`` it is ``(alpha+beta)/2``,
  subtracted from both ``alpha`` and ``beta``.
- ``zero_mean_shear``: subtract ``mean(shear)`` from every ``shear`` (affine).

All projections preserve the *relative* geometry between tilts (pairwise
differences of each named component are unchanged); only the global gauge moves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import torch
from .registry import NESTING_ORDER, model_rank


@dataclass(frozen=True)
class GaugeConfig:
    anchor_tilt: str = "closest_to_zero"  # or "none"
    zero_mean_rotation: bool = True
    zero_mean_log_scale: bool = True
    zero_mean_shear: bool = True


def anchor_index(tilt_angles) -> int:
    angles = np.asarray(tilt_angles, dtype=float).reshape(-1)
    if angles.size == 0:
        raise ValueError("no tilt angles for anchoring")
    return int(np.argmin(np.abs(angles)))


def apply_gauge(model_name: str, params, tilt_angles, config: GaugeConfig):
    """Return gauge-fixed parameters.

    Built out-of-place (column list + ``torch.stack``) so it is safe to call on
    a ``requires_grad`` tensor inside an optimization loop; the input is never
    modified.
    """
    p = torch.as_tensor(params)
    rank = model_rank(model_name)
    cols = [p[:, i] for i in range(p.shape[1])]

    # Translation anchor.
    if config.anchor_tilt and config.anchor_tilt != "none":
        if config.anchor_tilt != "closest_to_zero":
            raise ValueError(f"unsupported anchor_tilt {config.anchor_tilt!r}")
        a = anchor_index(tilt_angles)
        cols[0] = cols[0] - cols[0][a]
        cols[1] = cols[1] - cols[1][a]

    # Rotation (rigid+).
    if config.zero_mean_rotation and rank >= model_rank("rigid"):
        cols[2] = cols[2] - cols[2].mean()

    # Isotropic log-scale.
    if config.zero_mean_log_scale:
        if model_name == "similarity":
            cols[3] = cols[3] - cols[3].mean()
        elif model_name == "affine":
            m = (0.5 * (cols[3] + cols[4])).mean()
            cols[3] = cols[3] - m
            cols[4] = cols[4] - m

    # Shear (affine).
    if config.zero_mean_shear and model_name == "affine":
        cols[5] = cols[5] - cols[5].mean()

    return torch.stack(cols, dim=1)


def gauge_report(model_name: str, params) -> dict:
    """Diagnostics describing the residual gauge (means that should be ~0)."""
    p = torch.as_tensor(params)
    out = {
        "mean_tx": float(p[:, 0].mean()),
        "mean_ty": float(p[:, 1].mean()),
    }
    rank = model_rank(model_name)
    if rank >= model_rank("rigid"):
        out["mean_phi"] = float(p[:, 2].mean())
    if model_name == "similarity":
        out["mean_log_scale"] = float(p[:, 3].mean())
    if model_name == "affine":
        out["mean_iso_log_scale"] = float((0.5 * (p[:, 3] + p[:, 4])).mean())
        out["mean_shear"] = float(p[:, 5].mean())
    return out
