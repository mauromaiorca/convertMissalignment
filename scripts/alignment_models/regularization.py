#!/usr/bin/env python3
"""Configurable priors and smoothness for residual refinement.

Separate priors for translation, rotation, scale (isotropic + anisotropic
log-scale) and shear, plus first-difference (smoothness) and second-difference
(curvature) penalties applied along a chosen ordering.

Units of the weights
--------------------
- ``translation_prior``  : per (Angstrom)^2 of residual translation.
- ``rotation_prior``     : per (radian)^2 of residual rotation.
- ``scale_prior``        : per (log-scale)^2 (dimensionless).
- ``shear_prior``        : per (shear)^2 (dimensionless off-diagonal).
- ``smoothness``         : per squared first difference of each component
                           between adjacent tilts in the ordering.
- ``curvature``          : per squared second difference.

The ordering (``tilt_angle`` or ``acquisition``) is recorded in the run
manifest because smoothness is not invariant to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import torch
from .parameter_scope import decompose


@dataclass(frozen=True)
class RegularizationConfig:
    translation_prior: float = 0.0
    rotation_prior: float = 0.01
    scale_prior: float = 0.1
    shear_prior: float = 0.1
    smoothness: float = 0.01
    curvature: float = 0.0
    ordering: str = "tilt_angle"  # or "acquisition"

    def validate(self) -> None:
        if self.ordering not in ("tilt_angle", "acquisition"):
            raise ValueError(f"unknown ordering {self.ordering!r}")
        for f in ("translation_prior", "rotation_prior", "scale_prior",
                  "shear_prior", "smoothness", "curvature"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be non-negative")


def _ordering_indices(config: RegularizationConfig, tilt_angles, n: int):
    if config.ordering == "tilt_angle":
        if tilt_angles is None:
            raise ValueError("tilt_angle ordering requires tilt_angles")
        return np.argsort(np.asarray(tilt_angles, dtype=float).reshape(-1))
    return np.arange(n)


def _diff_penalty(vec, order, smoothness, curvature):
    seq = vec[order]
    loss = vec.new_zeros(())
    if seq.shape[0] >= 2 and smoothness > 0:
        d1 = seq[1:] - seq[:-1]
        loss = loss + smoothness * (d1 ** 2).sum()
    if seq.shape[0] >= 3 and curvature > 0:
        d2 = seq[2:] - 2 * seq[1:-1] + seq[:-2]
        loss = loss + curvature * (d2 ** 2).sum()
    return loss


def regularization_loss(model_name: str, params, tilt_angles, config: RegularizationConfig):
    """Scalar torch loss combining priors and smoothness/curvature."""
    config.validate()
    comp = decompose(model_name, params)
    n = comp["tx"].shape[0]
    order = _ordering_indices(config, tilt_angles, n)

    loss = comp["tx"].new_zeros(())

    # Priors (L2 on the residual away from identity).
    loss = loss + config.translation_prior * (comp["tx"] ** 2 + comp["ty"] ** 2).sum()
    if "rotation" in comp:
        loss = loss + config.rotation_prior * (comp["rotation"] ** 2).sum()
    if "isotropic_log_scale" in comp:
        loss = loss + config.scale_prior * (comp["isotropic_log_scale"] ** 2).sum()
    if "anisotropic_log_scale" in comp:
        loss = loss + config.scale_prior * (comp["anisotropic_log_scale"] ** 2).sum()
    if "shear" in comp:
        loss = loss + config.shear_prior * (comp["shear"] ** 2).sum()

    # Smoothness / curvature on every component along the ordering.
    for vec in comp.values():
        loss = loss + _diff_penalty(vec, order, config.smoothness, config.curvature)
    return loss
