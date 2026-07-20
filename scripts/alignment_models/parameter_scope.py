#!/usr/bin/env python3
"""Parameter scopes: controlled sharing/smoothing of residual parameters.

Models are decomposed into named scalar components per tilt:

- ``tx``, ``ty``                  (translation, Angstrom)
- ``rotation``                    (phi, radians; rigid+)
- ``isotropic_log_scale``         (similarity: log_scale; affine: (alpha+beta)/2)
- ``anisotropic_log_scale``       (affine: (alpha-beta)/2)
- ``shear``                       (affine)

A *scope* constrains how a component varies across tilts:

- ``per_tilt``        : free per tilt (N DOF).
- ``per_tilt_smooth`` : free per tilt, but smoothness is penalised in the loss
                        (projection is identity here; see ``regularization``).
- ``global``          : one shared value for all tilts (1 DOF) -> broadcast mean.
- ``fixed``           : forced to the identity value 0 (0 DOF).
- ``spline``          : K linear-interpolation control points over the ordering.

``global``/``fixed``/``spline`` reduce the degrees of freedom and improve
identifiability of weak components (anisotropic scale, shear). Adding more
flexible scopes requires identifiability + export tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import torch
from .registry import model_rank

ALLOWED_SCOPES = ("fixed", "global", "per_tilt", "per_tilt_smooth", "spline")


@dataclass(frozen=True)
class ScopeConfig:
    translation: str = "per_tilt"
    rotation: str = "per_tilt_smooth"
    isotropic_scale: str = "global"
    anisotropic_scale: str = "global"
    shear: str = "global"
    spline_control_points: int = 5

    def validate(self) -> None:
        for field in ("translation", "rotation", "isotropic_scale",
                      "anisotropic_scale", "shear"):
            value = getattr(self, field)
            if value not in ALLOWED_SCOPES:
                raise ValueError(
                    f"scope[{field}] = {value!r} not in {ALLOWED_SCOPES}"
                )
        if self.spline_control_points < 2:
            raise ValueError("spline_control_points must be >= 2")


def decompose(model_name: str, params) -> dict:
    p = torch.as_tensor(params)
    comp = {"tx": p[:, 0], "ty": p[:, 1]}
    rank = model_rank(model_name)
    if rank >= model_rank("rigid"):
        comp["rotation"] = p[:, 2]
    if model_name == "similarity":
        comp["isotropic_log_scale"] = p[:, 3]
    if model_name == "affine":
        comp["isotropic_log_scale"] = 0.5 * (p[:, 3] + p[:, 4])
        comp["anisotropic_log_scale"] = 0.5 * (p[:, 3] - p[:, 4])
        comp["shear"] = p[:, 5]
    return comp


def recompose(model_name: str, comp: dict):
    cols = [comp["tx"], comp["ty"]]
    rank = model_rank(model_name)
    if rank >= model_rank("rigid"):
        cols.append(comp["rotation"])
    if model_name == "similarity":
        cols.append(comp["isotropic_log_scale"])
    if model_name == "affine":
        iso = comp["isotropic_log_scale"]
        aniso = comp["anisotropic_log_scale"]
        alpha = iso + aniso
        beta = iso - aniso
        cols.extend([alpha, beta, comp["shear"]])
    return torch.stack(cols, dim=1)


def _project_global(vec):
    # Keep the mean IN the autograd graph (do not convert to a Python float):
    # the shared global value must receive gradient during refinement.
    return vec.mean().expand_as(vec)


def _project_fixed(vec):
    return torch.zeros_like(vec)


def _linear_interp_weights(src_positions, query_positions, n):
    """Constant (n_query, n_src-on-grid) matrix for piecewise-linear interp.

    ``src_positions`` are the float grid positions of the control samples and
    ``query_positions`` the float positions to evaluate. Returns a matrix that,
    multiplied by the source vector, gives the interpolated values."""
    src = np.asarray(src_positions, float)
    q = np.asarray(query_positions, float)
    W = np.zeros((len(q), len(src)), dtype=float)
    for i, x in enumerate(q):
        j = np.searchsorted(src, x)
        if j <= 0:
            W[i, 0] = 1.0
        elif j >= len(src):
            W[i, -1] = 1.0
        else:
            x0, x1 = src[j - 1], src[j]
            frac = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            W[i, j - 1] = 1.0 - frac
            W[i, j] = frac
    return W


def _project_spline(vec, order, k):
    """Differentiable projection onto a K-knot piecewise-linear subspace.

    The whole operation is linear in ``vec`` with a CONSTANT matrix, so it stays
    on the autograd graph (no numpy round-trip on the optimization path)."""
    n = vec.shape[0]
    k = min(k, n)
    if k >= n:
        return vec
    xs = np.arange(n, dtype=float)
    ctrl_idx = np.linspace(0, n - 1, k)
    # sample control values from the (ordered) vector at control indices ...
    S = _linear_interp_weights(xs, ctrl_idx, n)          # (k, n)
    # ... and interpolate back to every point.
    B = _linear_interp_weights(ctrl_idx, xs, k)          # (n, k)
    P = torch.as_tensor(B @ S, dtype=vec.dtype)          # (n, n) constant
    ordered = vec[order]
    projected_ordered = P @ ordered
    inv = np.argsort(order)
    return projected_ordered[inv]


# Which scope key governs each component.
_COMPONENT_SCOPE_KEY = {
    "tx": "translation",
    "ty": "translation",
    "rotation": "rotation",
    "isotropic_log_scale": "isotropic_scale",
    "anisotropic_log_scale": "anisotropic_scale",
    "shear": "shear",
}


def apply_scopes(model_name: str, params, config: ScopeConfig, tilt_angles=None):
    """Project parameters onto the configured scopes (new tensor)."""
    config.validate()
    comp = decompose(model_name, params)
    n = next(iter(comp.values())).shape[0]
    if tilt_angles is not None:
        order = np.argsort(np.asarray(tilt_angles, dtype=float).reshape(-1))
    else:
        order = np.arange(n)
    out = {}
    for name, vec in comp.items():
        scope = getattr(config, _COMPONENT_SCOPE_KEY[name])
        if scope in ("per_tilt", "per_tilt_smooth"):
            out[name] = vec
        elif scope == "global":
            out[name] = _project_global(vec)
        elif scope == "fixed":
            out[name] = _project_fixed(vec)
        elif scope == "spline":
            out[name] = _project_spline(vec, order, config.spline_control_points)
        else:  # pragma: no cover - validated above
            raise ValueError(scope)
    return recompose(model_name, out)


def degrees_of_freedom(model_name: str, n_tilts: int, config: ScopeConfig) -> int:
    """Total free DOF given the scopes (for identifiability reporting)."""
    config.validate()
    comp_keys = list(decompose(model_name, torch.zeros((1, _n_params(model_name)))).keys())
    dof = 0
    for name in comp_keys:
        scope = getattr(config, _COMPONENT_SCOPE_KEY[name])
        if scope in ("per_tilt", "per_tilt_smooth"):
            dof += n_tilts
        elif scope == "global":
            dof += 1
        elif scope == "fixed":
            dof += 0
        elif scope == "spline":
            dof += min(config.spline_control_points, n_tilts)
    return dof


def _n_params(model_name: str) -> int:
    from .registry import get_model
    return get_model(model_name).n_params
