#!/usr/bin/env python3
"""Differentiable constrained -> detector movement-field materialization (Option B).

This is the exact artifact the MissAlignment fork's differentiable forward pass
consumes when ``alignment: rigid``/``similarity`` (or ``translation``) is active:
a per-detector-pixel 2-D displacement field

    d(p) = (A - I) @ (p - c) + t

where ``A``/``t`` are produced by a *constrained* :class:`AlignmentModel`
(translation/rigid/similarity/affine) and stay attached to the model parameters,
so ``d`` is differentiable w.r.t. those parameters and gradients flow back through
the materialized field into the constrained DOF (no detach, no numpy round-trip,
no non-differentiable interpolation). The optimized variables remain the
constrained-model parameters -- the field is a *function* of them, never a free
grid that is constrained only at export.

``d(p)`` is the displacement (target minus source); the absolute mapped point is
``q(p) = p + d(p) = A (p - c) + c + t`` (identical to
``AlignmentModel.apply_centered``), which is what the equivalence test asserts.

Everything here is device/dtype-aware: pass ``device``/``dtype`` (or tensors
already on a device) and the grid, centres and identity are built there.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import AlignmentModel, require_torch, torch


def detector_grid_points(shape_xy: Sequence[int], *,
                         pixel_size_xy: Sequence[float] = (1.0, 1.0),
                         origin_xy: Sequence[float] = (0.0, 0.0),
                         device=None, dtype=None) -> "torch.Tensor":
    """Build the ``(H*W, 2)`` physical (x, y) coordinates of every detector pixel.

    Pixel ``(ix, iy)`` maps to physical ``origin + pixel_size * (ix, iy)``. The
    ordering is row-major in ``(iy, ix)`` so a returned field can be reshaped to
    ``(H, W, 2)``.
    """
    require_torch()
    dtype = dtype or torch.float64
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    ix = torch.arange(nx, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(iy, ix, indexing="ij")  # (ny, nx)
    px = origin_xy[0] + pixel_size_xy[0] * gx
    py = origin_xy[1] + pixel_size_xy[1] * gy
    return torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)  # (H*W, 2)


def materialize_field(A: "torch.Tensor", t: "torch.Tensor",
                      points_xy: "torch.Tensor", center_xy: "torch.Tensor") -> "torch.Tensor":
    """Option-B field ``d(p) = (A - I) @ (p - c) + t``, fully differentiable.

    Shapes
    ------
    - ``A``: ``(n_tilts, 2, 2)`` (or ``(2, 2)``)
    - ``t``: ``(n_tilts, 2)`` (or ``(2,)``)
    - ``points_xy``: ``(N, 2)`` shared grid, or ``(n_tilts, N, 2)`` per-tilt
    - ``center_xy``: ``(2,)`` or ``(n_tilts, 2)``

    Returns ``(n_tilts, N, 2)`` (or ``(N, 2)`` when ``A`` is a single matrix).
    ``A``/``t`` are used as-is so their grad (from the constrained params) is kept.
    """
    require_torch()
    single = A.ndim == 2
    if single:
        A = A.unsqueeze(0)
        t = t.reshape(1, 2)
    n = A.shape[0]
    pts = points_xy
    if pts.ndim == 2:
        pts = pts.unsqueeze(0).expand(n, *pts.shape)  # (n, N, 2)
    c = center_xy
    if c.ndim == 1:
        c = c.unsqueeze(0).expand(n, 2)
    eye = torch.eye(2, dtype=A.dtype, device=A.device).expand(n, 2, 2)
    A_minus_I = A - eye
    centered = pts - c.unsqueeze(1)                                   # (n, N, 2)
    d = torch.einsum("nij,nkj->nki", A_minus_I, centered) + t.unsqueeze(1)
    return d.squeeze(0) if single else d


def materialize_model_field(model: AlignmentModel, params,
                            shape_xy: Sequence[int], *,
                            pixel_size_xy: Sequence[float] = (1.0, 1.0),
                            origin_xy: Sequence[float] = (0.0, 0.0),
                            centers_phys=None, device=None, dtype=None,
                            as_image: bool = False) -> "torch.Tensor":
    """Materialize the detector movement field of a constrained model.

    ``params`` are the constrained-model parameters (the *optimized* variables);
    set ``params.requires_grad_(True)`` upstream and the returned field carries
    their gradient. ``centers_phys`` defaults to the geometric image centre
    ``(n-1)/2`` per axis (IMOD convention) in physical units.

    With ``as_image=True`` the field is returned as ``(n_tilts, ny, nx, 2)``.
    """
    require_torch()
    dtype = dtype or model.dtype
    # Compute the per-tilt matrices in the model's native dtype on the params'
    # current device (the constrained models build small CPU/float64 tensors
    # internally), then move the field to the requested device/dtype. ``.to`` is
    # differentiable, so gradients still flow back to ``params``.
    A, t = model.matrices_and_translations(params)
    if device is not None or dtype != A.dtype:
        A = A.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
    n = A.shape[0]
    pts = detector_grid_points(shape_xy, pixel_size_xy=pixel_size_xy,
                               origin_xy=origin_xy, device=device, dtype=dtype)
    if centers_phys is None:
        cx = origin_xy[0] + pixel_size_xy[0] * (int(shape_xy[0]) - 1) / 2.0
        cy = origin_xy[1] + pixel_size_xy[1] * (int(shape_xy[1]) - 1) / 2.0
        c = torch.tensor([cx, cy], dtype=dtype, device=device).expand(n, 2)
    else:
        c = torch.as_tensor(centers_phys, dtype=dtype, device=device)
        if c.ndim == 1:
            c = c.unsqueeze(0).expand(n, 2)
    field = materialize_field(A, t, pts, c)               # (n, N, 2)
    if as_image:
        nx, ny = int(shape_xy[0]), int(shape_xy[1])
        field = field.reshape(n, ny, nx, 2)
    return field


def analytic_field_numpy(A: np.ndarray, t: np.ndarray,
                         points_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Independent numpy reference ``d = (A - I)(p - c) + t`` for validation.

    ``A``: ``(2, 2)``, ``t``: ``(2,)``, ``points_xy``: ``(N, 2)``, ``center``: ``(2,)``.
    Computed without torch so it is a genuinely separate oracle.
    """
    A = np.asarray(A, dtype=np.float64).reshape(2, 2)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    return (pts - c) @ (A - np.eye(2)).T + t


def WARM_START_PRESETS() -> dict:
    """Per-mode warm-start parameter presets (identity residual) and the staged
    schedule the MissAlignment fork uses. translation warm-starts rigid;
    rigid warm-starts similarity (nested DOF), matching ``03_run_missalignment``.
    """
    return {
        "translation": {"identity": (0.0, 0.0), "warm_start_from": None,
                        "iteration_settings": ["global", "global"]},
        "rigid": {"identity": (0.0, 0.0, 0.0), "warm_start_from": "translation",
                  "iteration_settings": ["global", "rigid", "rigid"]},
        "similarity": {"identity": (0.0, 0.0, 0.0, 0.0), "warm_start_from": "rigid",
                       "iteration_settings": ["global", "rigid", "rigid", "similarity"]},
    }
