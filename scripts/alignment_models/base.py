#!/usr/bin/env python3
"""Shared API for constrained residual alignment models.

A model maps a per-tilt parameter tensor ``params`` of shape
``(n_tilts, n_params)`` to a per-tilt linear matrix ``A_i`` (2x2) and
translation ``t_i`` (2, in Angstrom). All transform algebra (homogeneous
matrices, centred application, composition, inversion) lives here and in
``coordinate_frames`` so it is never duplicated per model.

Conventions
-----------
- Every model's ``params[:, 0]`` and ``params[:, 1]`` are ``tx, ty`` in
  Angstrom. Subclasses define the remaining columns and ``linear_matrices``.
- ``A_i`` acts around the **physical centre** of the aligned image:
  ``q = c + A_i (p - c) + t_i`` (see ``coordinate_frames``).
- The parameter ordering is fixed so that ``translation`` parameters are a
  prefix of ``rigid`` of ``similarity`` of ``affine`` (nestedness).
- Torch is used for differentiability (gradient-based refinement and autograd
  vs finite-difference checks). Default dtype is float64 for numerical tests.
"""
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch is required for refinement
    torch = None  # type: ignore
    _HAVE_TORCH = False


def require_torch() -> None:
    if not _HAVE_TORCH:
        raise RuntimeError(
            "PyTorch is required for the constrained alignment models. "
            "Install torch in the MissAlignment environment."
        )


def rotation_matrix(phi):
    """2x2 rotation R(phi) = [[cos, -sin], [sin, cos]] (torch, differentiable)."""
    c = torch.cos(phi)
    s = torch.sin(phi)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class AlignmentModel(abc.ABC):
    """Abstract constrained residual model.

    Subclasses set ``name``, ``param_names`` and implement ``linear_matrices``.
    """

    name: str = "base"
    param_names: tuple[str, ...] = ("tx", "ty")

    def __init__(self, dtype=None):
        require_torch()
        self.dtype = dtype or torch.float64

    # -- introspection ------------------------------------------------------
    @property
    def n_params(self) -> int:
        return len(self.param_names)

    # -- parameter construction --------------------------------------------
    def identity_params(self, n_tilts: int):
        """Zero parameters -> identity transform (A=I, t=0) for every tilt.

        This relies on each model's parameterization being chosen so that the
        all-zero vector is the identity (rotation 0, log-scale 0 -> scale 1,
        shear 0). Verified by ``test_models`` for all four models.
        """
        return torch.zeros((int(n_tilts), self.n_params), dtype=self.dtype)

    def as_tensor(self, params) -> "torch.Tensor":
        t = torch.as_tensor(params, dtype=self.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim != 2 or t.shape[1] != self.n_params:
            raise ValueError(
                f"{self.name}: expected params of shape (n_tilts, {self.n_params}), "
                f"got {tuple(t.shape)}"
            )
        return t

    # -- core maps ----------------------------------------------------------
    @abc.abstractmethod
    def linear_matrices(self, params) -> "torch.Tensor":
        """Return the per-tilt 2x2 linear part ``A_i`` (differentiable)."""

    def translations(self, params) -> "torch.Tensor":
        """Per-tilt translation ``t_i`` in Angstrom (first two columns)."""
        p = self.as_tensor(params)
        return p[:, :2]

    def matrices_and_translations(self, params):
        p = self.as_tensor(params)
        return self.linear_matrices(p), self.translations(p)

    def homogeneous_physical(self, params, centers_phys):
        """Per-tilt absolute-physical homogeneous 3x3 (differentiable).

        ``centers_phys`` is ``(n_tilts, 2)`` physical centres in Angstrom.
        ``q = A (p - c) + t + c`` -> absolute ``q = A p + (t + (I - A) c)``.
        """
        A, t = self.matrices_and_translations(params)
        c = torch.as_tensor(centers_phys, dtype=self.dtype)
        if c.ndim == 1:
            c = c.unsqueeze(0).expand(A.shape[0], 2)
        n = A.shape[0]
        eye = torch.eye(2, dtype=self.dtype).expand(n, 2, 2)
        abs_t = t + torch.einsum("nij,nj->ni", eye - A, c)
        H = torch.zeros((n, 3, 3), dtype=self.dtype)
        H[:, :2, :2] = A
        H[:, :2, 2] = abs_t
        H[:, 2, 2] = 1.0
        return H

    def apply_centered(self, params, points_xy, centers_phys):
        """Apply the model to ``(n_tilts, N, 2)`` or ``(N, 2)`` physical points."""
        A, t = self.matrices_and_translations(params)
        c = torch.as_tensor(centers_phys, dtype=self.dtype)
        pts = torch.as_tensor(points_xy, dtype=self.dtype)
        if pts.ndim == 2:
            pts = pts.unsqueeze(0).expand(A.shape[0], *pts.shape)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        centered = pts - c.unsqueeze(1)
        out = torch.einsum("nij,nkj->nki", A, centered) + c.unsqueeze(1) + t.unsqueeze(1)
        return out

    # -- numpy export -------------------------------------------------------
    def matrices_numpy(self, params) -> np.ndarray:
        A, _ = self.matrices_and_translations(params)
        return A.detach().cpu().numpy()

    def translations_numpy(self, params) -> np.ndarray:
        _, t = self.matrices_and_translations(params)
        return t.detach().cpu().numpy()

    def determinants(self, params) -> np.ndarray:
        return np.linalg.det(self.matrices_numpy(params))
