#!/usr/bin/env python3
"""Affine residual model. DOF = 6 (tx, ty, phi, alpha, beta, shear).

Parameterization (positive-determinant, invertible, no accidental reflection)::

    A = R(phi) @ U,   U = [[exp(alpha), shear], [0, exp(beta)]]

``det(A) = det(U) = exp(alpha + beta) > 0`` for all finite parameters, so the
matrix is always invertible and orientation-preserving. The upper-triangular
``U`` with positive diagonal is the unique factor that makes the parameter
space nest the similarity model (``alpha = beta = log_scale, shear = 0``).
"""
from __future__ import annotations

from .base import AlignmentModel, rotation_matrix, torch


class AffineModel(AlignmentModel):
    name = "affine"
    param_names = ("tx", "ty", "phi", "alpha", "beta", "shear")

    def linear_matrices(self, params):
        p = self.as_tensor(params)
        n = p.shape[0]
        ea = torch.exp(p[:, 3])
        eb = torch.exp(p[:, 4])
        shear = p[:, 5]
        zero = torch.zeros_like(ea)
        row0 = torch.stack([ea, shear], dim=-1)
        row1 = torch.stack([zero, eb], dim=-1)
        U = torch.stack([row0, row1], dim=-2)
        R = rotation_matrix(p[:, 2])
        return torch.matmul(R, U)
