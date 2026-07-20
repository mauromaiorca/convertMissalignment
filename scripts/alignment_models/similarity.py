#!/usr/bin/env python3
"""Similarity residual model. DOF = 4 (tx, ty, phi, log_scale).

``A = exp(log_scale) * R(phi)`` -> positive isotropic scale, no shear,
``A^T A = scale^2 I``.
"""
from __future__ import annotations

from .base import AlignmentModel, rotation_matrix, torch


class SimilarityModel(AlignmentModel):
    name = "similarity"
    param_names = ("tx", "ty", "phi", "log_scale")

    def linear_matrices(self, params):
        p = self.as_tensor(params)
        scale = torch.exp(p[:, 3])
        return scale[:, None, None] * rotation_matrix(p[:, 2])
