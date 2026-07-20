#!/usr/bin/env python3
"""Rigid residual model. DOF = 3 (tx, ty, phi). A = R(phi), det = +1, A^T A = I."""
from __future__ import annotations

from .base import AlignmentModel, rotation_matrix


class RigidModel(AlignmentModel):
    name = "rigid"
    param_names = ("tx", "ty", "phi")

    def linear_matrices(self, params):
        p = self.as_tensor(params)
        return rotation_matrix(p[:, 2])
