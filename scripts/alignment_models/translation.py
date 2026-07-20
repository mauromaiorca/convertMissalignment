#!/usr/bin/env python3
"""Translation-only residual model. DOF = 2 (tx, ty). A = I always."""
from __future__ import annotations

from .base import AlignmentModel, torch


class TranslationModel(AlignmentModel):
    name = "translation"
    param_names = ("tx", "ty")

    def linear_matrices(self, params):
        p = self.as_tensor(params)
        n = p.shape[0]
        # Identity, but tied to params so autograd is well-defined (zero grad).
        eye = torch.eye(2, dtype=self.dtype).expand(n, 2, 2).clone()
        # Add 0 * params so the graph connects (gradient is exactly zero).
        return eye + 0.0 * p[:, :1, None]
