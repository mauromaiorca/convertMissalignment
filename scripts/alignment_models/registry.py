#!/usr/bin/env python3
"""Model registry and the nestedness embedding maps.

The parameter spaces are nested ``translation ⊂ rigid ⊂ similarity ⊂ affine``.
``embed_params`` lifts a simpler model's parameters into a larger model so that
both produce the *identical* transform; this is what the nestedness tests use.
"""
from __future__ import annotations

from .affine import AffineModel
from .base import AlignmentModel, torch
from .rigid import RigidModel
from .similarity import SimilarityModel
from .translation import TranslationModel

MODEL_CLASSES = {
    "translation": TranslationModel,
    "rigid": RigidModel,
    "similarity": SimilarityModel,
    "affine": AffineModel,
}

# Strict nesting order (simplest -> most general).
NESTING_ORDER = ("translation", "rigid", "similarity", "affine")


def get_model(name: str, dtype=None) -> AlignmentModel:
    if name not in MODEL_CLASSES:
        raise ValueError(
            f"unknown refinement model {name!r}; choose from {sorted(MODEL_CLASSES)}"
        )
    return MODEL_CLASSES[name](dtype=dtype)


def model_rank(name: str) -> int:
    if name not in NESTING_ORDER:
        raise ValueError(f"{name!r} is not in the nesting order")
    return NESTING_ORDER.index(name)


def _embed_step(params, from_name: str, to_name: str):
    """Embed one step up the nesting chain (params identical transform)."""
    p = torch.as_tensor(params)
    if from_name == "translation" and to_name == "rigid":
        # append phi = 0
        zeros = torch.zeros((p.shape[0], 1), dtype=p.dtype)
        return torch.cat([p, zeros], dim=1)
    if from_name == "rigid" and to_name == "similarity":
        # append log_scale = 0
        zeros = torch.zeros((p.shape[0], 1), dtype=p.dtype)
        return torch.cat([p, zeros], dim=1)
    if from_name == "similarity" and to_name == "affine":
        # [tx, ty, phi, ls] -> [tx, ty, phi, alpha=ls, beta=ls, shear=0]
        tx, ty, phi, ls = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        zeros = torch.zeros_like(tx)
        return torch.stack([tx, ty, phi, ls, ls, zeros], dim=1)
    raise ValueError(f"no single embedding step {from_name!r} -> {to_name!r}")


def embed_params(params, from_name: str, to_name: str):
    """Lift ``params`` from ``from_name`` to a higher model ``to_name``.

    The returned parameters produce a transform identical to the original.
    """
    if from_name == to_name:
        return torch.as_tensor(params)
    if model_rank(from_name) > model_rank(to_name):
        raise ValueError(
            f"cannot embed {from_name!r} into the smaller model {to_name!r}"
        )
    out = torch.as_tensor(params)
    i = model_rank(from_name)
    j = model_rank(to_name)
    for k in range(i, j):
        out = _embed_step(out, NESTING_ORDER[k], NESTING_ORDER[k + 1])
    return out
