#!/usr/bin/env python3
"""Alignment-model package with lazy imports.

Importing the package itself is intentionally lightweight. PyTorch-dependent
models and optimisers are loaded only when their public symbols are requested.
"""
from __future__ import annotations

from importlib import import_module

_MODULES = {
    "composition",
    "constraints",
    "coordinate_frames",
    "parameter_scope",
    "regularization",
    "serialization",
    "result_contract",
    "interop",
}

_SYMBOLS = {
    "AlignmentModel": ("base", "AlignmentModel"),
    "TranslationModel": ("translation", "TranslationModel"),
    "RigidModel": ("rigid", "RigidModel"),
    "SimilarityModel": ("similarity", "SimilarityModel"),
    "AffineModel": ("affine", "AffineModel"),
    "MODEL_CLASSES": ("registry", "MODEL_CLASSES"),
    "NESTING_ORDER": ("registry", "NESTING_ORDER"),
    "get_model": ("registry", "get_model"),
    "model_rank": ("registry", "model_rank"),
    "embed_params": ("registry", "embed_params"),
    "RefineResult": ("refine", "RefineResult"),
    "refine": ("refine", "refine"),
    "WARM_START_PRESETS": ("materialize", "WARM_START_PRESETS"),
    "analytic_field_numpy": ("materialize", "analytic_field_numpy"),
    "detector_grid_points": ("materialize", "detector_grid_points"),
    "materialize_field": ("materialize", "materialize_field"),
    "materialize_model_field": ("materialize", "materialize_model_field"),
    "ConstrainedResult": ("optimize_constrained_2d", "ConstrainedResult"),
    "OptimizerSettings": ("optimize_constrained_2d", "OptimizerSettings"),
    "ReconstructionSettings": ("optimize_constrained_2d", "ReconstructionSettings"),
    "SafetyBounds": ("optimize_constrained_2d", "SafetyBounds"),
    "grid_sample_image_scorer": ("optimize_constrained_2d", "grid_sample_image_scorer"),
    "optimize_constrained_2d": ("optimize_constrained_2d", "optimize_constrained_2d"),
}

__all__ = sorted(_MODULES | set(_SYMBOLS))


def __getattr__(name: str):
    if name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    elif name in _SYMBOLS:
        module_name, attribute = _SYMBOLS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
