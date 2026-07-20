#!/usr/bin/env python3
"""Canonical loader/validator for the ``[refinement]`` configuration block.

This is the single place that turns user TOML/CLI into a fully-resolved
refinement specification: the maximum model, parameter scopes, gauge
constraints, regularization, and the staged schedule. Other scripts must not
re-parse refinement settings.

TOML shape (see ``config/project_settings.example.toml``)::

    [refinement]
    model = "translation"            # translation|rigid|similarity|affine
    schedule = "automatic"           # automatic|explicit
    coordinate_frame = "aligned_physical"

    [refinement.parameter_scope]
    translation = "per_tilt"
    rotation = "per_tilt_smooth"
    isotropic_scale = "global"
    anisotropic_scale = "global"
    shear = "global"

    [refinement.constraints]
    anchor_tilt = "closest_to_zero"
    zero_mean_rotation = true
    zero_mean_log_scale = true
    zero_mean_shear = true

    [refinement.regularization]
    translation_prior = 0.0
    rotation_prior = 0.01
    scale_prior = 0.1
    shear_prior = 0.1
    smoothness = 0.01
    curvature = 0.0
    ordering = "tilt_angle"

    [[refinement.stages]]   # only when schedule = "explicit"
    model = "translation"
    downsample = 2
    max_epochs = 5
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constraints import GaugeConfig
from .parameter_scope import ScopeConfig
from .regularization import RegularizationConfig
from .registry import NESTING_ORDER, model_rank

VALID_MODELS = NESTING_ORDER
VALID_FRAMES = ("aligned_physical", "raw_physical")

# Automatic stage schedule per maximum model: warm up from coarse to the target.
AUTOMATIC_SCHEDULE = {
    "translation": ["anchoring", "translation", "translation"],
    "rigid": ["anchoring", "translation", "rigid"],
    "similarity": ["anchoring", "translation", "rigid", "similarity"],
    "affine": ["anchoring", "translation", "rigid", "similarity", "affine"],
}


@dataclass
class Stage:
    model: str
    downsample: int = 1
    max_epochs: int = 5

    def validate(self, max_model: str) -> None:
        if self.model not in VALID_MODELS and self.model != "anchoring":
            raise ValueError(f"stage model {self.model!r} invalid")
        if self.model != "anchoring" and model_rank(self.model) > model_rank(max_model):
            raise ValueError(
                f"stage model {self.model!r} exceeds the maximum refinement model "
                f"{max_model!r}; a stage may not be more general than [refinement].model"
            )
        if self.downsample < 1:
            raise ValueError("downsample must be >= 1")
        if self.max_epochs < 0:
            raise ValueError("max_epochs must be >= 0")


@dataclass
class RefinementConfig:
    model: str = "translation"
    schedule: str = "automatic"  # automatic | explicit
    coordinate_frame: str = "aligned_physical"
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    stages: list[Stage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- validation / resolution -------------------------------------------
    def validate(self) -> None:
        if self.model not in VALID_MODELS:
            raise ValueError(
                f"[refinement].model = {self.model!r} not in {list(VALID_MODELS)}"
            )
        if self.schedule not in ("automatic", "explicit"):
            raise ValueError(f"[refinement].schedule = {self.schedule!r} invalid")
        if self.coordinate_frame not in VALID_FRAMES:
            raise ValueError(
                f"[refinement].coordinate_frame = {self.coordinate_frame!r} not in {VALID_FRAMES}"
            )
        self.scope.validate()
        self.regularization.validate()
        self._validate_scope_model_compat()
        if self.schedule == "explicit":
            if not self.stages:
                raise ValueError(
                    "[refinement].schedule = 'explicit' requires at least one [[refinement.stages]]"
                )
            for st in self.stages:
                st.validate(self.model)
        elif self.stages:
            self.warnings.append(
                "[[refinement.stages]] are ignored because schedule = 'automatic'."
            )

    def _validate_scope_model_compat(self) -> None:
        """Warn when a scope is set for a component the model does not have."""
        rank = model_rank(self.model)
        if rank < model_rank("rigid") and self.scope.rotation != "per_tilt":
            self.warnings.append(
                f"rotation scope {self.scope.rotation!r} ignored: model {self.model!r} has no rotation."
            )
        if self.model not in ("similarity", "affine") and self.scope.isotropic_scale != "per_tilt":
            self.warnings.append(
                f"isotropic_scale scope ignored: model {self.model!r} has no scale."
            )
        if self.model != "affine":
            for comp in ("anisotropic_scale", "shear"):
                if getattr(self.scope, comp) != "per_tilt":
                    self.warnings.append(
                        f"{comp} scope ignored: only the affine model has {comp}."
                    )

    def resolved_stages(self) -> list[Stage]:
        if self.schedule == "explicit":
            return list(self.stages)
        # automatic: map names to Stage objects (anchoring is a translation-only
        # warm-up that also applies the gauge anchor).
        out = []
        for name in AUTOMATIC_SCHEDULE[self.model]:
            model = "translation" if name == "anchoring" else name
            out.append(Stage(model=model, downsample=2 if name == "anchoring" else 1, max_epochs=5))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model": self.model,
            "schedule": self.schedule,
            "coordinate_frame": self.coordinate_frame,
            "parameter_scope": asdict(self.scope),
            "constraints": asdict(self.gauge),
            "regularization": asdict(self.regularization),
            "stages": [asdict(s) for s in self.resolved_stages()],
            "warnings": self.warnings,
        }
        return d


def _scope_from(d: dict) -> ScopeConfig:
    return ScopeConfig(
        translation=d.get("translation", "per_tilt"),
        rotation=d.get("rotation", "per_tilt_smooth"),
        isotropic_scale=d.get("isotropic_scale", "global"),
        anisotropic_scale=d.get("anisotropic_scale", "global"),
        shear=d.get("shear", "global"),
        spline_control_points=int(d.get("spline_control_points", 5)),
    )


def _gauge_from(d: dict) -> GaugeConfig:
    return GaugeConfig(
        anchor_tilt=str(d.get("anchor_tilt", "closest_to_zero")),
        zero_mean_rotation=bool(d.get("zero_mean_rotation", True)),
        zero_mean_log_scale=bool(d.get("zero_mean_log_scale", True)),
        zero_mean_shear=bool(d.get("zero_mean_shear", True)),
    )


def _reg_from(d: dict) -> RegularizationConfig:
    return RegularizationConfig(
        translation_prior=float(d.get("translation_prior", 0.0)),
        rotation_prior=float(d.get("rotation_prior", 0.01)),
        scale_prior=float(d.get("scale_prior", 0.1)),
        shear_prior=float(d.get("shear_prior", 0.1)),
        smoothness=float(d.get("smoothness", 0.01)),
        curvature=float(d.get("curvature", 0.0)),
        ordering=str(d.get("ordering", "tilt_angle")),
    )


def from_toml_dict(refinement: dict[str, Any], cli_overrides: dict[str, Any] | None = None) -> RefinementConfig:
    """Build and validate a ``RefinementConfig`` from a parsed ``[refinement]`` dict."""
    refinement = dict(refinement or {})
    cli_overrides = cli_overrides or {}

    model = cli_overrides.get("model") or refinement.get("model", "translation")
    schedule = cli_overrides.get("schedule") or refinement.get("schedule", "automatic")
    frame = refinement.get("coordinate_frame", "aligned_physical")

    scope = _scope_from(refinement.get("parameter_scope", {}))
    # CLI scope overrides
    if cli_overrides.get("rotation_scope"):
        scope = ScopeConfig(**{**asdict(scope), "rotation": cli_overrides["rotation_scope"]})
    if cli_overrides.get("scale_scope"):
        scope = ScopeConfig(**{**asdict(scope), "isotropic_scale": cli_overrides["scale_scope"]})
    if cli_overrides.get("shear_scope"):
        scope = ScopeConfig(**{**asdict(scope), "shear": cli_overrides["shear_scope"]})

    gauge = _gauge_from(refinement.get("constraints", {}))
    reg = _reg_from(refinement.get("regularization", {}))
    stages = [
        Stage(model=str(s.get("model")), downsample=int(s.get("downsample", 1)),
              max_epochs=int(s.get("max_epochs", 5)))
        for s in refinement.get("stages", [])
    ]

    cfg = RefinementConfig(
        model=model, schedule=schedule, coordinate_frame=frame,
        scope=scope, gauge=gauge, regularization=reg, stages=stages,
    )
    cfg.validate()
    return cfg
