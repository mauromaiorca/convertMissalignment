#!/usr/bin/env python3
"""One image-based constrained 2-D optimizer for translation/rigid/similarity.

Spec §19: a *single* optimizer, not one per model. The optimized variables are
the constrained-model parameters. The forward pass is:

    params (requires_grad)
      -> apply_scopes        (differentiable: global/per_tilt/spline)
      -> apply_gauge         (differentiable gauge fixing)
      -> materialize field   d(p) = (A - I)(p - c) + t   (Option B)
      -> reconstruct_and_score(field, ...)   <-- PLUGGABLE production hook
      -> image loss + regularization
      -> backward -> optimizer.step()

``reconstruct_and_score`` is where the *real* MissAlignment reconstruction /
projector / scoring network / image loss is injected on the cluster. It receives
the differentiable detector movement field and must return a scalar loss that
depends on it (so gradients flow back to the constrained parameters). Locally we
ship :func:`grid_sample_image_scorer`, a real differentiable bilinear-resampling
image-L2 scorer used by the tests — a genuine image path with real interpolation,
**not** the production projector (which is not installed here).

Nothing in the loop detaches the parameters before the loss; ``.detach()`` is
used only for logging/reporting (under ``no_grad``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .base import AlignmentModel, require_torch, torch
from .constraints import GaugeConfig, apply_gauge, gauge_report
from .materialize import materialize_model_field
from .parameter_scope import ScopeConfig, apply_scopes
from .regularization import RegularizationConfig, regularization_loss
from .registry import get_model


@dataclass
class OptimizerSettings:
    steps: int = 400
    lr: float = 0.02
    optimizer: str = "adam"          # "adam" | "sgd" | "lbfgs"
    weight_decay: float = 0.0
    log_every: int = 0               # 0 = only first/last


@dataclass
class ReconstructionSettings:
    shape_xy: tuple[int, int] = (64, 64)
    pixel_size_xy_A: tuple[float, float] = (1.0, 1.0)
    origin_xy: tuple[float, float] = (0.0, 0.0)
    centers_phys: object = None      # None -> geometric (n-1)/2 centre per tilt


@dataclass
class SafetyBounds:
    """Conservative DEFAULTS, not scientific truth (spec §22). Internal units:
    translation in pixels of the working grid, rotation in radians, scale as
    log_scale. Bound crossings are recorded; warnings precede a hard failure."""
    translation_warn_px: float = 100.0
    translation_hard_px: float = 500.0
    rotation_warn_rad: float = 5.0 * 3.14159265 / 180.0
    rotation_hard_rad: float = 20.0 * 3.14159265 / 180.0
    log_scale_warn: float = 0.05      # ~ +-5%
    log_scale_hard: float = 0.20      # ~ +-20%
    grace_steps: int = 10             # before declaring an expected-zero gradient dead

    def check(self, model_name: str, params, step: int) -> tuple[list, list]:
        """Return (warnings, hard_violations) for this parameter tensor."""
        import torch
        p = torch.as_tensor(params)
        warns, hard = [], []
        t = p[:, :2].abs().max().item()
        if t > self.translation_hard_px:
            hard.append(f"translation {t:.1f}px > hard {self.translation_hard_px}px")
        elif t > self.translation_warn_px:
            warns.append(f"translation {t:.1f}px > warn {self.translation_warn_px}px")
        if p.shape[1] >= 3:
            r = p[:, 2].abs().max().item()
            if r > self.rotation_hard_rad:
                hard.append(f"rotation {r:.3f}rad > hard {self.rotation_hard_rad:.3f}")
            elif r > self.rotation_warn_rad:
                warns.append(f"rotation {r:.3f}rad > warn {self.rotation_warn_rad:.3f}")
        if p.shape[1] >= 4:
            s = p[:, 3].abs().max().item()
            if s > self.log_scale_hard:
                hard.append(f"log_scale {s:.3f} > hard {self.log_scale_hard}")
            elif s > self.log_scale_warn:
                warns.append(f"log_scale {s:.3f} > warn {self.log_scale_warn}")
        return warns, hard


@dataclass
class ConstrainedResult:
    model: str
    params: object                   # (n_tilts, n_params) detached tensor
    loss_history: list
    image_loss_history: list
    reg_loss_history: list
    gauge: dict
    n_tilts: int
    scopes: dict
    settings: dict


# --------------------------------------------------------------------------- #
# Pluggable scorers
# --------------------------------------------------------------------------- #
def grid_sample_image_scorer(reference_image, observed_image, *, align_corners=True,
                             padding_mode="border"):
    """Build a real differentiable image-L2 scorer from a moving/target image pair.

    Returns ``score(field, **_)`` where ``field`` is ``(n_tilts, ny, nx, 2)`` (or
    ``(n, N, 2)`` reshaped to image) displacement in pixels. Each tilt warps
    ``reference_image`` by sampling at ``p + d(p)`` with torch ``grid_sample``
    (real bilinear interpolation) and accumulates the mean-squared error against
    ``observed_image[i]``. The score depends on ``field`` -> gradients reach the
    constrained parameters. This is a real image path, NOT the MissAlignment
    projector.
    """
    require_torch()
    F = torch.nn.functional
    ref = torch.as_tensor(reference_image)
    obs = torch.as_tensor(observed_image)
    if ref.ndim == 2:
        ref = ref.unsqueeze(0)
    if obs.ndim == 2:
        obs = obs.unsqueeze(0)

    def score(field, **_):
        if field.ndim == 3:  # (n, N, 2) -> (n, ny, nx, 2)
            n = field.shape[0]
            ny, nx = obs.shape[-2], obs.shape[-1]
            field = field.reshape(n, ny, nx, 2)
        n, ny, nx, _ = field.shape
        dev, dt = field.device, field.dtype
        iy, ix = torch.meshgrid(torch.arange(ny, device=dev, dtype=dt),
                                torch.arange(nx, device=dev, dtype=dt), indexing="ij")
        total = field.new_zeros(())
        r = ref.to(device=dev, dtype=dt)
        o = obs.to(device=dev, dtype=dt)
        rsrc = r if r.shape[0] == n else r.expand(n, ny, nx)
        otgt = o if o.shape[0] == n else o.expand(n, ny, nx)
        for i in range(n):
            qx = ix + field[i, :, :, 0]
            qy = iy + field[i, :, :, 1]
            gx = 2.0 * qx / (nx - 1) - 1.0
            gy = 2.0 * qy / (ny - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            warped = F.grid_sample(rsrc[i][None, None], grid, mode="bilinear",
                                   align_corners=align_corners, padding_mode=padding_mode)
            total = total + ((warped[0, 0] - otgt[i]) ** 2).mean()
        return total / n

    return score


# --------------------------------------------------------------------------- #
# The single optimizer
# --------------------------------------------------------------------------- #
def optimize_constrained_2d(
    *,
    alignment_model: AlignmentModel | str,
    initial_parameters,
    reconstruct_and_score: Callable,
    tilt_angles=None,
    parameter_scopes: Optional[ScopeConfig] = None,
    gauge: Optional[GaugeConfig] = None,
    regularization: Optional[RegularizationConfig] = None,
    optimizer_settings: Optional[OptimizerSettings] = None,
    reconstruction_settings: Optional[ReconstructionSettings] = None,
    safety_bounds: Optional["SafetyBounds"] = None,
    telemetry_dir=None,
    result_dir=None,
    stage_label: str = "stage0",
    seed: Optional[int] = None,
    device=None,
    dtype=None,
) -> ConstrainedResult:
    """Optimize constrained 2-D alignment parameters against a real image loss.

    ``reconstruct_and_score(field, *, model, params, tilt_angles, recon)`` must
    return a scalar loss differentiable in ``field``. The same call signature is
    used by the local ``grid_sample`` scorer and by the cluster MissAlignment
    forward pass (see ``IMAGE_BASED_CONSTRAINED_INTEGRATION.md``).
    """
    require_torch()
    model = get_model(alignment_model) if isinstance(alignment_model, str) else alignment_model
    name = model.name
    opt_s = optimizer_settings or OptimizerSettings()
    recon = reconstruction_settings or ReconstructionSettings()
    scopes = parameter_scopes or ScopeConfig()
    gcfg = gauge or GaugeConfig()
    reg = regularization or RegularizationConfig()
    bounds = safety_bounds or SafetyBounds()
    dtype = dtype or model.dtype
    if seed is not None:
        torch.manual_seed(int(seed))
    tele = _Telemetry(telemetry_dir, stage_label, name) if telemetry_dir else None
    bound_crossings: list = []

    p0 = torch.as_tensor(initial_parameters, dtype=model.dtype)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    n_tilts = p0.shape[0]
    angles = (np.asarray(tilt_angles, dtype=float).reshape(-1) if tilt_angles is not None
              else np.zeros(n_tilts))

    p = p0.clone().detach().requires_grad_(True)
    if opt_s.optimizer == "adam":
        optim = torch.optim.Adam([p], lr=opt_s.lr, weight_decay=opt_s.weight_decay)
    elif opt_s.optimizer == "sgd":
        optim = torch.optim.SGD([p], lr=opt_s.lr, weight_decay=opt_s.weight_decay)
    elif opt_s.optimizer == "lbfgs":
        optim = torch.optim.LBFGS([p], lr=opt_s.lr, max_iter=opt_s.steps)
    else:
        raise ValueError(f"unknown optimizer {opt_s.optimizer!r}")

    loss_hist, img_hist, reg_hist = [], [], []

    def forward():
        # scopes + gauge are differentiable functions of the free params p.
        q = apply_scopes(name, p, scopes, angles)
        q = apply_gauge(name, q, angles, gcfg)
        field = materialize_model_field(
            model, q, recon.shape_xy, pixel_size_xy=recon.pixel_size_xy_A,
            origin_xy=recon.origin_xy, centers_phys=recon.centers_phys,
            device=device, dtype=dtype, as_image=True)
        img = reconstruct_and_score(field, model=model, params=q,
                                    tilt_angles=angles, recon=recon)
        rloss = regularization_loss(name, q, angles, reg)
        if not torch.is_tensor(rloss):
            rloss = img.new_tensor(float(rloss))
        return img + rloss.to(img.dtype), img, rloss

    if opt_s.optimizer == "lbfgs":
        state = {}

        def closure():
            optim.zero_grad()
            total, img, rloss = forward()
            total.backward()
            state["t"], state["i"], state["r"] = total, img, rloss
            return total
        optim.step(closure)
        loss_hist.append(float(state["t"].detach()))
        img_hist.append(float(state["i"].detach()))
        reg_hist.append(float(state["r"].detach()) if torch.is_tensor(state["r"]) else float(state["r"]))
    else:
        for step in range(opt_s.steps):
            optim.zero_grad()
            total, img, rloss = forward()
            if not torch.isfinite(total):
                _fail(telemetry_dir, name, step, "loss is NaN/Inf")
                raise FloatingPointError(f"{name}: non-finite loss at step {step}")
            total.backward()
            if p.grad is None or not torch.isfinite(p.grad).all():
                _fail(telemetry_dir, name, step, "gradient absent/NaN/Inf")
                raise FloatingPointError(f"{name}: non-finite/absent gradient at step {step}")
            optim.step()
            # safety bounds (warn, then hard-fail at implausible limits)
            with torch.no_grad():
                warns, hard = bounds.check(name, p, step)
            for w in warns:
                bound_crossings.append({"step": step, "level": "warn", "detail": w})
            if hard:
                bound_crossings.append({"step": step, "level": "hard", "detail": hard})
                _fail(telemetry_dir, name, step, f"parameter bound exceeded: {hard}")
                raise FloatingPointError(f"{name}: parameter safety hard limit exceeded at step {step}: {hard}")
            do_log = (opt_s.log_every and (step % opt_s.log_every == 0 or step == opt_s.steps - 1)) \
                or step in (0, opt_s.steps - 1)
            if do_log:
                with torch.no_grad():
                    loss_hist.append(float(total)); img_hist.append(float(img))
                    reg_hist.append(float(rloss) if torch.is_tensor(rloss) else float(rloss))
            if tele is not None and (do_log or (opt_s.log_every and step % opt_s.log_every == 0)):
                tele.record(step=step, total=total, img=img, rloss=rloss, params=p,
                            grad=p.grad, lr=opt_s.lr)

    if tele is not None:
        tele.close()

    with torch.no_grad():
        final = apply_scopes(name, p, scopes, angles)
        final = apply_gauge(name, final, angles, gcfg).detach()

    grad_summary = {}
    if p.grad is not None:
        with torch.no_grad():
            grad_summary = {"grad_norm_total": float(p.grad.norm()),
                            "grad_norm_translation": float(p.grad[:, :2].norm())}
    result = ConstrainedResult(
        model=name, params=final, loss_history=loss_hist,
        image_loss_history=img_hist, reg_loss_history=reg_hist,
        gauge=gauge_report(name, final), n_tilts=n_tilts,
        scopes={"translation": scopes.translation, "rotation": scopes.rotation,
                "isotropic_scale": scopes.isotropic_scale},
        settings={"optimizer": opt_s.optimizer, "lr": opt_s.lr, "steps": opt_s.steps,
                  "shape_xy": list(recon.shape_xy), "pixel_size_A": list(recon.pixel_size_xy_A),
                  "bound_crossings": bound_crossings})

    if result_dir is not None:
        _write_result(result_dir, model_obj=model, final=final, angles=angles, scopes=scopes,
                      gcfg=gcfg, reg=reg, recon=recon, loss_hist=loss_hist,
                      grad_summary=grad_summary, stage_label=stage_label, seed=seed)
    return result


class _Telemetry:
    """Per-step JSONL + TSV telemetry (spec §21). Compact: no full tensors."""
    def __init__(self, out_dir, stage_label, model_name):
        from pathlib import Path
        self.d = Path(out_dir); self.d.mkdir(parents=True, exist_ok=True)
        self.stage = stage_label; self.model = model_name
        self.events = (self.d / "training_events.jsonl").open("a")
        self.loss = (self.d / "loss_history.tsv").open("a")
        self.grad = (self.d / "gradient_history.tsv").open("a")
        self.par = (self.d / "parameter_history.tsv").open("a")
        if self.loss.tell() == 0:
            self.loss.write("stage\tstep\ttotal\timage\treg\tlr\n")
            self.grad.write("stage\tstep\tgrad_norm\tgrad_norm_translation\tn_nonfinite\n")
            self.par.write("stage\tstep\ttx_mean\tty_mean\trot_mean\tscale_mean\n")

    def record(self, *, step, total, img, rloss, params, grad, lr):
        import json as _j
        import torch
        with torch.no_grad():
            p = params.detach()
            gnorm = float(grad.norm()) if grad is not None else None
            gtrans = float(grad[:, :2].norm()) if grad is not None else None
            nnf = int((~torch.isfinite(p)).sum())
            rot = float(p[:, 2].mean()) if p.shape[1] >= 3 else 0.0
            sca = float(p[:, 3].mean()) if p.shape[1] >= 4 else 0.0
            rec = {"stage": self.stage, "model": self.model, "step": step,
                   "total_loss": float(total), "image_loss": float(img),
                   "reg_loss": float(rloss) if torch.is_tensor(rloss) else float(rloss),
                   "lr": lr, "grad_norm": gnorm, "grad_norm_translation": gtrans,
                   "n_nonfinite": nnf, "tx_mean": float(p[:, 0].mean()),
                   "ty_mean": float(p[:, 1].mean()), "rot_mean": rot, "scale_mean": sca}
        self.events.write(_j.dumps(rec) + "\n"); self.events.flush()
        self.loss.write(f"{self.stage}\t{step}\t{rec['total_loss']}\t{rec['image_loss']}\t{rec['reg_loss']}\t{lr}\n")
        self.grad.write(f"{self.stage}\t{step}\t{gnorm}\t{gtrans}\t{nnf}\n")
        self.par.write(f"{self.stage}\t{step}\t{rec['tx_mean']}\t{rec['ty_mean']}\t{rot}\t{sca}\n")

    def close(self):
        for fh in (self.events, self.loss, self.grad, self.par):
            try:
                fh.close()
            except Exception:
                pass


def _fail(telemetry_dir, name, step, reason):
    if not telemetry_dir:
        return
    from pathlib import Path
    import json as _j
    p = Path(telemetry_dir) / "training_failure.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_j.dumps({"model": name, "step": step, "reason": reason}, indent=2))


def _write_result(result_dir, *, model_obj, final, angles, scopes, gcfg, reg, recon,
                  loss_hist, grad_summary, stage_label, seed):
    from . import result_contract as RC
    import datetime
    versions = {}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    RC.write_constrained_result(
        result_dir, model=model_obj.name, params=final, tilt_angles=angles,
        param_names=tuple(model_obj.param_names),
        scopes={"translation": scopes.translation, "rotation": scopes.rotation,
                "isotropic_scale": scopes.isotropic_scale},
        gauge={"anchor_tilt": gcfg.anchor_tilt, "zero_mean_rotation": gcfg.zero_mean_rotation,
               "zero_mean_log_scale": gcfg.zero_mean_log_scale},
        regularization={"rotation_prior": reg.rotation_prior, "smoothness": reg.smoothness},
        working_raw_grid=None, working_aligned_grid={"shape_xy": list(recon.shape_xy),
                                                     "pixel_size_A": list(recon.pixel_size_xy_A)},
        input_hashes=None, warp_project_hash=None, loss_history=loss_hist,
        gradient_summary=grad_summary,
        stage_history=[{"stage": stage_label, "model": model_obj.name}],
        software_versions=versions, cuda_info=None, seed=seed,
        start_time=now, end_time=now, completion_status="completed")
