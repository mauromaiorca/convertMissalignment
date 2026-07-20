"""The single image-based constrained optimizer (spec §19) end to end, through a
REAL differentiable image loss (torch grid_sample), with nestedness (§31).

Proves the optimized variables are the constrained DOF and that they affect a real
differentiable image loss: params -> scopes/gauge -> Option-B field -> grid_sample
warp -> image L2 -> autograd -> Adam. NOT the MissAlignment projector (cluster-only).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    HAVE = True
except Exception:
    HAVE = False

if HAVE:
    import alignment_models as am
    from alignment_models.constraints import GaugeConfig
    from alignment_models.materialize import materialize_model_field
    from alignment_models.optimize_constrained_2d import (
        OptimizerSettings, ReconstructionSettings, grid_sample_image_scorer,
        optimize_constrained_2d,
    )

# No gauge fixing during a pure image-recovery test: the gauge resolves an
# export-time degeneracy (it anchors translation to the zero-tilt index and
# zero-means rotation/scale), which would prevent matching an observed image that
# carries a constant per-tilt shift. Recovery of the raw constrained DOF is the
# property under test here; the gauge is exercised in the export tests.
NO_GAUGE = lambda: GaugeConfig(anchor_tilt="none", zero_mean_rotation=False,
                               zero_mean_log_scale=False, zero_mean_shear=False) if HAVE else None

# Zero regularization so the IMAGE LOSS alone drives the parameters (the default
# RegularizationConfig has nonzero priors that would otherwise bias rotation/scale).
ZERO_REG = (lambda: am.regularization.RegularizationConfig(
    translation_prior=0.0, rotation_prior=0.0, scale_prior=0.0, shear_prior=0.0,
    smoothness=0.0, curvature=0.0)) if HAVE else (lambda: None)


def _phantom(ny=72, nx=72):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    img = np.zeros((ny, nx))
    for cx, cy, s, a in [(22, 28, 3.0, 1.0), (50, 44, 4.5, 0.8),
                         (34, 56, 2.5, 0.6), (56, 20, 3.5, 0.7)]:
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    img += 0.15 * (xx / nx)  # break symmetry
    return torch.tensor(img, dtype=torch.float64)


def _true_params(name, n, scale=1.0):
    base = {
        "translation": [[1.4, -1.1]],
        "rigid": [[1.4, -1.1, 0.06]],
        "similarity": [[1.4, -1.1, 0.06, 0.04]],
    }[name]
    return (torch.tensor(base, dtype=torch.float64).repeat(n, 1)) * scale


@unittest.skipUnless(HAVE, "torch unavailable")
class OptimizeConstrained2dTests(unittest.TestCase):
    def _observed(self, model, true_p, shape_xy, ref):
        field = materialize_model_field(model, true_p, shape_xy, as_image=True).detach()
        scorer = grid_sample_image_scorer(ref, ref)  # reuse warp to synthesize target
        # warp ref by the true field to get the observed image stack
        F = torch.nn.functional
        ny, nx = ref.shape
        iy, ix = torch.meshgrid(torch.arange(ny, dtype=torch.float64),
                                torch.arange(nx, dtype=torch.float64), indexing="ij")
        obs = []
        for i in range(field.shape[0]):
            qx = ix + field[i, :, :, 0]; qy = iy + field[i, :, :, 1]
            gx = 2.0 * qx / (nx - 1) - 1.0; gy = 2.0 * qy / (ny - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            obs.append(F.grid_sample(ref[None, None], grid, mode="bilinear",
                                     align_corners=True, padding_mode="border")[0, 0])
        return torch.stack(obs)

    def test_recovers_each_model_through_image_loss(self):
        ref = _phantom()
        shape_xy = (ref.shape[1], ref.shape[0])
        for name in ("translation", "rigid", "similarity"):
            model = am.get_model(name)
            n = 3
            true_p = _true_params(name, n)
            observed = self._observed(model, true_p, shape_xy, ref)
            scorer = grid_sample_image_scorer(ref, observed)
            res = optimize_constrained_2d(
                alignment_model=model,
                initial_parameters=model.identity_params(n),
                reconstruct_and_score=scorer,
                tilt_angles=np.linspace(-30, 30, n),
                gauge=NO_GAUGE(), regularization=ZERO_REG(),
                optimizer_settings=OptimizerSettings(steps=900, lr=0.02),
                reconstruction_settings=ReconstructionSettings(
                    shape_xy=shape_xy, pixel_size_xy_A=(1.0, 1.0)),
            )
            self.assertLess(res.image_loss_history[-1], 1e-4,
                            f"{name}: final image loss {res.image_loss_history[-1]:.2e}")
            # with the gauge disabled the raw constrained DOF are recovered: the
            # materialized field must match the truth to sub-pixel accuracy.
            recon_field = materialize_model_field(model, res.params, shape_xy, as_image=True)
            true_field = materialize_model_field(model, true_p, shape_xy, as_image=True)
            dev = float((recon_field - true_field).abs().max())
            self.assertLess(dev, 0.2, f"{name}: recovered field deviates {dev:.3f}px")

    def test_nestedness_translation_data(self):
        # On translation-only data, rigid -> ~0 rotation, similarity -> ~0 rot & ~unit scale.
        ref = _phantom()
        shape_xy = (ref.shape[1], ref.shape[0])
        n = 3
        tmodel = am.get_model("translation")
        true_p = _true_params("translation", n)
        observed = self._observed(tmodel, true_p, shape_xy, ref)

        rigid = am.get_model("rigid")
        rr = optimize_constrained_2d(
            alignment_model=rigid, initial_parameters=rigid.identity_params(n),
            reconstruct_and_score=grid_sample_image_scorer(ref, observed),
            tilt_angles=np.linspace(-30, 30, n), gauge=NO_GAUGE(), regularization=ZERO_REG(),
            optimizer_settings=OptimizerSettings(steps=900, lr=0.02),
            reconstruction_settings=ReconstructionSettings(shape_xy=shape_xy))
        phi = rr.params[:, 2].abs().max().item()
        self.assertLess(phi, 0.03, f"rigid on translation data: |phi|max {phi:.4f}")

        sim = am.get_model("similarity")
        sr = optimize_constrained_2d(
            alignment_model=sim, initial_parameters=sim.identity_params(n),
            reconstruct_and_score=grid_sample_image_scorer(ref, observed),
            tilt_angles=np.linspace(-30, 30, n), gauge=NO_GAUGE(), regularization=ZERO_REG(),
            optimizer_settings=OptimizerSettings(steps=1000, lr=0.02),
            reconstruction_settings=ReconstructionSettings(shape_xy=shape_xy))
        self.assertLess(sr.params[:, 2].abs().max().item(), 0.03, "similarity rot on transl data")
        self.assertLess(sr.params[:, 3].abs().max().item(), 0.03, "similarity log_scale on transl data")

    def test_image_loss_gradient_reaches_every_dof(self):
        # The core requirement: the constrained parameters affect a REAL differentiable
        # image loss. The image-loss gradient (regularization OFF) must be finite and
        # NONZERO in every DOF: translation has dL/d(tx,ty); rigid adds dL/dphi;
        # similarity adds dL/dlog_scale. Asymmetric data makes them identifiable.
        ref = _phantom()
        shape_xy = (ref.shape[1], ref.shape[0])
        expect_dofs = {"translation": 2, "rigid": 3, "similarity": 4}
        for name, ndof in expect_dofs.items():
            model = am.get_model(name)
            n = 3
            observed = self._observed(model, _true_params(name, n), shape_xy, ref)
            scorer = grid_sample_image_scorer(ref, observed)
            p = (model.identity_params(n) + 0.03).clone().requires_grad_(True)
            field = materialize_model_field(model, p, shape_xy, as_image=True)
            loss = scorer(field)
            loss.backward()
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: non-finite image-loss grad")
            per_dof = p.grad.abs().sum(dim=0)  # (n_params,)
            for k in range(ndof):
                self.assertGreater(float(per_dof[k]), 1e-8,
                                   f"{name}: dead image-loss gradient in DOF {k} "
                                   f"(param {model.param_names[k]})")

    def test_dead_scorer_does_not_move_params(self):
        # A scorer that detaches the field must NOT move the params via the image
        # loss (regularization off) -- guards against a spurious update path.
        ref = _phantom()
        shape_xy = (ref.shape[1], ref.shape[0])
        model = am.get_model("rigid")

        def detached_scorer(field, **_):
            return (field.detach() ** 2).mean()  # explicitly cuts the graph

        res = optimize_constrained_2d(
            alignment_model=model, initial_parameters=model.identity_params(3) + 0.1,
            reconstruct_and_score=detached_scorer, gauge=NO_GAUGE(),
            regularization=ZERO_REG(),
            optimizer_settings=OptimizerSettings(steps=20, lr=0.05),
            reconstruction_settings=ReconstructionSettings(shape_xy=shape_xy))
        # params stayed at the (non-identity) start: the detached image loss moved nothing
        self.assertLess(float((res.params - (model.identity_params(3) + 0.1)).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
