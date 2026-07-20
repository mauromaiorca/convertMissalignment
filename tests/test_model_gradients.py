"""Autograd vs finite-difference gradient checks for every constrained model.

Fails if any model's gradients are detached, missing, or numerically wrong.
Also asserts that the gradient w.r.t. each *forbidden-in-simpler-models*
parameter (rotation, log-scale, shear) is non-zero, proving the parameter is
genuinely differentiable (not stop-gradiented)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

if HAVE_TORCH:
    import alignment_models as am
    from alignment_models import coordinate_frames as cf


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class GradientChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.N = 3
        self.shape = (256, 192)
        self.pix = 10.0
        self.points = torch.tensor(
            [[20.0, 30.0], [200.0, 50.0], [120.0, 150.0], [240.0, 180.0]],
            dtype=torch.float64,
        ) * self.pix / 1.0
        self.center = torch.tensor(cf.physical_center_xy(self.shape, self.pix), dtype=torch.float64)

    def _loss(self, model, params):
        out = model.apply_centered(params, self.points, self.center)
        return (out ** 2).sum()

    def test_autograd_matches_finite_difference(self):
        for name in am.NESTING_ORDER:
            m = am.get_model(name)
            base = m.identity_params(self.N) + 0.15 * torch.randn((self.N, m.n_params), dtype=torch.float64)
            p = base.clone().requires_grad_(True)
            loss = self._loss(m, p)
            (grad,) = torch.autograd.grad(loss, p)
            grad = grad.detach().numpy()

            # central finite differences
            eps = 1e-6
            fd = np.zeros_like(grad)
            flat = base.clone()
            for i in range(self.N):
                for j in range(m.n_params):
                    pp = flat.clone(); pp[i, j] += eps
                    pm = flat.clone(); pm[i, j] -= eps
                    lp = float(self._loss(m, pp))
                    lm = float(self._loss(m, pm))
                    fd[i, j] = (lp - lm) / (2 * eps)
            self.assertTrue(
                np.allclose(grad, fd, atol=1e-4, rtol=1e-4),
                f"{name}: autograd vs FD mismatch\n grad={grad}\n fd={fd}",
            )

    def test_forbidden_params_are_differentiable(self):
        # rotation (rigid), log_scale (similarity), shear (affine) must have grad
        cases = {
            "rigid": 2,        # phi
            "similarity": 3,   # log_scale
            "affine": 5,       # shear
        }
        for name, col in cases.items():
            m = am.get_model(name)
            p = (m.identity_params(self.N) + 0.2 * torch.randn((self.N, m.n_params), dtype=torch.float64)).requires_grad_(True)
            loss = self._loss(m, p)
            (grad,) = torch.autograd.grad(loss, p)
            self.assertGreater(
                float(grad[:, col].abs().sum()), 1e-8,
                f"{name} param column {col} appears detached (zero gradient)",
            )

    def test_translation_rotation_grad_is_zero_by_construction(self):
        # translation model has no rotation column; its A is exactly identity,
        # so there is nothing to detach -- sanity that A does not depend on params.
        m = am.get_model("translation")
        p = (m.identity_params(self.N) + 0.3 * torch.randn((self.N, 2), dtype=torch.float64)).requires_grad_(True)
        A = m.linear_matrices(p)
        # d(sum A)/dp must be exactly zero (A is constant identity)
        gA = torch.autograd.grad(A.sum(), p, allow_unused=True)[0]
        self.assertTrue(gA is None or float(gA.abs().sum()) == 0.0)


if __name__ == "__main__":
    unittest.main()
