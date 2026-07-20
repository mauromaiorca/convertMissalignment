"""Extended gradient validation: float64 + float32, multi-seed, scopes, gauge, reg."""
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
    from alignment_models.constraints import GaugeConfig, apply_gauge
    from alignment_models.parameter_scope import ScopeConfig, apply_scopes
    from alignment_models.regularization import RegularizationConfig, regularization_loss


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class DtypeGradientTests(unittest.TestCase):
    N = 3
    shape = (256, 192)
    pix = 10.0

    def _loss(self, model, params, points, center):
        out = model.apply_centered(params, points, center)
        return (out ** 2).sum()

    def _check(self, dtype, eps, atol, rtol, seeds):
        center = torch.tensor(cf.physical_center_xy(self.shape, self.pix), dtype=dtype)
        points = torch.tensor([[200., 300.], [2000., 500.], [1200., 1500.], [2400., 1800.]], dtype=dtype)
        for name in am.NESTING_ORDER:
            for seed in seeds:
                torch.manual_seed(seed)
                m = am.get_model(name, dtype=dtype)
                base = m.identity_params(self.N) + 0.1 * torch.randn((self.N, m.n_params), dtype=dtype)
                p = base.clone().requires_grad_(True)
                loss = self._loss(m, p, points, center)
                (grad,) = torch.autograd.grad(loss, p)
                self.assertTrue(torch.isfinite(grad).all(), f"{name} {dtype}: non-finite grad")
                grad = grad.detach().cpu().numpy()
                fd = np.zeros_like(grad)
                for i in range(self.N):
                    for j in range(m.n_params):
                        pp = base.clone(); pp[i, j] += eps
                        pm = base.clone(); pm[i, j] -= eps
                        fd[i, j] = (float(self._loss(m, pp, points, center)) - float(self._loss(m, pm, points, center))) / (2 * eps)
                self.assertTrue(np.allclose(grad, fd, atol=atol, rtol=rtol),
                                f"{name} {dtype} seed {seed}: autograd vs FD mismatch")

    def test_float64(self):
        self._check(torch.float64, eps=1e-6, atol=1e-4, rtol=1e-4, seeds=[0, 1, 2, 3, 4])

    def test_float32_matches_float64_autograd(self):
        # float32 finite differences are unreliable for large-magnitude losses
        # (catastrophic cancellation), so the correct check is that float32
        # autograd reproduces the float64 autograd gradient and stays finite.
        center64 = torch.tensor(cf.physical_center_xy(self.shape, self.pix), dtype=torch.float64)
        center32 = center64.to(torch.float32)
        pts = torch.tensor([[200., 300.], [800., 500.], [400., 700.]], dtype=torch.float64)
        for name in am.NESTING_ORDER:
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                m64 = am.get_model(name, dtype=torch.float64)
                base = m64.identity_params(self.N) + 0.1 * torch.randn((self.N, m64.n_params), dtype=torch.float64)
                p64 = base.clone().requires_grad_(True)
                (g64,) = torch.autograd.grad(self._loss(m64, p64, pts, center64), p64)
                m32 = am.get_model(name, dtype=torch.float32)
                p32 = base.to(torch.float32).clone().requires_grad_(True)
                (g32,) = torch.autograd.grad(self._loss(m32, p32, pts.to(torch.float32), center32), p32)
                self.assertTrue(torch.isfinite(g32).all(), f"{name}: non-finite float32 grad")
                rel = (g32.double() - g64).abs().max() / (g64.abs().max() + 1e-9)
                self.assertLess(float(rel), 1e-3, f"{name} seed {seed}: float32 autograd diverges from float64")


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class ScopeGaugeRegGradientTests(unittest.TestCase):
    angles = [-30.0, -10.0, 0.0, 12.0, 28.0, 41.0, 55.0, -50.0]

    def _grad_finite_nonzero(self, fn, dtype):
        m = am.get_model("affine", dtype=dtype)
        torch.manual_seed(7)
        p = (m.identity_params(8) + 0.2 * torch.randn((8, 6), dtype=dtype)).requires_grad_(True)
        loss = fn(p)
        (g,) = torch.autograd.grad(loss, p)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.abs().sum()), 1e-9)

    def test_scopes_all_kinds_both_dtypes(self):
        for dtype in (torch.float64, torch.float32):
            for scope in ("per_tilt", "per_tilt_smooth", "global", "spline"):
                cfg = ScopeConfig(rotation=scope, isotropic_scale=scope, anisotropic_scale=scope, shear=scope)
                self._grad_finite_nonzero(lambda p: (apply_scopes("affine", p, cfg, self.angles) ** 2).sum(), dtype)

    def test_gauge_both_dtypes(self):
        for dtype in (torch.float64, torch.float32):
            self._grad_finite_nonzero(
                lambda p: (apply_gauge("affine", p, self.angles, GaugeConfig()) ** 2).sum(), dtype)

    def test_regularization_both_dtypes(self):
        for dtype in (torch.float64, torch.float32):
            cfg = RegularizationConfig(rotation_prior=0.1, scale_prior=0.1, shear_prior=0.1, smoothness=0.1, curvature=0.05)
            self._grad_finite_nonzero(
                lambda p: regularization_loss("affine", p, self.angles, cfg), dtype)

    def test_no_inplace_autograd_error_in_full_pipeline(self):
        # scopes -> gauge -> matrices -> reg, all in one graph, must backprop cleanly
        m = am.get_model("affine")
        torch.manual_seed(3)
        p = (m.identity_params(8) + 0.2 * torch.randn((8, 6), dtype=torch.float64)).requires_grad_(True)
        cfg_s = ScopeConfig()
        q = apply_scopes("affine", p, cfg_s, self.angles)
        q = apply_gauge("affine", q, self.angles, GaugeConfig())
        A = m.linear_matrices(q)
        loss = (A ** 2).sum() + regularization_loss("affine", q, self.angles, RegularizationConfig())
        loss.backward()  # raises on in-place autograd errors
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
