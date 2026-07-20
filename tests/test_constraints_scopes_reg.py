"""Tests for gauge fixing, parameter scopes, and regularization."""
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
    from alignment_models.constraints import GaugeConfig, apply_gauge, gauge_report, anchor_index
    from alignment_models.parameter_scope import (
        ScopeConfig, apply_scopes, decompose, degrees_of_freedom,
    )
    from alignment_models.regularization import RegularizationConfig, regularization_loss


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class GaugeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.angles = [-30.0, -10.0, 2.0, 18.0, 40.0]  # anchor = index 2 (closest to 0)

    def test_anchor_index(self):
        self.assertEqual(anchor_index(self.angles), 2)

    def test_zero_mean_and_anchor(self):
        m = am.get_model("affine")
        p = m.identity_params(5) + 0.3 * torch.randn((5, 6), dtype=torch.float64)
        g = apply_gauge("affine", p, self.angles, GaugeConfig())
        rep = gauge_report("affine", g)
        self.assertAlmostEqual(rep["mean_phi"], 0.0, places=9)
        self.assertAlmostEqual(rep["mean_iso_log_scale"], 0.0, places=9)
        self.assertAlmostEqual(rep["mean_shear"], 0.0, places=9)
        # anchor tilt translation ~ 0
        self.assertAlmostEqual(float(g[2, 0]), 0.0, places=9)
        self.assertAlmostEqual(float(g[2, 1]), 0.0, places=9)

    def test_relative_geometry_preserved(self):
        m = am.get_model("rigid")
        p = m.identity_params(5) + 0.3 * torch.randn((5, 3), dtype=torch.float64)
        g = apply_gauge("rigid", p, self.angles, GaugeConfig())
        # pairwise differences of phi unchanged by gauge fixing
        dp = (p[:, 2][:, None] - p[:, 2][None, :]).numpy()
        dg = (g[:, 2][:, None] - g[:, 2][None, :]).numpy()
        self.assertTrue(np.allclose(dp, dg, atol=1e-10))


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class ScopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.angles = [-20.0, -5.0, 5.0, 25.0]

    def test_global_makes_component_constant(self):
        m = am.get_model("affine")
        p = m.identity_params(4) + 0.3 * torch.randn((4, 6), dtype=torch.float64)
        cfg = ScopeConfig(shear="global", anisotropic_scale="global", isotropic_scale="global",
                          rotation="per_tilt", translation="per_tilt")
        q = apply_scopes("affine", p, cfg, self.angles)
        comp = decompose("affine", q)
        for key in ("isotropic_log_scale", "anisotropic_log_scale", "shear"):
            v = comp[key].detach().numpy()
            self.assertTrue(np.allclose(v, v[0], atol=1e-10), f"{key} not global")

    def test_fixed_zeroes_component(self):
        m = am.get_model("affine")
        p = m.identity_params(4) + 0.3 * torch.randn((4, 6), dtype=torch.float64)
        cfg = ScopeConfig(shear="fixed")
        q = apply_scopes("affine", p, cfg, self.angles)
        self.assertTrue(np.allclose(decompose("affine", q)["shear"].detach().numpy(), 0.0))

    def test_dof_counts(self):
        cfg = ScopeConfig()  # defaults
        # affine, 4 tilts: tx,ty per_tilt (4+4) + rotation per_tilt_smooth (4)
        # + iso global (1) + aniso global (1) + shear global (1) = 15
        self.assertEqual(degrees_of_freedom("affine", 4, cfg), 4 + 4 + 4 + 1 + 1 + 1)
        # translation defaults: tx,ty per_tilt
        self.assertEqual(degrees_of_freedom("translation", 4, cfg), 8)


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class ScopeGradientFlowTests(unittest.TestCase):
    """Scope/gauge projections must stay on the autograd graph (refinement path).

    Regression for the adversarial-review finding: global/spline projections that
    convert to Python floats or numpy silently freeze the parameters."""

    def _grad_through_scope(self, scope_kwargs, n=8):
        m = am.get_model("affine")
        torch.manual_seed(0)
        p = (m.identity_params(n) + 0.2 * torch.randn((n, 6), dtype=torch.float64)).requires_grad_(True)
        cfg = ScopeConfig(**scope_kwargs)
        angles = list(np.linspace(-30, 30, n))
        q = apply_scopes("affine", p, cfg, angles)
        # a loss that depends on every component of the projected params
        loss = (q ** 2).sum()
        (g,) = torch.autograd.grad(loss, p)
        return g

    def test_global_scope_keeps_gradient(self):
        # isotropic/anisotropic scale (cols 3,4) and shear (col 5) are global by default
        g = self._grad_through_scope(dict(isotropic_scale="global", anisotropic_scale="global", shear="global"))
        self.assertGreater(float(g[:, 3].abs().sum()), 1e-9, "global iso scale frozen (grad detached)")
        self.assertGreater(float(g[:, 4].abs().sum()), 1e-9, "global aniso scale frozen (grad detached)")
        self.assertGreater(float(g[:, 5].abs().sum()), 1e-9, "global shear frozen (grad detached)")

    def test_spline_scope_keeps_gradient(self):
        # n (8) > control points (5) so the interpolation path actually runs
        g = self._grad_through_scope(dict(rotation="spline"))
        self.assertGreater(float(g[:, 2].abs().sum()), 1e-9, "spline rotation frozen (grad detached)")

    def test_gauge_keeps_gradient(self):
        m = am.get_model("affine")
        p = (m.identity_params(5) + 0.2 * torch.randn((5, 6), dtype=torch.float64)).requires_grad_(True)
        g_in = apply_gauge("affine", p, [-20.0, -5.0, 0.0, 12.0, 30.0], GaugeConfig())
        loss = (g_in ** 2).sum()
        (g,) = torch.autograd.grad(loss, p)
        self.assertGreater(float(g.abs().sum()), 1e-9, "gauge projection detached gradient")


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class RegularizationTests(unittest.TestCase):
    def setUp(self):
        self.angles = [-20.0, -5.0, 5.0, 25.0]

    def test_zero_at_identity(self):
        m = am.get_model("affine")
        p = m.identity_params(4)
        loss = regularization_loss("affine", p, self.angles, RegularizationConfig())
        self.assertAlmostEqual(float(loss), 0.0, places=12)

    def test_nonnegative_and_differentiable(self):
        m = am.get_model("affine")
        p = (m.identity_params(4) + 0.2 * torch.randn((4, 6), dtype=torch.float64)).requires_grad_(True)
        loss = regularization_loss("affine", p, self.angles, RegularizationConfig())
        self.assertGreaterEqual(float(loss), 0.0)
        (g,) = torch.autograd.grad(loss, p)
        self.assertGreater(float(g.abs().sum()), 0.0)

    def test_ordering_matters(self):
        m = am.get_model("rigid")
        # rotation that is smooth in tilt-angle order but jagged in acquisition order
        p = m.identity_params(4).clone()
        p[:, 2] = torch.tensor([0.0, 0.3, 0.1, 0.4], dtype=torch.float64)  # acquisition order
        angles = [0.0, 30.0, 10.0, 40.0]  # tilt-angle order reshuffles
        l_acq = float(regularization_loss("rigid", p, angles, RegularizationConfig(ordering="acquisition", smoothness=1.0, rotation_prior=0.0)))
        l_tilt = float(regularization_loss("rigid", p, angles, RegularizationConfig(ordering="tilt_angle", smoothness=1.0, rotation_prior=0.0)))
        self.assertNotAlmostEqual(l_acq, l_tilt, places=6)


if __name__ == "__main__":
    unittest.main()
