"""Synthetic recovery: each model recovers residuals it can represent, fails on
those it cannot, and never invents forbidden degrees of freedom."""
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


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.N = 4
        self.shape = (256, 192)
        self.pix = 10.0
        self.center = torch.tensor(cf.physical_center_xy(self.shape, self.pix), dtype=torch.float64)
        # spread of points across the image in Angstrom
        xs = np.linspace(0, self.shape[0] * self.pix, 5)
        ys = np.linspace(0, self.shape[1] * self.pix, 4)
        pts = np.array([(x, y) for y in ys for x in xs], float)
        self.points = torch.tensor(pts, dtype=torch.float64)

    def _truth_targets(self, kind):
        """Per-tilt target points for a known truth transform of the given kind."""
        c = self.center.numpy()
        targets = []
        truths = []
        for i in range(self.N):
            if kind == "translation":
                A = np.eye(2); t = np.array([12.0 - 3 * i, -8.0 + 2 * i])
            elif kind == "rigid":
                A = _rot(np.deg2rad(4.0 + i)); t = np.array([6.0, -4.0 + i])
            elif kind == "similarity":
                A = np.exp(0.03 * (i + 1)) * _rot(np.deg2rad(3.0 + i)); t = np.array([5.0, -3.0])
            elif kind == "affine":
                A = _rot(np.deg2rad(3.0)) @ np.array([[np.exp(0.04), 0.05 + 0.01 * i], [0.0, np.exp(-0.03)]])
                t = np.array([4.0 + i, -2.0])
            else:
                raise ValueError(kind)
            P = self.points.numpy()
            tgt = (P - c) @ A.T + t + c
            targets.append(tgt)
            truths.append((A, t))
        return torch.tensor(np.stack(targets), dtype=torch.float64), truths

    def _fit(self, model_name, targets, iters=120):
        m = am.get_model(model_name)
        p = m.identity_params(self.N).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([p], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            out = m.apply_centered(p, self.points, self.center)
            loss = ((out - targets) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            out = m.apply_centered(p, self.points, self.center)
            rms = float(torch.sqrt(((out - targets) ** 2).sum(-1).mean()))
        return m, p.detach(), rms

    def test_recovery_hierarchy(self):
        # (truth_kind, {model: should_recover})
        order = ["translation", "rigid", "similarity", "affine"]
        for ti, truth_kind in enumerate(order):
            targets, _ = self._truth_targets(truth_kind)
            for mi, model_name in enumerate(order):
                _, _, rms = self._fit(model_name, targets)
                can = mi >= ti  # model at least as general as the truth
                if can:
                    self.assertLess(rms, 1e-2, f"{model_name} should recover {truth_kind} (rms={rms:.4g} A)")
                else:
                    self.assertGreater(rms, 1.0, f"{model_name} must NOT fit {truth_kind} (rms={rms:.4g} A)")

    def test_restricted_models_do_not_invent_dof(self):
        # Fit rigid to a translation-only truth -> rotation ~ 0.
        targets, _ = self._truth_targets("translation")
        m, p, rms = self._fit("rigid", targets)
        self.assertLess(rms, 1e-2)
        self.assertLess(float(p[:, 2].abs().max()), 1e-3, "rigid invented rotation on translation truth")

        # Fit affine to a similarity truth -> shear ~ 0 and isotropic (alpha ~ beta).
        targets, _ = self._truth_targets("similarity")
        m, p, rms = self._fit("affine", targets)
        self.assertLess(rms, 1e-2)
        self.assertLess(float(p[:, 5].abs().max()), 1e-3, "affine invented shear on similarity truth")
        self.assertLess(float((p[:, 3] - p[:, 4]).abs().max()), 1e-3, "affine invented anisotropy on similarity truth")


if __name__ == "__main__":
    unittest.main()
