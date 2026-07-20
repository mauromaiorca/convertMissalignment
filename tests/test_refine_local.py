"""Local refinement engine: recovery hierarchy and constraint enforcement.

Proves there is a real, executable LOCAL refinement forward pass (autograd over
coordinate correspondences, staged schedule, scopes, gauge, regularization),
independent of warpylib/GPU."""
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
    from alignment_models.constraints import GaugeConfig
    from alignment_models.parameter_scope import ScopeConfig
    from alignment_models.refine import refine
    from alignment_models.refinement_config import RefinementConfig
    from alignment_models.regularization import RegularizationConfig


def _rot(d):
    a = np.deg2rad(d)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _free_config(model):
    """Per-tilt scopes, no reg, no gauge: a well-posed coordinate-recovery setup."""
    return RefinementConfig(
        model=model, schedule="automatic",
        scope=ScopeConfig(translation="per_tilt", rotation="per_tilt",
                          isotropic_scale="per_tilt", anisotropic_scale="per_tilt", shear="per_tilt"),
        gauge=GaugeConfig(anchor_tilt="none", zero_mean_rotation=False,
                          zero_mean_log_scale=False, zero_mean_shear=False),
        regularization=RegularizationConfig(translation_prior=0, rotation_prior=0, scale_prior=0,
                                            shear_prior=0, smoothness=0, curvature=0),
    )


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class LocalRefineTests(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.shape = (256, 192)
        self.pix = 10.0
        self.center = cf.physical_center_xy(self.shape, self.pix)
        xs = np.linspace(0, self.shape[0] * self.pix, 6)
        ys = np.linspace(0, self.shape[1] * self.pix, 5)
        self.base = np.array([(x, y) for y in ys for x in xs])
        self.src = np.tile(self.base, (self.n, 1, 1))
        self.angles = list(np.linspace(-40, 40, self.n))

    def _truth(self, kind):
        out = []
        for i in range(self.n):
            if kind == "translation":
                A, t = np.eye(2), np.array([15.0 - 3 * i, -10.0 + 2 * i])
            elif kind == "rigid":
                A, t = _rot(3.0 + i), np.array([8.0, -5.0 + i])
            elif kind == "similarity":
                A, t = np.exp(0.02 * (i + 1)) * _rot(2.5 + i), np.array([5.0, -3.0])
            elif kind == "affine":
                A = _rot(3.0) @ np.array([[np.exp(0.03), 0.04 + 0.01 * i], [0.0, np.exp(-0.02)]])
                t = np.array([4.0 + i, -2.0])
            out.append((A, t))
        tgt = np.stack([(self.src[i] - self.center) @ A.T + t + self.center for i, (A, t) in enumerate(out)])
        return tgt

    def test_recovery_hierarchy(self):
        order = ["translation", "rigid", "similarity", "affine"]
        for ti, truth_kind in enumerate(order):
            tgt = self._truth(truth_kind)
            for mi, model in enumerate(order):
                r = refine(_free_config(model), self.src, tgt, self.angles, self.shape, self.pix,
                           iters_per_stage=400)
                if mi >= ti:
                    self.assertLess(r.final_data_rms_A, 1e-2,
                                    f"{model} should recover {truth_kind} (rms={r.final_data_rms_A:.4g} A)")
                else:
                    self.assertGreater(r.final_data_rms_A, 0.5,
                                       f"{model} must NOT fit {truth_kind} (rms={r.final_data_rms_A:.4g} A)")

    def test_global_scope_enforces_constraint(self):
        # Truth has PER-TILT scale; a global-scale config must NOT fit it (residual large),
        # proving the scope genuinely constrains the optimization.
        tgt = self._truth("similarity")  # per-tilt isotropic scale
        cfg = RefinementConfig(
            model="similarity", schedule="automatic",
            scope=ScopeConfig(translation="per_tilt", rotation="per_tilt", isotropic_scale="global"),
            gauge=GaugeConfig(anchor_tilt="none", zero_mean_rotation=False, zero_mean_log_scale=False, zero_mean_shear=False),
            regularization=RegularizationConfig(rotation_prior=0, scale_prior=0, shear_prior=0, smoothness=0, curvature=0),
        )
        r = refine(cfg, self.src, tgt, self.angles, self.shape, self.pix, iters_per_stage=400)
        # global scale collapses per-tilt variation -> the recovered scale is the same for all tilts
        from alignment_models.parameter_scope import decompose
        iso = decompose("similarity", r.params)["isotropic_log_scale"].detach().numpy()
        self.assertLess(float(np.std(iso)), 1e-6, "global scope did not make scale constant")

    def test_export_consumable_residual(self):
        # The refined residual params must round-trip through serialization for export.
        from alignment_models.serialization import params_to_dict, params_from_dict
        tgt = self._truth("affine")
        r = refine(_free_config("affine"), self.src, tgt, self.angles, self.shape, self.pix, iters_per_stage=300)
        d = params_to_dict(am.get_model("affine"), r.params)
        m2, p2 = params_from_dict(d)
        self.assertEqual(m2.name, "affine")
        self.assertEqual(tuple(p2.shape), (self.n, 6))


if __name__ == "__main__":
    unittest.main()
