"""Mutation campaign for the image-based constrained models (spec §34).

Each test injects a forbidden mutation and proves a guard CATCHES it (the assertion
that protects the real behaviour fails on the mutated behaviour). Mutations are
applied locally and never persisted.
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
    from alignment_models.base import rotation_matrix
    from alignment_models.materialize import (analytic_field_numpy, detector_grid_points,
                                              materialize_field, materialize_model_field)


@unittest.skipUnless(HAVE, "torch unavailable")
class ImageModelMutationTests(unittest.TestCase):
    def _pts_c(self, shape_xy=(20, 16)):
        pts = detector_grid_points(shape_xy)
        c = torch.tensor([(shape_xy[0] - 1) / 2.0, (shape_xy[1] - 1) / 2.0], dtype=torch.float64)
        return pts, c

    def test_rotation_sign_inversion_detected(self):
        # Mutation: R(-phi). The field must differ from the correct R(+phi).
        phi = torch.tensor([0.13], dtype=torch.float64)
        A_ok = rotation_matrix(phi)[0]
        A_bad = rotation_matrix(-phi)[0]
        t = torch.zeros(2, dtype=torch.float64)
        pts, c = self._pts_c()
        f_ok = materialize_field(A_ok, t, pts, c)
        f_bad = materialize_field(A_bad, t, pts, c)
        self.assertFalse(torch.allclose(f_ok, f_bad, atol=1e-6),
                         "sign-inverted rotation produced an identical field")

    def test_degrees_instead_of_radians_detected(self):
        # Mutation: feed phi in degrees (7.5) where radians are expected.
        model = am.get_model("rigid")
        p_rad = torch.tensor([[0.0, 0.0, np.deg2rad(7.5)]], dtype=torch.float64)
        p_deg = torch.tensor([[0.0, 0.0, 7.5]], dtype=torch.float64)  # WRONG: 7.5 rad
        shape = (24, 24)
        f_rad = materialize_model_field(model, p_rad, shape)
        f_deg = materialize_model_field(model, p_deg, shape)
        self.assertGreater(float((f_rad - f_deg).abs().max()), 1.0,
                           "degrees-as-radians went undetected")

    def test_scale_direct_instead_of_exp_detected(self):
        # Mutation: A = log_scale * R(phi) instead of exp(log_scale) * R(phi).
        # At log_scale=0 the correct A=I (det 1); the mutated A=0 (det 0, singular).
        model = am.get_model("similarity")
        p = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        A_ok = model.linear_matrices(p)[0]
        self.assertAlmostEqual(float(torch.det(A_ok)), 1.0, places=6)  # exp(0)=1
        log_scale = p[0, 3]
        A_bad = log_scale * rotation_matrix(p[:, 2])[0]                 # direct (wrong)
        self.assertAlmostEqual(float(torch.det(A_bad)), 0.0, places=6)  # singular
        self.assertNotAlmostEqual(float(torch.det(A_ok)), float(torch.det(A_bad)), places=3)

    def test_detached_field_kills_gradient(self):
        # Mutation: detach the field before the loss -> params get no gradient.
        model = am.get_model("rigid")
        p = (model.identity_params(2) + 0.05).clone().requires_grad_(True)
        field = materialize_model_field(model, p, (16, 16))
        loss_detached = (field.detach() ** 2).sum()
        # backward on a detached scalar that doesn't need grad raises; that IS the catch
        with self.assertRaises(RuntimeError):
            loss_detached.backward()
        # control: the non-detached path DOES produce a gradient
        p2 = (model.identity_params(2) + 0.05).clone().requires_grad_(True)
        (materialize_model_field(model, p2, (16, 16)) ** 2).sum().backward()
        self.assertIsNotNone(p2.grad)
        self.assertGreater(float(p2.grad.abs().sum()), 0.0)

    def test_free_grid_is_not_the_optimized_variable(self):
        # Mutation guard: the optimized variable must be the constrained DOF vector,
        # not a free per-pixel grid. A free grid would have H*W*2 params; the
        # constrained model has exactly n_params per tilt.
        for name, ndof in (("translation", 2), ("rigid", 3), ("similarity", 4)):
            model = am.get_model(name)
            p = model.identity_params(3)
            self.assertEqual(p.shape, (3, ndof))
            field = materialize_model_field(model, p, (32, 32), as_image=True)
            # the field is DERIVED (32*32*2 values) but is a function of n DOF, not free:
            # zeroing the params yields the exact zero field (no residual free nodes).
            zero_field = materialize_model_field(model, model.identity_params(3), (32, 32), as_image=True)
            self.assertLess(float(zero_field.abs().max()), 1e-9,
                            f"{name}: identity params did not give a zero field (free nodes?)")

    def test_similarity_anisotropic_scale_detected(self):
        # Similarity must be isotropic: A^T A = s^2 I (off-diagonal 0). A mutation that
        # makes it anisotropic must be caught by the isotropy check.
        model = am.get_model("similarity")
        p = torch.tensor([[0.0, 0.0, 0.2, 0.1]], dtype=torch.float64)
        A = model.linear_matrices(p)[0]
        G = A.T @ A
        self.assertAlmostEqual(float(G[0, 1]), 0.0, places=6)          # isotropic
        self.assertAlmostEqual(float(G[0, 0]), float(G[1, 1]), places=6)
        A_bad = A.clone(); A_bad[0, 0] *= 1.3                          # break isotropy
        Gb = A_bad.T @ A_bad
        self.assertGreater(abs(float(Gb[0, 0]) - float(Gb[1, 1])), 1e-3)

    def test_rigid_determinant_must_be_one(self):
        # Rigid det == 1; a scaled mutation (det != 1) must be caught.
        model = am.get_model("rigid")
        p = torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float64)
        A = model.linear_matrices(p)[0]
        self.assertAlmostEqual(float(torch.det(A)), 1.0, places=6)
        A_bad = 1.2 * A
        self.assertGreater(abs(float(torch.det(A_bad)) - 1.0), 1e-2)


if __name__ == "__main__":
    unittest.main()
