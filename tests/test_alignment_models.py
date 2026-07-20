"""Unit, degrees-of-freedom, nestedness, determinant, serialization tests for
the constrained alignment models."""
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
    from alignment_models.serialization import params_from_dict, params_to_dict


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class ModelBasicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n = 7
        self.shapes = [(256, 192), (257, 193)]
        self.pix = [10.0, 12.5]

    def _random_params(self, name):
        m = am.get_model(name)
        p = m.identity_params(self.n).clone()
        # fill with small but non-trivial values
        p = p + 0.1 * torch.randn_like(p)
        return m, p

    def test_identity_params_give_identity(self):
        for name in am.NESTING_ORDER:
            m = am.get_model(name)
            p = m.identity_params(self.n)
            self.assertTrue(np.allclose(m.matrices_numpy(p), np.eye(2)))
            self.assertTrue(np.allclose(m.translations_numpy(p), 0.0))
            self.assertTrue(np.allclose(m.determinants(p), 1.0))

    def test_param_counts(self):
        self.assertEqual(am.get_model("translation").n_params, 2)
        self.assertEqual(am.get_model("rigid").n_params, 3)
        self.assertEqual(am.get_model("similarity").n_params, 4)
        self.assertEqual(am.get_model("affine").n_params, 6)

    def test_forward_inverse_homogeneous(self):
        for name in am.NESTING_ORDER:
            m, p = self._random_params(name)
            for shape, pix in zip(self.shapes, self.pix):
                c = cf.physical_center_xy(shape, pix)
                H = m.homogeneous_physical(p, np.tile(c, (self.n, 1))).detach().numpy()
                for h in H:
                    Hinv = cf.invert_homogeneous(h)
                    self.assertTrue(np.allclose(Hinv @ h, np.eye(3), atol=1e-9))

    def test_determinant_positive(self):
        for name in am.NESTING_ORDER:
            m, p = self._random_params(name)
            self.assertTrue(np.all(m.determinants(p) > 0))

    def test_serialization_roundtrip(self):
        for name in am.NESTING_ORDER:
            m, p = self._random_params(name)
            data = params_to_dict(m, p)
            m2, p2 = params_from_dict(data)
            self.assertEqual(m2.name, m.name)
            self.assertTrue(np.allclose(m.matrices_numpy(p), m2.matrices_numpy(p2)))
            self.assertTrue(np.allclose(m.translations_numpy(p), m2.translations_numpy(p2)))


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class DegreesOfFreedomExclusionTests(unittest.TestCase):
    """Each model must respect exactly its allowed degrees of freedom."""

    N = 11

    def _fuzz(self, name):
        m = am.get_model(name)
        # large random parameters to expose any leakage of forbidden DOF
        p = 0.7 * torch.randn((self.N, m.n_params), dtype=torch.float64)
        return m, p

    def test_translation_cannot_rotate_scale_shear(self):
        m, p = self._fuzz("translation")
        A = m.matrices_numpy(p)
        for a in A:
            self.assertTrue(np.allclose(a, np.eye(2), atol=1e-12))

    def test_rigid_is_orthonormal_no_scale_no_shear(self):
        m, p = self._fuzz("rigid")
        A = m.matrices_numpy(p)
        for a in A:
            self.assertTrue(np.allclose(a.T @ a, np.eye(2), atol=1e-10))  # A^T A = I
            self.assertAlmostEqual(np.linalg.det(a), 1.0, places=10)       # det = +1

    def test_similarity_is_isotropic_no_shear(self):
        m, p = self._fuzz("similarity")
        A = m.matrices_numpy(p)
        for a in A:
            gram = a.T @ a
            s2 = gram[0, 0]
            self.assertGreater(s2, 0)
            # A^T A = s^2 I : equal diagonal, zero off-diagonal -> no shear, isotropic
            self.assertTrue(np.allclose(gram, s2 * np.eye(2), atol=1e-9))

    def test_affine_positive_determinant_invertible(self):
        m, p = self._fuzz("affine")
        # include extreme values
        p[0] = torch.tensor([100.0, -50.0, 3.0, 2.0, -2.0, 5.0], dtype=torch.float64)
        A = m.matrices_numpy(p)
        for a in A:
            self.assertGreater(np.linalg.det(a), 0.0)
            self.assertGreater(abs(np.linalg.det(a)), 1e-12)  # invertible


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class NestednessTests(unittest.TestCase):
    """translation ⊂ rigid ⊂ similarity ⊂ affine : exact reduction."""

    N = 5

    def test_zeroing_extra_params_reduces_model(self):
        torch.manual_seed(1)
        chain = am.NESTING_ORDER
        for i in range(len(chain) - 1):
            simpler, larger = chain[i], chain[i + 1]
            ms = am.get_model(simpler)
            ml = am.get_model(larger)
            p_s = ms.identity_params(self.N) + 0.2 * torch.randn((self.N, ms.n_params), dtype=torch.float64)
            p_l = am.embed_params(p_s, simpler, larger)
            self.assertTrue(
                np.allclose(ms.matrices_numpy(p_s), ml.matrices_numpy(p_l), atol=1e-12),
                f"{simpler} not exactly nested in {larger}",
            )
            self.assertTrue(
                np.allclose(ms.translations_numpy(p_s), ml.translations_numpy(p_l), atol=1e-12)
            )

    def test_full_chain_translation_in_affine(self):
        ms = am.get_model("translation")
        p = torch.tensor([[3.0, -4.0], [1.0, 2.0]], dtype=torch.float64)
        p_aff = am.embed_params(p, "translation", "affine")
        self.assertTrue(np.allclose(am.get_model("affine").matrices_numpy(p_aff), np.eye(2)))


if __name__ == "__main__":
    unittest.main()
