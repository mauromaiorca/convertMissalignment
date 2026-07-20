"""High-volume randomized DOF + nestedness validation (>=1000 sets per model)."""
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

N = 1000  # randomized parameter sets per model


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class HighVolumeDOFTests(unittest.TestCase):
    """For N>=1000 random params per model, the DOF invariants must hold for ALL."""

    def _params(self, model_name, scale, seed):
        torch.manual_seed(seed)
        m = am.get_model(model_name)
        return m, scale * torch.randn((N, m.n_params), dtype=torch.float64)

    def test_translation_identity_1000(self):
        m, p = self._params("translation", 5.0, 1)
        A = m.matrices_numpy(p)
        self.assertTrue(np.allclose(A, np.eye(2), atol=1e-12))

    def test_rigid_orthonormal_1000(self):
        for seed, scale in [(1, 0.5), (2, 3.0), (3, 10.0)]:  # include large angles
            m, p = self._params("rigid", scale, seed)
            A = m.matrices_numpy(p)
            gram = np.einsum("nij,nik->njk", A, A)
            self.assertTrue(np.allclose(gram, np.eye(2), atol=1e-9))
            self.assertTrue(np.allclose(np.linalg.det(A), 1.0, atol=1e-9))

    def test_similarity_isotropic_1000(self):
        for seed, scale in [(1, 0.5), (2, 2.0)]:
            m, p = self._params("similarity", scale, seed)
            A = m.matrices_numpy(p)
            gram = np.einsum("nij,nik->njk", A, A)
            s2 = gram[:, 0, 0]
            self.assertTrue(np.all(s2 > 0))
            # off-diagonal zero, diagonal equal -> isotropic, no shear
            self.assertTrue(np.allclose(gram[:, 0, 1], 0.0, atol=1e-9))
            self.assertTrue(np.allclose(gram[:, 0, 0], gram[:, 1, 1], atol=1e-9))

    def test_affine_positive_determinant_1000(self):
        for seed, scale in [(1, 0.5), (2, 2.0), (3, 5.0)]:  # extreme params
            m, p = self._params("affine", scale, seed)
            det = np.linalg.det(m.matrices_numpy(p))
            self.assertTrue(np.all(det > 0), f"affine det not all positive (min={det.min():.3e})")
            self.assertTrue(np.all(np.isfinite(det)))


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class HighVolumeNestednessTests(unittest.TestCase):
    def test_nestedness_1000_each_step(self):
        chain = am.NESTING_ORDER
        for i in range(len(chain) - 1):
            simpler, larger = chain[i], chain[i + 1]
            torch.manual_seed(100 + i)
            ms = am.get_model(simpler)
            ml = am.get_model(larger)
            p_s = 0.7 * torch.randn((N, ms.n_params), dtype=torch.float64)
            p_l = am.embed_params(p_s, simpler, larger)
            self.assertTrue(np.allclose(ms.matrices_numpy(p_s), ml.matrices_numpy(p_l), atol=1e-12),
                            f"{simpler} not nested in {larger} over {N} sets")
            self.assertTrue(np.allclose(ms.translations_numpy(p_s), ml.translations_numpy(p_l), atol=1e-12))


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class BoundaryParameterTests(unittest.TestCase):
    """Near-boundary but valid parameters stay well-conditioned."""

    def test_affine_large_but_valid(self):
        m = am.get_model("affine")
        # large rotation, strong anisotropy, strong shear -- still det>0, invertible
        p = torch.tensor([
            [0, 0, np.pi, 1.5, -1.5, 2.0],
            [0, 0, -np.pi / 2, -1.5, 1.5, -2.0],
            [0, 0, 0.0, 3.0, 3.0, 0.0],
        ], dtype=torch.float64)
        A = m.matrices_numpy(p)
        det = np.linalg.det(A)
        self.assertTrue(np.all(det > 0))
        for a in A:
            np.linalg.inv(a)  # must be invertible (raises otherwise)

    def test_similarity_extreme_scale(self):
        m = am.get_model("similarity")
        p = torch.tensor([[0, 0, 0.0, 5.0], [0, 0, 0.0, -5.0]], dtype=torch.float64)  # scale e^±5
        A = m.matrices_numpy(p)
        gram = np.einsum("nij,nik->njk", A, A)
        self.assertTrue(np.allclose(gram[:, 0, 1], 0.0, atol=1e-8))  # still no shear


if __name__ == "__main__":
    unittest.main()
