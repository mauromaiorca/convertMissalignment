"""The cluster constrained-integration dispatch module wires the supported modes to
the real optimizer. Exercised with the LOCAL grid_sample reference backend (NOT the
production projector, which is cluster-only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cluster_integration" / "missalignment_patch"))

try:
    import torch  # noqa: F401
    import alignment_models as am
    import constrained_integration as CI
    from alignment_models.materialize import materialize_model_field
    from alignment_models.optimize_constrained_2d import (
        ReconstructionSettings, grid_sample_image_scorer)
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "torch/integration unavailable")
class ClusterIntegrationDispatchTests(unittest.TestCase):
    def test_supported_modes_and_warm_start(self):
        self.assertEqual(set(CI.SUPPORTED_ALIGNMENTS), {"translation", "rigid", "similarity"})
        # warm start translation -> rigid copies tx,ty and sets phi=0
        tp = am.get_model("translation").identity_params(3) + torch.tensor([1.0, -2.0], dtype=torch.float64)
        rp = CI.warm_start("translation", tp, "rigid")
        self.assertTrue(torch.allclose(rp[:, :2], tp[:, :2]))
        self.assertTrue(torch.allclose(rp[:, 2], torch.zeros(3, dtype=torch.float64)))

    def test_rejects_unsupported_mode(self):
        with self.assertRaises(ValueError):
            CI.run_constrained_iteration(
                alignment_mode="affine2d", reconstruct_and_score=lambda *a, **k: None,
                initial_parameters=[[0, 0]], tilt_angles=[0],
                reconstruction_settings=ReconstructionSettings(), device="cpu")

    def test_dispatch_runs_with_reference_backend(self):
        # the dispatcher must call the real optimizer; here the projector hook is the
        # local grid_sample reference (NOT production), proving the wiring works.
        ref = torch.tensor(np.random.RandomState(1).rand(40, 40), dtype=torch.float64)
        model = am.get_model("rigid")
        n = 3
        true_p = model.identity_params(n) + torch.tensor([0.5, -0.4, 0.02], dtype=torch.float64)
        field = materialize_model_field(model, true_p, (40, 40), as_image=True).detach()
        import torch.nn.functional as F
        iy, ix = torch.meshgrid(torch.arange(40, dtype=torch.float64),
                                torch.arange(40, dtype=torch.float64), indexing="ij")
        obs = []
        for i in range(n):
            gx = 2 * (ix + field[i, :, :, 0]) / 39 - 1
            gy = 2 * (iy + field[i, :, :, 1]) / 39 - 1
            obs.append(F.grid_sample(ref[None, None], torch.stack([gx, gy], -1)[None],
                                     align_corners=True, padding_mode="border")[0, 0])
        scorer = grid_sample_image_scorer(ref, torch.stack(obs))
        res = CI.run_constrained_iteration(
            alignment_mode="rigid", reconstruct_and_score=scorer,
            initial_parameters=model.identity_params(n), tilt_angles=np.linspace(-20, 20, n),
            reconstruction_settings=ReconstructionSettings(shape_xy=(40, 40)),
            optimizer_settings=am.OptimizerSettings(steps=60, lr=0.02), device=None)
        self.assertEqual(res.model, "rigid")
        self.assertEqual(res.n_tilts, n)
        self.assertLess(res.image_loss_history[-1], res.image_loss_history[0])


if __name__ == "__main__":
    unittest.main()
