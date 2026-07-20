"""Canonical result contract (§24), optimizer telemetry (§21), safety bounds (§22).
The optimizer emits constrained_alignment.json/.pt + stage_history + run_manifest;
the reader validates schema/model/tilt-count and refuses incomplete results."""
from __future__ import annotations

import json
import sys
import tempfile
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
    from alignment_models import result_contract as RC
    from alignment_models.constraints import GaugeConfig
    from alignment_models.materialize import materialize_model_field
    from alignment_models.optimize_constrained_2d import (
        OptimizerSettings, ReconstructionSettings, SafetyBounds,
        grid_sample_image_scorer, optimize_constrained_2d)

NO_GAUGE = (lambda: GaugeConfig(anchor_tilt="none", zero_mean_rotation=False,
                                zero_mean_log_scale=False, zero_mean_shear=False)) if HAVE else (lambda: None)
ZERO_REG = (lambda: am.regularization.RegularizationConfig(
    translation_prior=0.0, rotation_prior=0.0, scale_prior=0.0, shear_prior=0.0,
    smoothness=0.0, curvature=0.0)) if HAVE else (lambda: None)


@unittest.skipUnless(HAVE, "torch unavailable")
class ResultContractTests(unittest.TestCase):
    def _scorer_and_obs(self, model, n, shape_xy):
        ref = torch.tensor(np.random.RandomState(0).rand(shape_xy[1], shape_xy[0]), dtype=torch.float64)
        true_p = (model.identity_params(n) + torch.tensor([0.4, -0.3] + [0.0] * (model.n_params - 2),
                                                          dtype=torch.float64))
        field = materialize_model_field(model, true_p, shape_xy, as_image=True).detach()
        import torch.nn.functional as F
        ny, nx = shape_xy[1], shape_xy[0]
        iy, ix = torch.meshgrid(torch.arange(ny, dtype=torch.float64),
                                torch.arange(nx, dtype=torch.float64), indexing="ij")
        obs = []
        for i in range(n):
            qx = ix + field[i, :, :, 0]; qy = iy + field[i, :, :, 1]
            gx = 2 * qx / (nx - 1) - 1; gy = 2 * qy / (ny - 1) - 1
            grid = torch.stack([gx, gy], -1).unsqueeze(0)
            obs.append(F.grid_sample(ref[None, None], grid, align_corners=True, padding_mode="border")[0, 0])
        return grid_sample_image_scorer(ref, torch.stack(obs))

    def test_optimizer_emits_canonical_contract(self):
        model = am.get_model("rigid")
        n = 3; shape = (40, 40)
        with tempfile.TemporaryDirectory() as td:
            res = optimize_constrained_2d(
                alignment_model=model, initial_parameters=model.identity_params(n),
                reconstruct_and_score=self._scorer_and_obs(model, n, shape),
                tilt_angles=np.linspace(-20, 20, n), gauge=NO_GAUGE(), regularization=ZERO_REG(),
                optimizer_settings=OptimizerSettings(steps=60, lr=0.02, log_every=10),
                reconstruction_settings=ReconstructionSettings(shape_xy=shape),
                telemetry_dir=Path(td) / "tele", result_dir=Path(td) / "result", seed=7)
            rd = Path(td) / "result"
            for f in RC.CANONICAL_FILES:
                self.assertTrue((rd / f).is_file(), f)
            data = json.loads((rd / "constrained_alignment.json").read_text())
            self.assertEqual(data["model"], "rigid")
            self.assertEqual(data["n_tilts"], n)
            self.assertEqual(data["completion_status"], "completed")
            self.assertEqual(data["parameter_names"], ["tx", "ty", "phi"])
            # telemetry
            tele = Path(td) / "tele"
            self.assertTrue((tele / "training_events.jsonl").is_file())
            self.assertTrue((tele / "loss_history.tsv").is_file())
            lines = (tele / "training_events.jsonl").read_text().splitlines()
            self.assertGreater(len(lines), 0)
            self.assertIn("grad_norm", json.loads(lines[0]))

    def test_reader_validates_and_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            RC.write_constrained_result(
                Path(td), model="similarity", params=[[0.1, 0.2, 0.0, 0.0]], tilt_angles=[0.0],
                param_names=("tx", "ty", "phi", "log_scale"), scopes={}, gauge={}, regularization={},
                working_raw_grid=None, working_aligned_grid=None, input_hashes={}, warp_project_hash=None,
                loss_history=[1.0, 0.1], gradient_summary={}, stage_history=[],
                software_versions={}, cuda_info=None, seed=1,
                start_time="t0", end_time="t1", completion_status="completed")
            ref = RC.read_constrained_result(Path(td), expected_model="similarity", expected_n_tilts=1)
            self.assertEqual(ref.json["model"], "similarity")
            with self.assertRaises(RC.ResultContractError):
                RC.read_constrained_result(Path(td), expected_model="rigid")
            with self.assertRaises(RC.ResultContractError):
                RC.read_constrained_result(Path(td), expected_n_tilts=5)

    def test_reader_refuses_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            RC.write_constrained_result(
                Path(td), model="rigid", params=[[0.0, 0.0, 0.0]], tilt_angles=[0.0],
                param_names=("tx", "ty", "phi"), scopes={}, gauge={}, regularization={},
                working_raw_grid=None, working_aligned_grid=None, input_hashes={}, warp_project_hash=None,
                loss_history=[], gradient_summary={}, stage_history=[], software_versions={},
                cuda_info=None, seed=1, start_time="t0", end_time="t1", completion_status="crashed")
            with self.assertRaises(RC.ResultContractError):
                RC.read_constrained_result(Path(td))  # require_completed default
            ok = RC.read_constrained_result(Path(td), require_completed=False)
            self.assertEqual(ok.json["completion_status"], "crashed")

    def test_missing_result_is_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RC.ResultContractError):
                RC.read_constrained_result(Path(td))

    def test_safety_hard_limit_raises(self):
        # A rotation hard limit far below the truth must trigger a hard failure.
        model = am.get_model("rigid")
        n = 2; shape = (24, 24)
        bounds = SafetyBounds(rotation_hard_rad=0.001, rotation_warn_rad=0.0005)
        with self.assertRaises(FloatingPointError):
            optimize_constrained_2d(
                alignment_model=model,
                initial_parameters=model.identity_params(n) + torch.tensor([0, 0, 0.2], dtype=torch.float64),
                reconstruct_and_score=self._scorer_and_obs(model, n, shape),
                gauge=NO_GAUGE(), regularization=ZERO_REG(), safety_bounds=bounds,
                optimizer_settings=OptimizerSettings(steps=50, lr=0.05),
                reconstruction_settings=ReconstructionSettings(shape_xy=shape))


if __name__ == "__main__":
    unittest.main()
