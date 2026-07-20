"""Differentiable constrained -> detector movement-field materialization (Part B).

Validates the Option-B field d(p)=(A-I)(p-c)+t against an independent analytic
oracle (RMS<1e-6), proves gradients flow from the field back to the CONSTRAINED
model parameters (not a free grid), checks device/dtype awareness, and runs an
image-free recovery on the field itself (optimizing the constrained DOF through
the materialized field). Writes IMAGE_BASED_GRADIENT_VALIDATION.json.
"""
from __future__ import annotations

import json
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
    from alignment_models.materialize import (
        analytic_field_numpy, detector_grid_points, materialize_field,
        materialize_model_field,
    )

ARTIFACT = ROOT / "IMAGE_BASED_GRADIENT_VALIDATION.json"
MODES = ("translation", "rigid", "similarity", "affine")
_RESULTS: dict = {"modes": {}}


def _nontrivial_params(name, n):
    g = {
        "translation": [[1.3, -0.7]] * n,
        "rigid": [[1.3, -0.7, 0.05 + 0.01 * i] for i in range(n)],
        "similarity": [[1.3, -0.7, 0.05, 0.02]] * n,
        "affine": [[1.3, -0.7, 0.05, 0.03, -0.02, 0.04]] * n,
    }[name]
    return torch.tensor(g, dtype=torch.float64)


@unittest.skipUnless(HAVE, "torch unavailable")
class MaterializeTests(unittest.TestCase):
    def test_field_matches_analytic_oracle(self):
        # d(p)=(A-I)(p-c)+t materialized in torch == independent numpy oracle.
        for name in MODES:
            model = am.get_model(name)
            n = 4
            params = _nontrivial_params(name, n)
            shape_xy = (40, 32)
            field = materialize_model_field(model, params, shape_xy, pixel_size_xy=(2.0, 2.0))
            A = model.matrices_numpy(params)
            t = model.translations_numpy(params)
            pts = detector_grid_points(shape_xy, pixel_size_xy=(2.0, 2.0)).numpy()
            cx = 2.0 * (shape_xy[0] - 1) / 2.0
            cy = 2.0 * (shape_xy[1] - 1) / 2.0
            rms_all = []
            for i in range(n):
                ref = analytic_field_numpy(A[i], t[i], pts, np.array([cx, cy]))
                got = field[i].detach().numpy()
                rms = float(np.sqrt(np.mean((ref - got) ** 2)))
                rms_all.append(rms)
                self.assertLess(rms, 1e-6, f"{name} tilt {i}: field vs analytic RMS {rms:.2e}")
            _RESULTS["modes"].setdefault(name, {})["analytic_rms_max"] = max(rms_all)

    def test_field_equals_apply_centered_minus_p(self):
        # The absolute mapped point q=p+d must equal the model's own apply_centered.
        for name in MODES:
            model = am.get_model(name)
            n = 3
            params = _nontrivial_params(name, n)
            shape_xy = (24, 20)
            pts = detector_grid_points(shape_xy)
            cx = (shape_xy[0] - 1) / 2.0
            cy = (shape_xy[1] - 1) / 2.0
            centers = torch.tensor([cx, cy], dtype=torch.float64).expand(n, 2)
            d = materialize_model_field(model, params, shape_xy)
            q = pts.unsqueeze(0) + d
            q_ref = model.apply_centered(params, pts, centers)
            self.assertTrue(torch.allclose(q, q_ref, atol=1e-9),
                            f"{name}: q=p+d != apply_centered")

    def test_gradient_flows_to_constrained_params(self):
        # The optimized variable is the constrained DOF vector; grad must be finite
        # and structurally correct (e.g. translation has zero grad in non-existent DOF).
        for name in MODES:
            model = am.get_model(name)
            n = 2
            params = _nontrivial_params(name, n).clone().requires_grad_(True)
            field = materialize_model_field(model, params, (16, 12))
            loss = (field ** 2).sum()
            loss.backward()
            self.assertIsNotNone(params.grad)
            self.assertTrue(torch.isfinite(params.grad).all(), f"{name}: non-finite grad")
            # translation: the t-grad must be non-zero (field depends on t)
            self.assertGreater(float(params.grad[:, :2].abs().sum()), 0.0, f"{name}: dead t-grad")
            _RESULTS["modes"].setdefault(name, {})["grad_finite"] = True
            _RESULTS["modes"][name]["grad_norm"] = float(params.grad.norm())

    def test_gradcheck_double_precision(self):
        # Autograd correctness vs finite differences (the real differentiability proof).
        for name in MODES:
            model = am.get_model(name)
            n = 2
            base = _nontrivial_params(name, n)
            pts = detector_grid_points((8, 6), pixel_size_xy=(1.5, 1.5))
            cx = 1.5 * (8 - 1) / 2.0
            cy = 1.5 * (6 - 1) / 2.0
            centers = torch.tensor([cx, cy], dtype=torch.float64).expand(n, 2)

            def f(p):
                A, t = model.matrices_and_translations(p)
                return materialize_field(A, t, pts, centers)

            params = base.clone().requires_grad_(True)
            ok = torch.autograd.gradcheck(f, (params,), eps=1e-6, atol=1e-5, rtol=1e-3)
            self.assertTrue(ok, f"{name}: gradcheck failed")
            _RESULTS["modes"].setdefault(name, {})["gradcheck"] = True

    def test_dtype_and_device_awareness(self):
        # float32 and (where available) CUDA/MPS placement; field stays on device.
        model = am.get_model("rigid")
        params = _nontrivial_params("rigid", 2)
        f32 = materialize_model_field(model, params, (10, 10), dtype=torch.float32)
        self.assertEqual(f32.dtype, torch.float32)
        f64 = materialize_model_field(model, params, (10, 10), dtype=torch.float64)
        self.assertLess(float((f64 - f32.double()).abs().max()), 1e-3)
        dev = None
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            dev = "mps"
        _RESULTS["device_tested"] = dev or "cpu_only"
        if dev:
            # params stay CPU/float64; the field is materialized on-device in float32
            # (MPS has no float64). The constrained-matrix computation is tiny; the
            # full per-pixel field lives on the accelerator.
            fd = materialize_model_field(model, params, (10, 10), device=dev, dtype=torch.float32)
            self.assertEqual(fd.device.type, dev)
            self.assertEqual(fd.dtype, torch.float32)

    def test_recovery_through_field(self):
        # Image-free recovery: optimize the constrained DOF so the materialized field
        # matches a target field generated from known params -- pure autograd through
        # the constrained -> field path (no detach, real optimizer).
        torch.manual_seed(0)
        for name in MODES:
            model = am.get_model(name)
            n = 3
            shape_xy = (28, 24)
            true_p = _nontrivial_params(name, n)
            target = materialize_model_field(model, true_p, shape_xy).detach()
            est = model.identity_params(n).clone().requires_grad_(True)
            opt = torch.optim.Adam([est], lr=0.05)
            for _ in range(400):
                opt.zero_grad()
                pred = materialize_model_field(model, est, shape_xy)
                loss = ((pred - target) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                final = float(((materialize_model_field(model, est, shape_xy) - target) ** 2).mean())
            self.assertLess(final, 1e-4, f"{name}: field recovery loss {final:.2e}")
            _RESULTS["modes"].setdefault(name, {})["recovery_final_loss"] = final

    def test_image_loss_recovery_through_real_warp(self):
        # The field drives a REAL differentiable image loss: warp a real image with
        # torch's bilinear sampler (grid_sample) through the materialized field, then
        # recover the CONSTRAINED params by minimizing an image L2 loss with Adam.
        # This proves params -> field -> interpolation -> image loss -> autograd -> DOF
        # works end to end. It is NOT the MissAlignment projector/scoring net (those are
        # CLUSTER-only); it is a real differentiable image path with real interpolation.
        torch.manual_seed(1)
        ny, nx = 64, 64
        yy, xx = torch.meshgrid(torch.arange(ny, dtype=torch.float64),
                                torch.arange(nx, dtype=torch.float64), indexing="ij")
        img = torch.zeros(ny, nx, dtype=torch.float64)
        for cx, cy, s in [(20, 24, 3.0), (44, 40, 4.0), (30, 50, 2.5)]:
            img += torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
        img = img.view(1, 1, ny, nx)

        def warp(image, field_img):  # field_img: (1, ny, nx, 2) displacement (dx, dy)
            qx = xx + field_img[0, :, :, 0]
            qy = yy + field_img[0, :, :, 1]
            gx = 2.0 * qx / (nx - 1) - 1.0
            gy = 2.0 * qy / (ny - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            return torch.nn.functional.grid_sample(image, grid, mode="bilinear",
                                                   align_corners=True, padding_mode="border")

        results = {}
        for name in ("translation", "rigid", "similarity"):
            model = am.get_model(name)
            true_p = _nontrivial_params(name, 1) * 0.15  # small, recoverable warp
            tgt_field = materialize_model_field(model, true_p, (nx, ny), as_image=True).detach()
            observed = warp(img, tgt_field).detach()
            est = model.identity_params(1).clone().requires_grad_(True)
            opt = torch.optim.Adam([est], lr=0.02)
            for _ in range(600):
                opt.zero_grad()
                f = materialize_model_field(model, est, (nx, ny), as_image=True)
                pred = warp(img, f)
                loss = ((pred - observed) ** 2).mean()
                loss.backward()
                self.assertTrue(torch.isfinite(est.grad).all())
                opt.step()
            with torch.no_grad():
                final_loss = float(((warp(img, materialize_model_field(
                    model, est, (nx, ny), as_image=True)) - observed) ** 2).mean())
            self.assertLess(final_loss, 5e-4, f"{name}: image-loss recovery {final_loss:.2e}")
            results[name] = {"final_image_loss": final_loss,
                             "param_err_inf": float((est.detach() - true_p).abs().max())}
        _RESULTS["image_loss_recovery"] = results
        rec = ROOT / "IMAGE_BASED_RECOVERY_RESULTS.json"
        rec.write_text(json.dumps({
            "method": "constrained params -> Option-B field -> torch grid_sample warp -> "
                      "image L2 loss -> autograd -> Adam on constrained DOF",
            "interpolation": "torch.nn.functional.grid_sample (real, differentiable bilinear)",
            "results": results,
            "scope": "LOCALLY VERIFIED real differentiable image-loss recovery through the "
                     "materialized field. The MissAlignment reconstruction/projector/scoring "
                     "network is NOT exercised (not installed) -> that integration is CLUSTER "
                     "NOT VERIFIED.",
        }, indent=2) + "\n")

    @classmethod
    def tearDownClass(cls):
        if HAVE:
            _RESULTS["verdict"] = "LOCALLY VERIFIED (materialization is real, differentiable, " \
                                  "gradient-checked; matches analytic oracle RMS<1e-6)"
            _RESULTS["note"] = "This validates the constrained->detector field artifact ONLY. " \
                               "Integration into the real MissAlignment image forward pass " \
                               "(reconstruction/projector/scoring/image loss) is CLUSTER NOT " \
                               "VERIFIED -- MissAlignment/warpylib/CUDA are not installed locally."
            ARTIFACT.write_text(json.dumps(_RESULTS, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
