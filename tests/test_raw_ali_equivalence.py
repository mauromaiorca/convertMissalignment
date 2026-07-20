"""Raw-vs-ali equivalence and composition-order guard.

For the same Hfinal = DeltaH @ H0, the raw composed export (raw->final) and the
ali residual export (ali->final, applied after the original raw->ali xf) must
produce identical final pixel coordinates. Tested across equal/unequal
dimensions and pixel sizes (crop/pad/bin). Also proves the composition order is
DeltaH @ H0, not H0 @ DeltaH."""
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

from imod_affine import forward_points_pixels  # audited (n-1)/2 forward map

if HAVE_TORCH:
    import alignment_models as am
    from alignment_models import composition as comp
    from alignment_models import coordinate_frames as cf
    from alignment_models.serialization import homogeneous_to_xf_rows


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class RawAliEquivalenceTests(unittest.TestCase):
    # (raw_shape, ali_shape, final_shape, p_raw, p_ali, p_final, label)
    CASES = [
        ((256, 192), (256, 192), (256, 192), 10.0, 10.0, 10.0, "equal"),
        ((256, 192), (300, 240), (280, 210), 10.0, 10.0, 10.0, "unequal_dims_crop_pad"),
        ((512, 384), (256, 192), (256, 192), 5.0, 10.0, 10.0, "binning_unequal_pixels"),
        ((257, 193), (257, 193), (257, 193), 8.0, 8.0, 8.0, "odd_dims"),
    ]

    def _setup_transforms(self, n=4):
        torch.manual_seed(5)
        # original raw->ali IMOD xf (full affine), per tilt
        A0 = np.stack([_rot(np.deg2rad(2.0 + i)) @ np.array([[1.02, 0.03], [-0.01, 0.98]]) for i in range(n)])
        d0 = np.stack([np.array([3.0 + i, -2.0 - i]) for i in range(n)])
        # residual via affine model
        model = am.get_model("affine")
        params = model.identity_params(n) + 0.05 * torch.randn((n, 6), dtype=torch.float64)
        return A0, d0, model, params

    def test_equivalence_across_geometry(self):
        n = 4
        A0, d0, model, params = self._setup_transforms(n)
        raw_pts = np.array([[10, 12], [200, 40], [120, 150], [40, 170], [250, 90]], float)
        for raw_shape, ali_shape, final_shape, p_raw, p_ali, p_final, label in self.CASES:
            H0 = comp.initial_homogeneous_per_tilt("ali_identity", A0, d0, raw_shape, ali_shape, p_raw, p_ali)
            dH = comp.residual_homogeneous_per_tilt(model, params, ali_shape, p_ali)
            Hfinal = comp.compose_final_per_tilt(H0, dH)

            comp_A, comp_d = homogeneous_to_xf_rows(Hfinal, raw_shape, final_shape, p_raw, p_final)
            res_A, res_d = homogeneous_to_xf_rows(dH, ali_shape, final_shape, p_ali, p_final)

            for i in range(n):
                # Path A: raw -> final via composed xf
                finalA = forward_points_pixels(raw_pts, comp_A[i], comp_d[i], raw_shape, final_shape)
                # Path B: raw -> ali via original xf, then ali -> final via residual xf
                ali_pts = forward_points_pixels(raw_pts, A0[i], d0[i], raw_shape, ali_shape)
                finalB = forward_points_pixels(ali_pts, res_A[i], res_d[i], ali_shape, final_shape)
                err = np.max(np.abs(finalA - finalB))
                self.assertLess(err, 1e-6, f"[{label}] tilt {i} raw/ali mismatch {err:.3e} px")

    def test_composition_order_is_deltaH_at_H0(self):
        n = 3
        A0, d0, model, params = self._setup_transforms(n)
        shape = (256, 192); p = 10.0
        H0 = comp.initial_homogeneous_per_tilt("ali_identity", A0, d0, shape, shape, p, p)
        dH = comp.residual_homogeneous_per_tilt(model, params, shape, p)
        Hfinal = comp.compose_final_per_tilt(H0, dH)
        for i in range(n):
            self.assertTrue(np.allclose(Hfinal[i], dH[i] @ H0[i], atol=1e-12))
            # wrong order must differ (matrices do not commute here)
            wrong = H0[i] @ dH[i]
            self.assertGreater(np.max(np.abs(Hfinal[i] - wrong)), 1e-6,
                               "test transforms commute; choose non-commuting ones")


if __name__ == "__main__":
    unittest.main()
