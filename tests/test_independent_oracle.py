"""Phase 3 - independent analytical oracle for the centred-affine mathematics.

The oracle re-derives every map from first principles and does NOT call the
production helper it checks on the prediction side, so a logic error in
``imod_affine`` cannot be hidden by using the same function on both sides.
The IMOD centre convention used here, ``(n-1)/2``, was established empirically
against real ``newstack`` (see ``test_imod_center_convention``).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import imod_affine as prod


def c(shape):
    return (np.asarray(shape, float) - 1.0) / 2.0


def oracle_forward(p, A, d, si, so):
    return (np.asarray(p, float) - c(si)) @ np.asarray(A).T + np.asarray(d) + c(so)


def oracle_inverse(q, A, d, si, so):
    inv = np.linalg.inv(np.asarray(A))
    return (np.asarray(q, float) - c(so) - np.asarray(d)) @ inv.T + c(si)


def oracle_inv_physical(A, d, p_raw, p_ali):
    inv = np.linalg.inv(np.asarray(A))
    return (p_raw / p_ali) * inv, -p_raw * (inv @ np.asarray(d))


def oracle_homogeneous(A, d, si, so):
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = c(so) + np.asarray(d) - np.asarray(A) @ c(si)
    return M


def oracle_compose(A1, d1, A2, d2, si, sm, so):
    H = oracle_homogeneous(A2, d2, sm, so) @ oracle_homogeneous(A1, d1, si, sm)
    A = H[:2, :2]
    d = H[:2, 2] - c(so) + A @ c(si)
    return A, d


class IndependentOracleTests(unittest.TestCase):
    def setUp(self):
        a = np.deg2rad(7.0)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        self.A = r @ np.array([[1.04, 0.03], [-0.02, 0.97]])
        self.d = np.array([6.2, -4.1])
        self.si = (257, 193)
        self.so = (224, 176)
        rng = np.random.default_rng(0)
        self.pts = rng.uniform([0, 0], self.si, size=(40, 2))

    def test_forward_matches_oracle(self):
        prod_out = prod.forward_points_pixels(self.pts, self.A, self.d, self.si, self.so)
        orc_out = oracle_forward(self.pts, self.A, self.d, self.si, self.so)
        self.assertLess(np.max(np.abs(prod_out - orc_out)), 1e-9)

    def test_inverse_matches_oracle(self):
        q = oracle_forward(self.pts, self.A, self.d, self.si, self.so)
        prod_in = prod.inverse_points_pixels(q, self.A, self.d, self.si, self.so)
        orc_in = oracle_inverse(q, self.A, self.d, self.si, self.so)
        self.assertLess(np.max(np.abs(prod_in - orc_in)), 1e-9)
        self.assertLess(np.max(np.abs(prod_in - self.pts)), 1e-9)

    def test_inverse_physical_matches_oracle(self):
        for p_raw, p_ali in [(10.0, 10.0), (2.0, 4.0), (5.0, 2.5)]:
            Bp, bp = prod.inverse_physical_map(self.A, self.d, p_raw, p_ali)
            Bo, bo = oracle_inv_physical(self.A, self.d, p_raw, p_ali)
            self.assertLess(np.max(np.abs(Bp - Bo)), 1e-12)
            self.assertLess(np.max(np.abs(bp - bo)), 1e-9)

    def test_homogeneous_matches_oracle(self):
        Hp = prod.xf_to_homogeneous(self.A, self.d, self.si, self.so)
        Ho = oracle_homogeneous(self.A, self.d, self.si, self.so)
        self.assertLess(np.max(np.abs(Hp - Ho)), 1e-9)

    def test_compose_matches_oracle_and_double_application(self):
        A2 = np.array([[1.0, 0.004], [-0.003, 1.0]])
        d2 = np.array([1.1, -0.7])
        sm = (240, 184)
        Ap, dp = prod.compose_xf(self.A, self.d, A2, d2, self.si, sm, self.so)
        Ao, do = oracle_compose(self.A, self.d, A2, d2, self.si, sm, self.so)
        self.assertLess(np.max(np.abs(Ap - Ao)), 1e-9)
        self.assertLess(np.max(np.abs(dp - do)), 1e-9)
        # And composition equals applying the two maps in sequence.
        seq = oracle_forward(oracle_forward(self.pts, self.A, self.d, self.si, sm),
                             A2, d2, sm, self.so)
        comp = oracle_forward(self.pts, Ap, dp, self.si, self.so)
        self.assertLess(np.max(np.abs(seq - comp)), 1e-8)

    def test_physical_composition_independent_pixels(self):
        """ali_identity composition in physical Å, derived independently."""
        p_raw, p_ali, p_fin = 2.0, 4.0, 2.5
        A0, d0 = self.A, self.d
        Ar = np.array([[0.999, 0.002], [-0.001, 1.001]])
        dr = np.array([0.75, -0.4])
        # production-style physical maps
        M0 = (p_ali / p_raw) * A0
        t0 = p_ali * d0
        Mr = Ar
        tr = p_ali * dr  # residual expressed in ali Å
        Mf = Mr @ M0
        tf = Mr @ t0 + tr
        # independent direct application in Å
        raw_c = np.array([[10.0, -7.0], [0.0, 0.0], [-25.0, 12.0]])
        direct = (raw_c * p_raw) @ M0.T + t0
        direct = direct @ Mr.T + tr
        viaf = (raw_c * p_raw) @ Mf.T + tf
        self.assertLess(np.max(np.abs(direct - viaf)), 1e-9)


if __name__ == "__main__":
    unittest.main()
