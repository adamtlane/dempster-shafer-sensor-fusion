"""
Validation suite.  Every expected value below was derived by hand before the
code was run (the arithmetic is written out in README.md, section
"Hand-derived test cases"), and the tests compare those values against the
observed output of the implementation.

Run from the Code folder with:   python -m unittest -v
"""

import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsfusion.bayes import naive_bayes_fuse  # noqa: E402
from dsfusion.core import (  # noqa: E402
    TotalConflictError, belief, combine_all, dempster_combine, discount,
    full_frame, make_mass, pignistic, plausibility, subset, total_conflict,
    vacuous, validate,
)
from dsfusion.experiment import accuracy_credit, brier, run_cell, cell_rng  # noqa: E402

PLACES = 6  # agreement to six decimal places


class T1_ZadehHighConflict(unittest.TestCase):
    """Zadeh (1986): Theta = {Meningitis, Concussion, Tumor}, indices 0, 1, 2."""
    n = 3
    M, C, T = subset(0), subset(1), subset(2)

    def setUp(self):
        self.m1 = make_mass([(self.M, 0.99), (self.T, 0.01)])
        self.m2 = make_mass([(self.C, 0.99), (self.T, 0.01)])

    def test_dempster_returns_certain_tumor(self):
        m, K = dempster_combine(self.m1, self.m2)
        # K = 1 - 0.01*0.01 = 0.9999 ; m({T}) = 0.0001 / 0.0001 = 1
        self.assertAlmostEqual(K, 0.9999, places=PLACES)
        self.assertEqual(set(m), {self.T})
        self.assertAlmostEqual(m[self.T], 1.0, places=PLACES)

    def test_naive_bayes_shares_the_pathology(self):
        # Likelihoods (0.99, 0, 0.01) and (0, 0.99, 0.01): product (0, 0, 1e-4)
        p = naive_bayes_fuse([self.m1, self.m2], self.n)
        for got, exp in zip(p, [0.0, 0.0, 1.0]):
            self.assertAlmostEqual(got, exp, places=PLACES)

    def test_discounting_repairs_the_pathology(self):
        # alpha = 0.1 on both sources (hand trace in README):
        # K = 0.809919, m(M) = m(C) = 0.0891/0.190081, m(T) = 0.001881/0.190081,
        # m(Theta) = 0.01/0.190081
        d1 = discount(self.m1, 0.1, self.n)
        d2 = discount(self.m2, 0.1, self.n)
        m, K = dempster_combine(d1, d2)
        self.assertAlmostEqual(K, 0.809919, places=PLACES)
        self.assertAlmostEqual(m[self.M], 0.0891 / 0.190081, places=PLACES)
        self.assertAlmostEqual(m[self.C], 0.0891 / 0.190081, places=PLACES)
        self.assertAlmostEqual(m[self.T], 0.001881 / 0.190081, places=PLACES)
        self.assertAlmostEqual(m[full_frame(3)], 0.01 / 0.190081, places=PLACES)

    def test_total_conflict_is_refused(self):
        a = make_mass([(self.M, 1.0)])
        b = make_mass([(self.C, 1.0)])
        with self.assertRaises(TotalConflictError):
            dempster_combine(a, b)


class T2_AgreeingSensors(unittest.TestCase):
    """Three sensors each report m({F}) = 0.8, m(Theta) = 0.2 on the 4-class frame."""
    n = 4

    def setUp(self):
        self.reports = [make_mass([(subset(0), 0.8), (full_frame(4), 0.2)])] * 3

    def test_dempster_mass(self):
        m, ks = combine_all(self.reports)
        # m(Theta) = 0.2^3 = 0.008, m({F}) = 0.992, no conflict
        self.assertAlmostEqual(m[full_frame(4)], 0.008, places=PLACES)
        self.assertAlmostEqual(m[subset(0)], 0.992, places=PLACES)
        self.assertAlmostEqual(total_conflict(ks), 0.0, places=PLACES)

    def test_pignistic_and_bayes_agree_on_decision(self):
        m, _ = combine_all(self.reports)
        betp = pignistic(m, self.n)
        post = naive_bayes_fuse(self.reports, self.n)
        # BetP(F) = 0.992 + 0.008/4 = 0.994 ; Bayes(F) = 0.85^3 / (0.85^3 + 3 * 0.05^3)
        self.assertAlmostEqual(betp[0], 0.994, places=PLACES)
        self.assertAlmostEqual(post[0], 0.614125 / 0.6145, places=PLACES)
        self.assertEqual(int(np.argmax(betp)), 0)
        self.assertEqual(int(np.argmax(post)), 0)


class T3_IgnoranceAndDiscountLimits(unittest.TestCase):
    n = 4

    def test_vacuous_is_identity_element(self):
        m = make_mass([(subset(0, 1), 0.6), (subset(2), 0.3), (full_frame(4), 0.1)])
        out, K = dempster_combine(m, vacuous(self.n))
        self.assertAlmostEqual(K, 0.0)
        for A in m:
            self.assertAlmostEqual(out[A], m[A], places=PLACES)

    def test_vacuous_pignistic_is_uniform(self):
        self.assertEqual(pignistic(vacuous(self.n), self.n), [0.25] * 4)

    def test_discount_endpoints(self):
        m = make_mass([(subset(1), 0.7), (full_frame(4), 0.3)])
        self.assertEqual(discount(m, 0.0, self.n), m)
        self.assertEqual(discount(m, 1.0, self.n), {full_frame(4): 1.0})


class T4_AlgebraicProperties(unittest.TestCase):
    """Commutativity, associativity, normalisation, Bel <= BetP <= Pl."""
    n = 4

    def random_mass(self, rng):
        focal = rng.choice(np.arange(1, 16), size=rng.integers(1, 5), replace=False)
        w = rng.dirichlet(np.ones(len(focal)))
        return make_mass(zip(map(int, focal), map(float, w)))

    def test_properties_on_random_masses(self):
        rng = np.random.default_rng(0)
        checked = 0
        while checked < 200:
            a, b, c = (discount(self.random_mass(rng), 0.05, self.n) for _ in range(3))
            ab, _ = dempster_combine(a, b)
            ba, _ = dempster_combine(b, a)
            for A in set(ab) | set(ba):
                self.assertAlmostEqual(ab.get(A, 0), ba.get(A, 0), places=9)
            left, _ = dempster_combine(ab, c)
            bc, _ = dempster_combine(b, c)
            right, _ = dempster_combine(a, bc)
            for A in set(left) | set(right):
                self.assertAlmostEqual(left.get(A, 0), right.get(A, 0), places=9)
            validate(left)
            betp = pignistic(left, self.n)
            self.assertAlmostEqual(sum(betp), 1.0, places=9)
            for i in range(self.n):
                s = subset(i)
                self.assertLessEqual(belief(left, s), betp[i] + 1e-12)
                self.assertLessEqual(betp[i], plausibility(left, s) + 1e-12)
            checked += 1


class T5_HandTracedThreeSensorCase(unittest.TestCase):
    """
    Frame {F, T, H, U}.  Sensor 1 m({F}) = 0.8 discounted by 0.25, sensor 2
    m({F,T}) = 0.7 trusted, sensor 3 m({H}) = 0.5 trusted.  Full trace in README.
    """
    n = 4
    TH = full_frame(4)

    def setUp(self):
        self.r1 = make_mass([(subset(0), 0.8), (self.TH, 0.2)])
        self.r2 = make_mass([(subset(0, 1), 0.7), (self.TH, 0.3)])
        self.r3 = make_mass([(subset(2), 0.5), (self.TH, 0.5)])
        self.alphas = [0.25, 0.0, 0.0]

    def test_dempster_trace(self):
        d = [discount(r, a, self.n) for r, a in zip([self.r1, self.r2, self.r3], self.alphas)]
        m12, k1 = dempster_combine(d[0], d[1])
        self.assertAlmostEqual(k1, 0.0, places=PLACES)
        self.assertAlmostEqual(m12[subset(0)], 0.60, places=PLACES)
        self.assertAlmostEqual(m12[subset(0, 1)], 0.28, places=PLACES)
        self.assertAlmostEqual(m12[self.TH], 0.12, places=PLACES)
        m, k2 = dempster_combine(m12, d[2])
        self.assertAlmostEqual(k2, 0.44, places=PLACES)
        self.assertAlmostEqual(m[subset(0)], 0.30 / 0.56, places=PLACES)
        self.assertAlmostEqual(m[subset(0, 1)], 0.14 / 0.56, places=PLACES)
        self.assertAlmostEqual(m[subset(2)], 0.06 / 0.56, places=PLACES)
        self.assertAlmostEqual(m[self.TH], 0.06 / 0.56, places=PLACES)
        self.assertAlmostEqual(belief(m, subset(0)), 0.30 / 0.56, places=PLACES)
        self.assertAlmostEqual(plausibility(m, subset(0)), 0.50 / 0.56, places=PLACES)
        betp = pignistic(m, self.n)
        for got, exp in zip(betp, [0.6875, 0.085 / 0.56, 0.075 / 0.56, 0.015 / 0.56]):
            self.assertAlmostEqual(got, exp, places=PLACES)

    def test_bayes_variants(self):
        reports = [self.r1, self.r2, self.r3]
        plain = naive_bayes_fuse(reports, self.n)
        z = 0.045156250 + 0.002656250 + 0.002343750 + 0.000468750
        for got, exp in zip(plain, [0.045156250 / z, 0.002656250 / z, 0.002343750 / z, 0.000468750 / z]):
            self.assertAlmostEqual(got, exp, places=PLACES)
        disc = naive_bayes_fuse(reports, self.n, alphas=self.alphas)
        z = 0.0371875 + 0.0053125 + 0.0046875 + 0.0009375
        for got, exp in zip(disc, [0.0371875 / z, 0.0053125 / z, 0.0046875 / z, 0.0009375 / z]):
            self.assertAlmostEqual(got, exp, places=PLACES)


class T6_MetricsAndSimulator(unittest.TestCase):

    def test_metrics(self):
        self.assertEqual(accuracy_credit([0.7, 0.1, 0.1, 0.1], 0), 1.0)
        self.assertEqual(accuracy_credit([0.4, 0.4, 0.1, 0.1], 1), 0.5)
        self.assertAlmostEqual(brier([1, 0, 0, 0], 0), 0.0)
        self.assertAlmostEqual(brier([0, 1, 0, 0], 0), 2.0)
        self.assertAlmostEqual(brier([0.25] * 4, 0), 0.75)

    def test_reproducible_with_fixed_seed(self):
        a = run_cell(0.3, 0.2, 200, cell_rng(7, 2, 2))
        b = run_cell(0.3, 0.2, 200, cell_rng(7, 2, 2))
        np.testing.assert_array_equal(a.brier["DS"], b.brier["DS"])

    def test_clean_sensors_are_accurate(self):
        # With no unreliability and no decoys every report points at the
        # truth, so both methods must name the true class every time.
        r = run_cell(0.0, 0.0, 300, cell_rng(1, 0, 0))
        self.assertEqual(r.acc["DS"].mean(), 1.0)
        self.assertEqual(r.acc["Bayes"].mean(), 1.0)
        self.assertAlmostEqual(float(r.K.max()), 0.0, places=12)


class T7_ValidationTable(unittest.TestCase):
    """Every hand derived row shown in the notebook's expected versus observed table."""

    def test_all_rows_agree(self):
        from dsfusion.validation import validation_rows
        rows = validation_rows()
        self.assertGreaterEqual(len(rows), 30)
        for case, quantity, expected, observed in rows:
            with self.subTest(case=case, quantity=quantity):
                self.assertAlmostEqual(observed, expected, places=PLACES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
