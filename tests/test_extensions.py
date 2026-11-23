"""Hand-derived checks for Yager's rule and the oracle Bayesian reference.
Derivations are in Appendix A of the case study paper."""
import unittest

import numpy as np

from dsfusion.core import discount, full_frame, make_mass, pignistic, subset
from dsfusion.extensions import (oracle_bayes_fuse, point_likelihood, run_grid_ext,
                                 yager_combine_all)
from dsfusion.simulator import SENSORS


class TestYager(unittest.TestCase):
    def test_zadeh_conflict_becomes_ignorance(self):
        # Frame {M, C, T}: K = 0.9999 goes to Theta instead of forcing T.
        th = full_frame(3)
        m1 = make_mass([(subset(0), 0.99), (subset(2), 0.01)])
        m2 = make_mass([(subset(1), 0.99), (subset(2), 0.01)])
        m, K = yager_combine_all([m1, m2], 3)
        self.assertAlmostEqual(K, 0.9999, places=6)
        self.assertAlmostEqual(m[subset(2)], 0.0001, places=6)
        self.assertAlmostEqual(m[th], 0.9999, places=6)
        p = pignistic(m, 3)
        self.assertAlmostEqual(p[0], 0.3333, places=4)
        self.assertAlmostEqual(p[2], 0.3334, places=4)

    def test_hand_traced_three_sensor_case(self):
        # Case A.3: Yager keeps the 0.44 conflict on Theta.
        th = full_frame(4)
        s1 = discount({subset(0): 0.8, th: 0.2}, 0.25, 4)
        s2 = {subset(0, 1): 0.7, th: 0.3}
        s3 = {subset(2): 0.5, th: 0.5}
        m, K = yager_combine_all([s1, s2, s3], 4)
        self.assertAlmostEqual(K, 0.44, places=6)
        self.assertAlmostEqual(m[subset(0)], 0.30, places=6)
        self.assertAlmostEqual(m[subset(0, 1)], 0.14, places=6)
        self.assertAlmostEqual(m[subset(2)], 0.06, places=6)
        self.assertAlmostEqual(m[th], 0.50, places=6)
        p = pignistic(m, 4)
        for got, want in zip(p, [0.495, 0.195, 0.185, 0.125]):
            self.assertAlmostEqual(got, want, places=6)

    def test_no_conflict_matches_dempster(self):
        th = full_frame(4)
        ms = [{subset(0): 0.8, th: 0.2}] * 3
        m, K = yager_combine_all(ms, 4)
        self.assertEqual(K, 0.0)
        self.assertAlmostEqual(m[subset(0)], 0.992, places=6)
        self.assertAlmostEqual(m[th], 0.008, places=6)


class TestOracle(unittest.TestCase):
    def test_point_likelihood_columns_sum_to_one(self):
        P = point_likelihood(0.2, 0.3, 4)
        np.testing.assert_allclose(P.sum(axis=0), 1.0)
        self.assertAlmostEqual(P[0, 0], 0.61, places=12)
        self.assertAlmostEqual(P[1, 0], 0.13, places=12)

    def test_single_fine_report(self):
        # Radar points at F: posterior 0.61 / (0.61 + 3 x 0.13) = 0.61.
        th = full_frame(4)
        p = oracle_bayes_fuse([{subset(0): 0.8, th: 0.2}], 0.2, 0.3, sensors=SENSORS[:1])
        for got, want in zip(p, [0.61, 0.13, 0.13, 0.13]):
            self.assertAlmostEqual(got, want, places=12)

    def test_single_coarse_report(self):
        # Radar reports {F,T}: 0.74 for F and T, 0.26 for H and U, total 2.0.
        th = full_frame(4)
        p = oracle_bayes_fuse([{subset(0, 1): 0.8, th: 0.2}], 0.2, 0.3, sensors=SENSORS[:1])
        for got, want in zip(p, [0.37, 0.37, 0.13, 0.13]):
            self.assertAlmostEqual(got, want, places=12)

    def test_oracle_is_best_calibrated_on_average(self):
        # Strict propriety: in expectation no method beats the true posterior.
        rows = run_grid_ext(trials=300, alphas=[0.3], conflicts=[0.0, 0.4])
        for r in rows:
            for k in ("DS", "Bayes", "BayesDisc", "Yager"):
                self.assertLessEqual(r["brier_Oracle"], r[f"brier_{k}"] + 0.01)


if __name__ == "__main__":
    unittest.main()
