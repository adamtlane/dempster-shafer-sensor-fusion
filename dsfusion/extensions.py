"""
Extensions for the final case study: Yager's combination rule, an oracle
Bayesian reference, and seed replication of the main grid.

Yager's rule (Yager, 1987)
--------------------------
Dempster's rule removes conflict by renormalising (closed world).  Yager's
rule keeps the unnormalised conjunctive combination and transfers the
conflict mass to Theta, so disagreement becomes ignorance rather than
being redistributed over the surviving focal sets:

    q(C)       = sum_{A1 ∩ ... ∩ Ak = C} m1(A1) ... mk(Ak)
    mY(C)      = q(C)                 for C != empty, C != Theta
    mY(Theta)  = q(Theta) + q(empty)

Yager's rule is not associative, so the k sources are combined
conjunctively first and the accumulated empty-set mass is moved to Theta
once, at the end (the standard multi-source form).

Oracle Bayes
------------
The exact Bayesian posterior under the true generative model of
simulator.py, given knowledge that neither practical method has: the
malfunction rate alpha, the decoy rate c, each sensor's coarse-report
probability and grouping.  For sensor k and true class t,

    P(x | t)       = alpha / n + (1 - alpha) [ (1 - c) 1{x = t}
                                               + c / (n - 1) 1{x != t} ]
    P({x} | t)     = (1 - p_k) P(x | t)
    P(G | t)       = p_k * sum_{x in G} P(x | t)      for a coarse group G

Because the Brier score is strictly proper, this posterior minimises the
expected Brier score among all fusion rules that see the same reports.
It is therefore a lower bound (the best achievable calibration), not a
competitor that could be fielded.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .bayes import naive_bayes_fuse
from .core import Mass, cardinality, discount, full_frame, members, pignistic
from .experiment import (ALPHAS, CONFLICTS, SEED, TRIALS, accuracy_credit, brier,
                         cell_rng, fuse_ds, mean_ci)
from .simulator import N_CLASSES, SENSORS, Sensor, simulate_contact

METHODS_EXT = ("DS", "Bayes", "BayesDisc", "Yager", "Oracle")


# ---------------------------------------------------------------------------
# Yager's rule
# ---------------------------------------------------------------------------
def conjunctive_combine(m1: Dict[int, float], m2: Dict[int, float]) -> Dict[int, float]:
    """Unnormalised conjunctive combination; key 0 holds the empty-set mass."""
    q: Dict[int, float] = {}
    for A, a in m1.items():
        for B, b in m2.items():
            C = A & B
            q[C] = q.get(C, 0.0) + a * b
    return q


def yager_combine_all(masses: Sequence[Mass], n: int):
    """Combine all sources conjunctively, then move total conflict to Theta.

    Returns (combined mass, overall conflict K)."""
    acc: Dict[int, float] = dict(masses[0])
    for m in masses[1:]:
        acc = conjunctive_combine(acc, m)
    K = acc.pop(0, 0.0)
    theta = full_frame(n)
    acc[theta] = acc.get(theta, 0.0) + K
    return {C: v for C, v in acc.items() if v > 1e-15}, K


def fuse_yager(reports, alpha: float, n: int = N_CLASSES) -> List[float]:
    """Discount by alpha, combine with Yager's rule, decide with BetP."""
    disc = [discount(m, alpha, n) for m in reports]
    m, _ = yager_combine_all(disc, n)
    return pignistic(m, n)


# ---------------------------------------------------------------------------
# Oracle Bayes
# ---------------------------------------------------------------------------
def point_likelihood(alpha: float, c: float, n: int) -> np.ndarray:
    """Matrix P[x, t] = P(sensor points at x | true class t)."""
    P = np.full((n, n), alpha / n + (1.0 - alpha) * c / (n - 1))
    np.fill_diagonal(P, alpha / n + (1.0 - alpha) * (1.0 - c))
    return P


def oracle_bayes_fuse(reports: Sequence[Mass], alpha: float, c: float,
                      sensors: Sequence[Sensor] = SENSORS, n: int = N_CLASSES) -> List[float]:
    """Exact posterior given the true generative model (uniform prior)."""
    P = point_likelihood(alpha, c, n)
    theta = full_frame(n)
    post = np.full(n, 1.0 / n)
    for sen, m in zip(sensors, reports):
        focal = next(A for A in m if A != theta)
        if cardinality(focal) == 1:
            lik = (1.0 - sen.p_coarse) * P[members(focal)[0], :]
        else:
            lik = sen.p_coarse * P[members(focal), :].sum(axis=0)
        post = post * lik
    return list(post / post.sum())


# ---------------------------------------------------------------------------
# Five-method grid and seed replication
# ---------------------------------------------------------------------------
def fuse_all_ext(reports, alpha: float, c: float, n: int = N_CLASSES) -> Dict[str, object]:
    p_ds, K = fuse_ds(reports, alpha, n)
    return {
        "DS": p_ds,
        "Bayes": naive_bayes_fuse(reports, n),
        "BayesDisc": naive_bayes_fuse(reports, n, alphas=[alpha] * len(reports)),
        "Yager": fuse_yager(reports, alpha, n),
        "Oracle": oracle_bayes_fuse(reports, alpha, c, n=n),
        "K": K,
    }


def run_cell_ext(alpha: float, c: float, trials: int, rng: np.random.Generator):
    acc = {k: np.empty(trials) for k in METHODS_EXT}
    br = {k: np.empty(trials) for k in METHODS_EXT}
    for j in range(trials):
        t, reports, _ = simulate_contact(alpha, c, rng)
        out = fuse_all_ext(reports, alpha, c)
        for k in METHODS_EXT:
            acc[k][j] = accuracy_credit(out[k], t)
            br[k][j] = brier(out[k], t)
    return acc, br


def summarise_cell(alpha, c, acc, br) -> Dict[str, float]:
    row = {"alpha": alpha, "conflict": c}
    for k in METHODS_EXT:
        row[f"acc_{k}"], row[f"acc_{k}_ci"] = mean_ci(acc[k])
        row[f"brier_{k}"], row[f"brier_{k}_ci"] = mean_ci(br[k])
    for a, b in (("DS", "Bayes"), ("DS", "BayesDisc"), ("Yager", "DS"),
                 ("Yager", "BayesDisc"), ("DS", "Oracle"), ("BayesDisc", "Oracle"),
                 ("Bayes", "Oracle"), ("Yager", "Oracle")):
        row[f"d_{a}_{b}"], row[f"d_{a}_{b}_ci"] = mean_ci(br[a] - br[b])
    return row


def run_grid_ext(seed: int = SEED, trials: int = TRIALS, alphas=ALPHAS, conflicts=CONFLICTS):
    """Same per-cell streams as experiment.run_grid, so DS and both Bayes
    columns reproduce the main grid exactly when seed = SEED."""
    rows = []
    for i, a in enumerate(alphas):
        for j, c in enumerate(conflicts):
            acc, br = run_cell_ext(a, c, trials, cell_rng(seed, i, j))
            rows.append(summarise_cell(a, c, acc, br))
    return rows


def replication_seeds(base: int = SEED, reps: int = 5) -> List[int]:
    """Replicate 0 is the original seed; the rest are fixed offsets."""
    return [base + 1000 * r for r in range(reps)]
