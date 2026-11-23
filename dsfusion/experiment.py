"""
Monte Carlo sensitivity experiment: Dempster-Shafer versus naive Bayes.

For every cell of the reliability x conflict grid, the same simulated
reports are fused by three methods

    DS        : Shafer-discount each report by alpha, combine with Dempster's
                rule, decide with the pignistic transform
    Bayes     : plain naive Bayes, uniform prior, no discounting
    Bayes+disc: naive Bayes with likelihoods tempered by the same alpha

and scored with accuracy and the multiclass Brier score.  Using identical
reports for all methods makes the per-trial differences paired, so the
confidence interval on the difference is much tighter than the intervals
on the two means separately.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .bayes import naive_bayes_fuse
from .core import TotalConflictError, combine_all, discount, pignistic, total_conflict
from .simulator import N_CLASSES, SENSORS, simulate_contact

METHODS = ("DS", "Bayes", "BayesDisc")

# Grid fixed by the annotated outline
ALPHAS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
CONFLICTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TRIALS = 2000
SEED = 605745


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy_credit(p: Sequence[float], truth: int) -> float:
    """
    1 if the argmax is the true class, 0 otherwise.  Ties are credited 1/k
    when the truth is one of k tied maxima, which equals the expected
    accuracy of breaking the tie at random and avoids biasing toward the
    lowest class index.
    """
    best = max(p)
    tied = [i for i, v in enumerate(p) if abs(v - best) <= 1e-12]
    return 1.0 / len(tied) if truth in tied else 0.0


def brier(p: Sequence[float], truth: int) -> float:
    """Multiclass Brier score sum_i (p_i - y_i)^2, from 0 (perfect) to 2."""
    return sum((v - (1.0 if i == truth else 0.0)) ** 2 for i, v in enumerate(p))


def mean_ci(x: np.ndarray, z: float = 1.96):
    """Mean and normal-approximation 95 percent half width."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    hw = float(z * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    return m, hw


# ---------------------------------------------------------------------------
# One fused decision per method
# ---------------------------------------------------------------------------
def fuse_ds(reports, alpha: float, n: int = N_CLASSES):
    """Discount, combine, transform.  Returns (BetP, overall conflict K)."""
    disc = [discount(m, alpha, n) for m in reports]
    try:
        m, ks = combine_all(disc)
        return pignistic(m, n), total_conflict(ks)
    except TotalConflictError:
        # Only reachable when alpha = 0 and the sensors place no mass on
        # Theta, which the simulator never does.  Fall back to ignorance.
        return [1.0 / n] * n, 1.0


def fuse_all(reports, alpha: float, n: int = N_CLASSES) -> Dict[str, object]:
    p_ds, K = fuse_ds(reports, alpha, n)
    p_b = naive_bayes_fuse(reports, n)
    p_bd = naive_bayes_fuse(reports, n, alphas=[alpha] * len(reports))
    return {"DS": p_ds, "Bayes": p_b, "BayesDisc": p_bd, "K": K}


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    alpha: float
    conflict: float
    trials: int
    acc: Dict[str, np.ndarray] = field(default_factory=dict)
    brier: Dict[str, np.ndarray] = field(default_factory=dict)
    K: np.ndarray = None


def run_cell(alpha: float, conflict: float, trials: int, rng: np.random.Generator,
             alpha_assumed: float | None = None) -> CellResult:
    """Run `trials` contacts in one grid cell; alpha_assumed defaults to alpha."""
    a_fuse = alpha if alpha_assumed is None else alpha_assumed
    acc = {k: np.empty(trials) for k in METHODS}
    br = {k: np.empty(trials) for k in METHODS}
    K = np.empty(trials)
    for j in range(trials):
        t, reports, _ = simulate_contact(alpha, conflict, rng)
        out = fuse_all(reports, a_fuse)
        for k in METHODS:
            acc[k][j] = accuracy_credit(out[k], t)
            br[k][j] = brier(out[k], t)
        K[j] = out["K"]
    return CellResult(alpha, conflict, trials, acc, br, K)


def cell_rng(seed: int, i: int, j: int) -> np.random.Generator:
    """Independent, reproducible stream per grid cell (order independent)."""
    return np.random.default_rng(np.random.SeedSequence([seed, i, j]))


def run_grid(alphas=ALPHAS, conflicts=CONFLICTS, trials=TRIALS, seed=SEED,
             verbose: bool = True) -> List[CellResult]:
    results = []
    t0 = time.perf_counter()
    for i, a in enumerate(alphas):
        for j, c in enumerate(conflicts):
            results.append(run_cell(a, c, trials, cell_rng(seed, i, j)))
        if verbose:
            print(f"  alpha={a:.2f} done ({time.perf_counter() - t0:.1f} s)")
    return results


def summarise(results: List[CellResult]) -> List[Dict[str, float]]:
    """One row per cell: means, 95 percent half widths, paired differences."""
    rows = []
    for r in results:
        row = {"alpha": r.alpha, "conflict": r.conflict, "trials": r.trials}
        for k in METHODS:
            row[f"acc_{k}"], row[f"acc_{k}_ci"] = mean_ci(r.acc[k])
            row[f"brier_{k}"], row[f"brier_{k}_ci"] = mean_ci(r.brier[k])
        # Paired differences DS minus Bayes (positive acc diff favours DS,
        # negative Brier diff favours DS)
        for base in ("Bayes", "BayesDisc"):
            row[f"dacc_DS_{base}"], row[f"dacc_DS_{base}_ci"] = mean_ci(r.acc["DS"] - r.acc[base])
            row[f"dbrier_DS_{base}"], row[f"dbrier_DS_{base}_ci"] = mean_ci(r.brier["DS"] - r.brier[base])
        row["K_mean"], row["K_ci"] = mean_ci(r.K)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


# ---------------------------------------------------------------------------
# Misspecified reliability: the discount the fusion uses differs from the
# unreliability that generated the reports
# ---------------------------------------------------------------------------
MISSPEC_TRUE = [0.15, 0.30, 0.45, 0.60]
MISSPEC_ASSUMED = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
MISSPEC_CONFLICTS = [0.0, 0.3, 0.5]


def run_misspec(true_alphas=MISSPEC_TRUE, assumed_alphas=MISSPEC_ASSUMED,
                conflicts=MISSPEC_CONFLICTS, trials=TRIALS, seed=SEED) -> List[Dict[str, float]]:
    """
    Tests modeling assumption 3 (discounting is the correct reliability model
    and its rate is known).  For each true unreliability and conflict level,
    the SAME simulated contacts are fused under every assumed discount: the
    random stream depends only on (true alpha, conflict), never on the
    assumed value, so differences across assumed values are paired.

    DS and discounted Bayes use the assumed discount; plain Bayes ignores it
    and serves as a fixed reference line.
    """
    rows = []
    for i, a_true in enumerate(true_alphas):
        for j, c in enumerate(conflicts):
            for a_hat in assumed_alphas:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 7, i, j]))
                r = run_cell(a_true, c, trials, rng, alpha_assumed=a_hat)
                row = {"alpha_true": a_true, "alpha_assumed": a_hat, "conflict": c,
                       "misspec": round(a_hat - a_true, 2), "trials": trials}
                for k in METHODS:
                    row[f"brier_{k}"], row[f"brier_{k}_ci"] = mean_ci(r.brier[k])
                    row[f"acc_{k}"], row[f"acc_{k}_ci"] = mean_ci(r.acc[k])
                for base in ("Bayes", "BayesDisc"):
                    row[f"dbrier_DS_{base}"], row[f"dbrier_DS_{base}_ci"] = mean_ci(r.brier["DS"] - r.brier[base])
                rows.append(row)
    return rows
