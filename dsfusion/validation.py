"""
Expected versus observed table for the hand derived test cases.

Each row pairs a value worked out by hand (arithmetic in README.md, section
"Hand-derived test cases") with the value the implementation returns, so the
comparison is visible in the notebook rather than only as a passing assert.
The same values are asserted in tests/test_fusion.py.
"""

from __future__ import annotations

from typing import List, Tuple

from .bayes import naive_bayes_fuse
from .core import (belief, combine_all, dempster_combine, discount, full_frame,
                   make_mass, pignistic, plausibility, subset, total_conflict, vacuous)
from .experiment import brier


def validation_rows() -> List[Tuple[str, str, float, float]]:
    """Return (test case, quantity, expected by hand, observed) rows."""
    rows = []
    add = lambda case, q, exp, obs: rows.append((case, q, float(exp), float(obs)))

    # T1 Zadeh, frame {M, C, T}
    M, C, T, TH3 = subset(0), subset(1), subset(2), full_frame(3)
    z1 = make_mass([(M, 0.99), (T, 0.01)])
    z2 = make_mass([(C, 0.99), (T, 0.01)])
    m, K = dempster_combine(z1, z2)
    add("T1 Zadeh, trusted", "K", 0.9999, K)
    add("T1 Zadeh, trusted", "m({T})", 1.0, m[T])
    add("T1 Zadeh, trusted", "Bayes P(T)", 1.0, naive_bayes_fuse([z1, z2], 3)[2])
    m, K = dempster_combine(discount(z1, 0.1, 3), discount(z2, 0.1, 3))
    add("T1 Zadeh, discount 0.1", "K", 0.809919, K)
    add("T1 Zadeh, discount 0.1", "m({M}) = m({C})", 0.0891 / 0.190081, m[M])
    add("T1 Zadeh, discount 0.1", "m({T})", 0.001881 / 0.190081, m[T])
    add("T1 Zadeh, discount 0.1", "m(Theta)", 0.01 / 0.190081, m[TH3])

    # T2 three agreeing sensors, frame {F, T, H, U}
    TH = full_frame(4)
    rep = [make_mass([(subset(0), 0.8), (TH, 0.2)])] * 3
    m, ks = combine_all(rep)
    add("T2 agreeing sensors", "overall K", 0.0, total_conflict(ks))
    add("T2 agreeing sensors", "m({F})", 0.992, m[subset(0)])
    add("T2 agreeing sensors", "m(Theta)", 0.008, m[TH])
    add("T2 agreeing sensors", "BetP(F)", 0.994, pignistic(m, 4)[0])
    add("T2 agreeing sensors", "Bayes P(F)", 0.614125 / 0.6145, naive_bayes_fuse(rep, 4)[0])

    # T3 ignorance limits
    add("T3 ignorance", "BetP of vacuous mass, each class", 0.25, pignistic(vacuous(4), 4)[0])
    r = make_mass([(subset(1), 0.7), (TH, 0.3)])
    add("T3 ignorance", "m(Theta) after discount 1.0", 1.0, discount(r, 1.0, 4)[TH])
    add("T3 ignorance", "m({T}) after discount 0.0", 0.7, discount(r, 0.0, 4)[subset(1)])

    # T5 hand traced three sensor case
    r1 = make_mass([(subset(0), 0.8), (TH, 0.2)])
    r2 = make_mass([(subset(0, 1), 0.7), (TH, 0.3)])
    r3 = make_mass([(subset(2), 0.5), (TH, 0.5)])
    m12, k1 = dempster_combine(discount(r1, 0.25, 4), r2)
    add("T5 three sensors", "K after sensors 1+2", 0.0, k1)
    add("T5 three sensors", "m({F,T}) after sensors 1+2", 0.28, m12[subset(0, 1)])
    m, k2 = dempster_combine(m12, r3)
    add("T5 three sensors", "K after sensor 3", 0.44, k2)
    add("T5 three sensors", "m({F})", 0.30 / 0.56, m[subset(0)])
    add("T5 three sensors", "m({F,T})", 0.25, m[subset(0, 1)])
    add("T5 three sensors", "m({H})", 0.06 / 0.56, m[subset(2)])
    add("T5 three sensors", "Bel({F})", 0.30 / 0.56, belief(m, subset(0)))
    add("T5 three sensors", "Pl({F})", 0.50 / 0.56, plausibility(m, subset(0)))
    betp = pignistic(m, 4)
    for i, (c, e) in enumerate(zip("FTHU", [0.6875, 0.085 / 0.56, 0.075 / 0.56, 0.015 / 0.56])):
        add("T5 three sensors", f"BetP({c})", e, betp[i])
    add("T5 three sensors", "Bayes P(F), plain", 0.04515625 / 0.050625, naive_bayes_fuse([r1, r2, r3], 4)[0])
    add("T5 three sensors", "Bayes P(F), discounted", 0.0371875 / 0.048125,
        naive_bayes_fuse([r1, r2, r3], 4, alphas=[0.25, 0.0, 0.0])[0])

    # T6 metrics
    add("T6 metrics", "Brier of uniform forecast", 0.75, brier([0.25] * 4, 0))
    add("T6 metrics", "Brier of confident wrong forecast", 2.0, brier([0, 1, 0, 0], 0))
    return rows
