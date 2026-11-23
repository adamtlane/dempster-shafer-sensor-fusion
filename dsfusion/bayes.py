"""
Naive Bayes fusion baseline, applied to the identical sensor evidence.

Each sensor report arrives as a mass function m_k.  The Bayesian baseline
cannot hold belief on a set of classes, so it first converts the report to a
probability over single classes by spreading every focal mass evenly over
the members of its focal set (the same equal sharing the pignistic
transform uses).  That vector is read as the class likelihood
P(report_k | class), and the reports are fused under the naive assumption
that they are conditionally independent given the class:

    posterior(c)  proportional to  prior(c) * prod_k L_k(c)

The prior is uniform by default.  This is a deliberately neutral choice: the
baseline is not handed any information the evidence-theory model lacks.

Two variants are provided.

* plain      : L_k is the report as given.  The sensor's stated confidence is
               taken at face value, so a fully reliable sensor is assumed.
* discounted : L_k is first tempered toward the uniform distribution by the
               same discount rate alpha used for Dempster-Shafer,
               L_k^alpha = (1 - alpha) L_k + alpha / n.
               This is the Bayesian counterpart of Shafer discounting (a
               mixture with an uninformative report) and exists to test the
               counter-explanation that any Dempster-Shafer advantage comes
               from discounting rather than from Dempster's rule itself.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .core import Mass, cardinality, members


def mass_to_likelihood(m: Mass, n: int) -> List[float]:
    """Spread each focal mass evenly over its members (forced precision)."""
    p = [0.0] * n
    for A, v in m.items():
        share = v / cardinality(A)
        for i in members(A):
            p[i] += share
    return p


def temper(p: Sequence[float], alpha: float) -> List[float]:
    """Mix a likelihood with the uniform vector: (1 - alpha) p + alpha / n."""
    n = len(p)
    return [(1.0 - alpha) * x + alpha / n for x in p]


def naive_bayes_fuse(
    reports: Sequence[Mass],
    n: int,
    prior: Optional[Sequence[float]] = None,
    alphas: Optional[Sequence[float]] = None,
) -> List[float]:
    """
    Fuse sensor reports with the naive Bayes product rule.

    reports : raw (undiscounted) mass functions, one per sensor
    prior   : prior over classes, uniform if None
    alphas  : per-sensor discount rates for the discounted variant, or None
              for the plain variant

    Returns the normalised posterior.  Zero likelihoods are allowed, and if
    every class receives zero posterior weight the function raises, which
    mirrors the K = 1 failure of Dempster's rule.
    """
    post = list(prior) if prior is not None else [1.0 / n] * n
    for k, m in enumerate(reports):
        L = mass_to_likelihood(m, n)
        if alphas is not None:
            L = temper(L, alphas[k])
        post = [a * b for a, b in zip(post, L)]
    z = sum(post)
    if z <= 0.0:
        raise ZeroDivisionError("All classes have zero posterior; evidence is totally conflicting.")
    return [x / z for x in post]
