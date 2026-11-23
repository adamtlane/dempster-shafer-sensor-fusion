"""
Dempster-Shafer evidence theory, implemented from first principles.

Representation
--------------
A frame of discernment Theta with n singleton hypotheses is indexed 0..n-1.
Every subset A of Theta is encoded as an integer bitmask: bit i is set when
hypothesis i belongs to A.  For the four-class frame used in the experiment
there are 2**4 = 16 subsets, 15 of them nonempty, and Theta itself is the
mask 0b1111 = 15.

A mass function (basic probability assignment) is stored as a plain dict
{bitmask: mass} that holds only the focal elements (the subsets with
nonzero mass).  A valid mass function satisfies

    m(empty) = 0        and        sum_A m(A) = 1.

This sparse dict form keeps Dempster's rule cheap: the cost of combining
two mass functions is (#focal elements of m1) x (#focal elements of m2)
set intersections, each a single bitwise AND.

Modeling assumptions built into this module (stated, not hidden)
---------------------------------------------------------------
1. Closed world.  The true class is assumed to lie in Theta, so any mass
   Dempster's rule sends to the empty set is treated as conflict and
   removed by renormalisation (Shafer, 1976).  The open-world alternative
   of the transferable belief model (Smets and Kennes, 1994) would keep
   that mass on the empty set instead.
2. Distinct, independent sources.  Dempster's rule is only justified when
   the bodies of evidence are independent.  The naive Bayes baseline makes
   the same assumption, so the comparison is fair on this point.
3. Reliability as discounting.  A source believed reliable with
   probability 1 - alpha is discounted by moving a fraction alpha of every
   focal mass onto Theta (Shafer's discounting).  This is one specific
   model of unreliability, namely "with probability alpha the report
   carries no information".
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Mass = Dict[int, float]

# Numerical tolerance used when checking that masses sum to one and when
# dropping focal elements whose mass has underflowed to (almost) zero.
TOL = 1e-12


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def full_frame(n: int) -> int:
    """Bitmask of Theta for a frame with n singletons (e.g. n=4 -> 0b1111)."""
    return (1 << n) - 1


def subset(*members: int) -> int:
    """Bitmask for the subset holding the given singleton indices."""
    mask = 0
    for i in members:
        mask |= 1 << i
    return mask


def members(mask: int) -> List[int]:
    """Singleton indices contained in a bitmask, in ascending order."""
    out, i = [], 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


def cardinality(mask: int) -> int:
    """|A|, the number of singletons in subset A."""
    return bin(mask).count("1")


def label(mask: int, names: Sequence[str], n: int | None = None) -> str:
    """Human readable form of a subset, e.g. '{F,T}' or 'Theta'."""
    if n is not None and mask == full_frame(n):
        return "Theta"
    return "{" + ",".join(names[i] for i in members(mask)) + "}"


# ---------------------------------------------------------------------------
# Mass function construction and validation
# ---------------------------------------------------------------------------
def make_mass(assignments: Iterable[Tuple[int, float]]) -> Mass:
    """Build a mass function from (bitmask, mass) pairs, merging duplicates."""
    m: Mass = {}
    for mask, v in assignments:
        if mask == 0:
            raise ValueError("A mass function may not assign mass to the empty set.")
        if v < -TOL:
            raise ValueError("Masses must be nonnegative.")
        if v > TOL:
            m[mask] = m.get(mask, 0.0) + v
    validate(m)
    return m


def vacuous(n: int) -> Mass:
    """The vacuous mass function m(Theta) = 1, i.e. total ignorance."""
    return {full_frame(n): 1.0}


def validate(m: Mass, tol: float = 1e-9) -> None:
    """Raise ValueError unless m is a normalised mass function."""
    if 0 in m:
        raise ValueError("Mass on the empty set is not allowed (closed world).")
    total = sum(m.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Masses sum to {total:.12f}, not 1.")


# ---------------------------------------------------------------------------
# Reliability discounting
# ---------------------------------------------------------------------------
def discount(m: Mass, alpha: float, n: int) -> Mass:
    """
    Shafer discounting with rate alpha in [0, 1].

        m_alpha(A)     = (1 - alpha) * m(A)               for A != Theta
        m_alpha(Theta) = (1 - alpha) * m(Theta) + alpha

    alpha = 0 leaves m unchanged (fully trusted source); alpha = 1 returns
    the vacuous mass function (source ignored entirely).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    theta = full_frame(n)
    out: Mass = {}
    for A, v in m.items():
        w = (1.0 - alpha) * v
        if w > TOL:
            out[A] = out.get(A, 0.0) + w
    out[theta] = out.get(theta, 0.0) + alpha
    return out


# ---------------------------------------------------------------------------
# Dempster's rule of combination
# ---------------------------------------------------------------------------
class TotalConflictError(ValueError):
    """Raised when K = 1, where Dempster's rule is undefined."""


def dempster_combine(m1: Mass, m2: Mass) -> Tuple[Mass, float]:
    """
    Combine two mass functions with Dempster's rule.

        unnormalised:  q(C) = sum_{A intersect B = C} m1(A) m2(B)
        conflict:      K    = q(empty)
        normalised:    m12(C) = q(C) / (1 - K)      for C != empty

    Returns (m12, K).  K is returned rather than discarded because it is the
    quantity that exposes disagreement between the sources, which is the
    point of comparison with the Bayesian rule.
    """
    q: Dict[int, float] = {}
    K = 0.0
    for A, a in m1.items():
        for B, b in m2.items():
            C = A & B
            if C == 0:
                K += a * b
            else:
                q[C] = q.get(C, 0.0) + a * b
    if 1.0 - K <= TOL:
        raise TotalConflictError("Total conflict (K = 1); Dempster's rule is undefined.")
    norm = 1.0 - K
    return {C: v / norm for C, v in q.items() if v > TOL}, K


def combine_all(masses: Sequence[Mass]) -> Tuple[Mass, List[float]]:
    """
    Fold Dempster's rule left to right over several sources.

    Returns the combined mass and the conflict K produced at each step.  The
    overall conflict of the whole combination, the K one would get by
    combining all sources at once, is 1 - prod(1 - K_i).
    """
    if not masses:
        raise ValueError("Need at least one mass function.")
    acc = dict(masses[0])
    ks: List[float] = []
    for m in masses[1:]:
        acc, k = dempster_combine(acc, m)
        ks.append(k)
    return acc, ks


def total_conflict(step_conflicts: Sequence[float]) -> float:
    """Overall conflict 1 - prod(1 - K_i) from the per-step conflicts."""
    keep = 1.0
    for k in step_conflicts:
        keep *= 1.0 - k
    return 1.0 - keep


# ---------------------------------------------------------------------------
# Belief, plausibility and the pignistic transform
# ---------------------------------------------------------------------------
def belief(m: Mass, A: int) -> float:
    """Bel(A) = sum of m(B) over nonempty B contained in A."""
    return sum(v for B, v in m.items() if B & ~A == 0)


def plausibility(m: Mass, A: int) -> float:
    """Pl(A) = sum of m(B) over B that intersect A."""
    return sum(v for B, v in m.items() if B & A)


def pignistic(m: Mass, n: int) -> List[float]:
    """
    Pignistic transform (Smets and Kennes, 1994):

        BetP(theta_i) = sum_{A containing theta_i} m(A) / |A|

    Each focal mass is shared equally among its members at decision time.
    The result is a probability over singletons that can be scored with
    accuracy and the Brier score exactly like a Bayesian posterior.
    """
    p = [0.0] * n
    for A, v in m.items():
        share = v / cardinality(A)
        for i in members(A):
            p[i] += share
    return p
