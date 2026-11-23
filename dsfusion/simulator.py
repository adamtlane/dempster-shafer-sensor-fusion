"""
Simulated multi-sensor contacts with known ground truth.

Frame of discernment (n = 4):
    0 F  Fighter
    1 T  Transport
    2 H  Helicopter
    3 U  UAV

Three dissimilar sensors observe each contact.  Each sensor can sometimes
resolve a contact only to a coarse group of two classes, and every sensor
groups the classes differently, which is the epistemic ignorance the
proposal names.  The groupings are illustrative, not drawn from any real
system:

    Radar   (kinematics) : {F,T} fixed wing          vs {H,U} slow movers
    Imaging (size/shape) : {F,U} small airframe      vs {T,H} large airframe
    ESM     (emitters)   : {F,H} military emitter    vs {T,U} other emitter

Report generation for one sensor, given the true class t
--------------------------------------------------------
1. Unreliability.  With probability alpha_true the sensor malfunctions and
   its pointed-at class x is drawn uniformly from all n classes.  This is
   exactly the event that Shafer discounting with rate alpha models.
2. Conflict.  Otherwise, with probability c the sensor is deceived and x is a
   decoy drawn uniformly from the n - 1 classes other than t, independently
   for each sensor.  Neither fusion method is told c.
3. Otherwise x = t.
4. Ignorance.  With probability p_coarse the sensor resolves x only to its
   coarse group G(x); otherwise to the singleton {x}.
5. The report is the mass function m(focal) = s, m(Theta) = 1 - s, where s
   is the sensor's stated confidence.

The fusion methods receive only the reports (and, for discounting, the
assumed reliability alpha_assumed, which equals alpha_true unless a
misspecification study sets it otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .core import Mass, full_frame, subset

CLASS_NAMES = ["F", "T", "H", "U"]
CLASS_LONG = ["Fighter", "Transport", "Helicopter", "UAV"]
N_CLASSES = 4


@dataclass(frozen=True)
class Sensor:
    name: str
    confidence: float          # s, mass placed on the reported focal set
    p_coarse: float            # probability of a coarse (two-class) report
    groups: Tuple[int, int]    # the two coarse groups as bitmasks

    def group_of(self, x: int) -> int:
        bit = 1 << x
        return self.groups[0] if self.groups[0] & bit else self.groups[1]


SENSORS: List[Sensor] = [
    Sensor("Radar",   0.80, 0.40, (subset(0, 1), subset(2, 3))),
    Sensor("Imaging", 0.85, 0.30, (subset(0, 3), subset(1, 2))),
    Sensor("ESM",     0.70, 0.50, (subset(0, 2), subset(1, 3))),
]


def sensor_report(
    sensor: Sensor, true_class: int, alpha_true: float, conflict: float,
    rng: np.random.Generator, n: int = N_CLASSES,
) -> Tuple[Mass, str]:
    """Generate one raw (undiscounted) report and the mechanism that produced it."""
    if rng.random() < alpha_true:
        x, kind = int(rng.integers(n)), "unreliable"
    elif rng.random() < conflict:
        others = [k for k in range(n) if k != true_class]
        x, kind = int(others[rng.integers(n - 1)]), "decoy"
    else:
        x, kind = true_class, "honest"
    focal = sensor.group_of(x) if rng.random() < sensor.p_coarse else (1 << x)
    s = sensor.confidence
    return {focal: s, full_frame(n): 1.0 - s}, kind


def simulate_contact(
    alpha_true: float, conflict: float, rng: np.random.Generator,
    sensors: List[Sensor] = SENSORS, n: int = N_CLASSES,
) -> Tuple[int, List[Mass], List[str]]:
    """Draw a true class uniformly and one report from every sensor."""
    t = int(rng.integers(n))
    reports, kinds = [], []
    for sen in sensors:
        m, kind = sensor_report(sen, t, alpha_true, conflict, rng, n)
        reports.append(m)
        kinds.append(kind)
    return t, reports, kinds
