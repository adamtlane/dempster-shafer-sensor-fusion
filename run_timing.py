"""
Runtime versus frame size, the cost limitation named in the Annotated Outline.

    python run_timing.py            # frame sizes n = 2 .. 10, about one minute
    python run_timing.py --max-n 8  # smaller sweep for a fast check

For each frame size n the script times three fusion workloads on three
sensor reports, using the same functions the main experiment uses.

* DS dense  : every report is a random mass function with mass on all
              2**n - 1 nonempty subsets.  This is the worst case for
              Dempster's rule, whose cost per combination is
              (#focal of m1) x (#focal of m2), here (2**n - 1)**2.
* DS sensor : every report has the shape the simulator produces, mass on one
              singleton or two class group plus Theta, discounted by 0.30,
              then combined and passed through the pignistic transform.
* Bayes     : plain naive Bayes on the same sensor shaped reports, cost O(n)
              per report.

Each workload is measured in 15 independent timing blocks.  Within a block
the call is repeated until at least 20 ms have elapsed and the mean per
call is recorded, so the reported median and interquartile range describe
the spread of 15 block means.  Inputs come from a fixed seed, so the work
done is identical across runs; only wall clock timing varies by machine.

Writes results/timing_summary.csv and results/fig_runtime_vs_frame.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dsfusion.bayes import naive_bayes_fuse  # noqa: E402
from dsfusion.core import combine_all, discount, full_frame, pignistic  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SEED = 605745
BLOCKS = 15
MIN_BLOCK_SECONDS = 0.02
N_SENSORS = 3


def dense_mass(n, rng):
    """Random mass function with every nonempty subset focal (worst case)."""
    w = rng.dirichlet(np.ones(2 ** n - 1))
    return {mask: float(v) for mask, v in zip(range(1, 2 ** n), w)}


def sensor_mass(n, rng):
    """Mass on one singleton or a two class group, the rest on Theta."""
    i = int(rng.integers(n))
    focal = 1 << i
    if n > 2 and rng.random() < 0.4:
        j = int(rng.choice([k for k in range(n) if k != i]))
        focal |= 1 << j
    s = float(rng.uniform(0.6, 0.9))
    return {focal: s, full_frame(n): 1.0 - s}


def time_call(fn):
    """Median and quartiles of the per call time over BLOCKS blocks."""
    fn()  # warm up
    means = []
    for _ in range(BLOCKS):
        reps, t0 = 0, time.perf_counter()
        while True:
            fn()
            reps += 1
            el = time.perf_counter() - t0
            if el >= MIN_BLOCK_SECONDS:
                break
        means.append(el / reps)
    q1, med, q3 = np.percentile(means, [25, 50, 75])
    return med, q1, q3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    rows = []
    print(f"{'n':>3} {'subsets':>8} | {'DS dense (s)':>14} {'DS sensor (s)':>14} {'Bayes (s)':>12}")
    for n in range(2, args.max_n + 1):
        rng = np.random.default_rng([SEED, n])
        dense = [dense_mass(n, rng) for _ in range(N_SENSORS)]
        reports = [sensor_mass(n, rng) for _ in range(N_SENSORS)]

        def ds_dense():
            return combine_all(dense)

        def ds_sensor():
            m, _ = combine_all([discount(r, 0.30, n) for r in reports])
            return pignistic(m, n)

        def bayes():
            return naive_bayes_fuse(reports, n)

        row = {"n": n, "subsets": 2 ** n - 1,
               "dense_focal_out": len(combine_all(dense)[0])}
        for key, fn in [("ds_dense", ds_dense), ("ds_sensor", ds_sensor), ("bayes", bayes)]:
            med, q1, q3 = time_call(fn)
            row[f"{key}_median_s"], row[f"{key}_q1_s"], row[f"{key}_q3_s"] = med, q1, q3
        rows.append(row)
        print(f"{n:>3} {row['subsets']:>8} | {row['ds_dense_median_s']:>14.3e} "
              f"{row['ds_sensor_median_s']:>14.3e} {row['bayes_median_s']:>12.3e}", flush=True)

    with open(os.path.join(RESULTS, "timing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Growth rates fitted on the upper half of the sweep, where fixed overhead is negligible
    big = [r for r in rows if r["n"] >= 6]
    x = np.log2([r["subsets"] for r in big])
    slope = np.polyfit(x, np.log2([r["ds_dense_median_s"] for r in big]), 1)[0]
    ratio = [rows[i + 1]["ds_dense_median_s"] / rows[i]["ds_dense_median_s"]
             for i in range(len(rows) - 1) if rows[i]["n"] >= 6]
    print(f"\nDS dense: log-log slope of time against number of subsets (n >= 6) = {slope:.2f} "
          f"(2.0 expected for a cost of (2^n - 1)^2)")
    print(f"DS dense: mean time ratio per added class (n >= 6) = {np.mean(ratio):.2f} (4.0 expected)")
    last = rows[-1]
    print(f"At n = {last['n']}: DS dense / Bayes = {last['ds_dense_median_s'] / last['bayes_median_s']:.0f}x, "
          f"DS sensor / Bayes = {last['ds_sensor_median_s'] / last['bayes_median_s']:.1f}x")
    four = [r for r in rows if r["n"] == 4][0]
    print(f"At n = 4 (the experiment): DS sensor {four['ds_sensor_median_s'] * 1e6:.1f} us, "
          f"Bayes {four['bayes_median_s'] * 1e6:.1f} us, DS dense {four['ds_dense_median_s'] * 1e6:.1f} us")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    })
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for key, color, marker, lab in [
            ("ds_dense", "#2a78d6", "o", "Dempster-Shafer, Dense Masses (Worst Case)"),
            ("ds_sensor", "#7a4fc9", "D", "Dempster-Shafer, Sensor Shaped Masses"),
            ("bayes", "#eb6834", "s", "Naive Bayes (Plain)")]:
        med = np.array([r[f"{key}_median_s"] for r in rows])
        lo = med - np.array([r[f"{key}_q1_s"] for r in rows])
        hi = np.array([r[f"{key}_q3_s"] for r in rows]) - med
        ax.errorbar(ns, med * 1e3, yerr=[lo * 1e3, hi * 1e3], color=color, marker=marker,
                    lw=2, ms=6, capsize=3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Classes n in the Frame of Discernment")
    ax.set_ylabel("Time per Three Sensor Fusion (ms, Median and IQR)")
    ax.set_xticks(ns)
    ax.axvline(4, color="#9a9893", lw=1, ls="--")
    ax.text(4.1, 20, "Experiment (n = 4)", color="#52514e", fontsize=9)
    ax.set_title("Fusion Runtime versus Frame Size", fontweight="bold", color="#0b0b0b")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_runtime_vs_frame.png"), dpi=150)
    plt.close(fig)
    print("Wrote results/timing_summary.csv and results/fig_runtime_vs_frame.png")


if __name__ == "__main__":
    main()
