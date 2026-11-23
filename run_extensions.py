"""
Extensions for the final case study paper.

    python run_extensions.py            # 5 seeds x 36 cells x 2,000 contacts
    python run_extensions.py --quick    # 200 contacts per cell, for a fast check

1. Five-method grid at the original seed: DS, plain Bayes, discounted Bayes,
   Yager's rule (discounted, BetP) and the oracle Bayesian posterior.  The
   per-cell random streams are those of run_experiment.py, so the first three
   columns reproduce results/grid_summary.csv exactly.
2. Seed replication: the same grid under five independent seeds.  For every
   cell and comparison it records whether the paired 95 percent interval
   favours one method or spans zero, and how much the difference varies
   across seeds.
Writes results/ext_grid_summary.csv, results/replication_cells.csv,
results/replication_summary.csv, results/fig_yager_oracle.png and
results/fig_replication.png, and prints the headline numbers.
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

from dsfusion.experiment import ALPHAS, CONFLICTS, SEED, TRIALS  # noqa: E402
from dsfusion.extensions import METHODS_EXT, replication_seeds, run_grid_ext  # noqa: E402

RESULTS = os.path.join(HERE, "results")
STYLE = {
    "DS":        dict(color="#2a78d6", marker="o", ls="-",  label="Dempster-Shafer (Discounted, BetP)"),
    "Bayes":     dict(color="#eb6834", marker="s", ls="-",  label="Naive Bayes (Plain)"),
    "BayesDisc": dict(color="#1baf7a", marker="^", ls="-",  label="Naive Bayes (Discounted)"),
    "Yager":     dict(color="#8a5cd6", marker="D", ls="-",  label="Yager's Rule (Discounted, BetP)"),
    "Oracle":    dict(color="#3d3c39", marker="x", ls="--", label="Oracle Bayes (Knows c)"),
}
COMPARISONS = [("DS", "Bayes"), ("DS", "BayesDisc"), ("Yager", "DS"), ("Yager", "BayesDisc")]


def verdict(d, ci):
    if d + ci < 0:
        return -1          # first method better calibrated
    if d - ci > 0:
        return 1           # first method worse
    return 0


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    trials = 200 if args.quick else TRIALS
    os.makedirs(RESULTS, exist_ok=True)

    seeds = replication_seeds(SEED, args.reps)
    grids = {}
    t0 = time.perf_counter()
    for s in seeds:
        grids[s] = run_grid_ext(seed=s, trials=trials)
        print(f"  seed {s} done ({time.perf_counter() - t0:.1f} s)")
    base = grids[SEED]
    write_csv(base, os.path.join(RESULTS, "ext_grid_summary.csv"))

    # ---- headline numbers at the original seed ---------------------------
    print("\nFive-method grid at seed", SEED)
    for a, b in COMPARISONS + [("DS", "Oracle"), ("BayesDisc", "Oracle"), ("Yager", "Oracle"), ("Bayes", "Oracle")]:
        v = [verdict(r[f"d_{a}_{b}"], r[f"d_{a}_{b}_ci"]) for r in base] if f"d_{a}_{b}" in base[0] else None
        d = np.array([r[f"d_{a}_{b}"] for r in base])
        msg = f"  {a:>9} - {b:<9}: mean {d.mean():+.4f}, min {d.min():+.4f}, max {d.max():+.4f}"
        if v is not None:
            msg += f" | better {v.count(-1)}, worse {v.count(1)}, ns {v.count(0)}"
        print(msg)
    print("  cells with Brier > 0.75:",
          {k: sum(r[f"brier_{k}"] > 0.75 for r in base) for k in METHODS_EXT})
    print("  worst Brier:", {k: round(max(r[f"brier_{k}"] for r in base), 3) for k in METHODS_EXT})
    print("  max |acc diff| vs oracle:",
          {k: round(max(abs(r[f"acc_{k}"] - r["acc_Oracle"]) for r in base), 4) for k in METHODS_EXT})

    # ---- replication --------------------------------------------------------
    cell_rows, summ = [], []
    for i, a in enumerate(ALPHAS):
        for j, c in enumerate(CONFLICTS):
            row = {"alpha": a, "conflict": c}
            for (m1, m2) in COMPARISONS:
                key = f"d_{m1}_{m2}"
                ds = np.array([grids[s][i * len(CONFLICTS) + j][key] for s in seeds])
                vs = [verdict(grids[s][i * len(CONFLICTS) + j][key],
                              grids[s][i * len(CONFLICTS) + j][key + "_ci"]) for s in seeds]
                row[f"{key}_mean"] = float(ds.mean())
                row[f"{key}_sd"] = float(ds.std(ddof=1))
                row[f"{key}_min"] = float(ds.min())
                row[f"{key}_max"] = float(ds.max())
                row[f"{key}_stable"] = int(len(set(vs)) == 1)
                row[f"{key}_verdicts"] = "".join({-1: "-", 0: "0", 1: "+"}[v] for v in vs)
            cell_rows.append(row)
    write_csv(cell_rows, os.path.join(RESULTS, "replication_cells.csv"))

    print(f"\nSeed replication over seeds {seeds}")
    for (m1, m2) in COMPARISONS:
        key = f"d_{m1}_{m2}"
        counts = []
        for s in seeds:
            v = [verdict(r[key], r[key + "_ci"]) for r in grids[s]]
            counts.append((v.count(-1), v.count(1), v.count(0)))
        counts = np.array(counts)
        stable = sum(r[f"{key}_stable"] for r in cell_rows)
        flips = sum(("-" in r[f"{key}_verdicts"]) and ("+" in r[f"{key}_verdicts"]) for r in cell_rows)
        sd = np.array([r[f"{key}_sd"] for r in cell_rows])
        summ.append({"comparison": f"{m1} - {m2}",
                     "better_mean": counts[:, 0].mean(), "better_sd": counts[:, 0].std(ddof=1),
                     "better_min": int(counts[:, 0].min()), "better_max": int(counts[:, 0].max()),
                     "worse_mean": counts[:, 1].mean(), "worse_sd": counts[:, 1].std(ddof=1),
                     "worse_min": int(counts[:, 1].min()), "worse_max": int(counts[:, 1].max()),
                     "ns_mean": counts[:, 2].mean(),
                     "cells_same_verdict_all_seeds": stable, "cells_sign_flip": flips,
                     "median_between_seed_sd": float(np.median(sd)), "max_between_seed_sd": float(sd.max())})
        print(f"  {m1:>5} - {m2:<9}: better {counts[:,0].mean():.1f}±{counts[:,0].std(ddof=1):.1f} "
              f"[{counts[:,0].min()}-{counts[:,0].max()}], worse {counts[:,1].mean():.1f}±{counts[:,1].std(ddof=1):.1f} "
              f"[{counts[:,1].min()}-{counts[:,1].max()}] | identical verdict in {stable}/36 cells, "
              f"sign flips {flips}, between-seed SD median {np.median(sd):.4f} max {sd.max():.4f}")
    write_csv(summ, os.path.join(RESULTS, "replication_summary.csv"))

    # crossover location per alpha per seed for DS vs BayesDisc
    print("\n  First conflict level where DS is significantly better than discounted Bayes,"
          " and last where it is significantly worse, per alpha (one entry per seed)")
    for i, a in enumerate(ALPHAS):
        firsts, lasts = [], []
        for s in seeds:
            sub = grids[s][i * len(CONFLICTS):(i + 1) * len(CONFLICTS)]
            v = [verdict(r["d_DS_BayesDisc"], r["d_DS_BayesDisc_ci"]) for r in sub]
            firsts.append(next((CONFLICTS[k] for k in range(len(v)) if v[k] == -1), None))
            lasts.append(max((CONFLICTS[k] for k in range(len(v)) if v[k] == 1), default=None))
        print(f"   alpha {a:.2f}: first better {firsts} | last worse {lasts}")

    # pooled max advantages per seed
    for key, lab in (("d_DS_Bayes", "max DS advantage over plain Bayes"),
                     ("d_DS_BayesDisc", "max DS advantage over disc Bayes")):
        vals = [-min(r[key] for r in grids[s]) for s in seeds]
        print(f"  {lab} per seed: {[round(v, 3) for v in vals]}")
    print("  cells with plain Bayes Brier > 0.75 per seed:",
          [sum(r["brier_Bayes"] > 0.75 for r in grids[s]) for s in seeds])
    print("  cells with DS Brier > 0.75 per seed:",
          [sum(r["brier_DS"] > 0.75 for r in grids[s]) for s in seeds])
    print("  worst DS Brier per seed:", [round(max(r["brier_DS"] for r in grids[s]), 3) for s in seeds])

    figures(base, grids, seeds)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def figures(base, grids, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    })
    # Figure 6: five methods, Brier vs conflict
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, a in zip(axes, [0.0, 0.30, 0.60]):
        sub = [r for r in base if abs(r["alpha"] - a) < 1e-9]
        x = [r["conflict"] for r in sub]
        for k, st in STYLE.items():
            y = np.array([r[f"brier_{k}"] for r in sub])
            e = np.array([r[f"brier_{k}_ci"] for r in sub])
            ax.plot(x, y, lw=2, ms=6, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
            ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.12, lw=0)
        ax.axhline(0.75, color="#52514e", lw=1, ls=":")
        ax.text(0.0, 0.765, "Uniform Forecast (0.75)", fontsize=8, color="#52514e")
        ax.set_title(f"Discount Alpha = {a:.2f}", color="#0b0b0b")
        ax.set_xlabel("Conflict c (Decoy Probability)")
    axes[0].set_ylabel("Brier Score (Mean, 95% CI, Lower Is Better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Brier Score of All Five Methods versus Sensor Conflict", y=1.15,
                 fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(RESULTS, "fig_yager_oracle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 7: replication of the DS vs discounted Bayes crossover
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap("viridis")
    for ax, (m1, m2, ttl) in zip(axes, [("DS", "BayesDisc", "(a) Dempster-Shafer minus Discounted Naive Bayes"),
                                         ("Yager", "DS", "(b) Yager's Rule minus Dempster-Shafer")]):
        key = f"d_{m1}_{m2}"
        for i, a in enumerate(ALPHAS):
            if a == 0.0 and m2 == "BayesDisc":
                pass
            col = cmap(i / (len(ALPHAS) - 1) * 0.9)
            pts = np.array([[grids[s][i * len(CONFLICTS) + j][key] for j in range(len(CONFLICTS))] for s in seeds])
            x = np.array(CONFLICTS) + (i - 2.5) * 0.006
            ax.plot(x, pts.mean(axis=0), color=col, lw=1.8, marker="o", ms=4, label=f"Alpha = {a:.2f}")
            for row in pts:
                ax.scatter(x, row, color=col, s=8, alpha=0.45, lw=0)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_title(ttl, fontsize=10.5, color="#0b0b0b")
        ax.set_xlabel("Conflict c (Decoy Probability)")
        ax.set_ylabel("Paired Brier Difference (Below 0 Favours First)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Paired Brier Differences under {len(seeds)} Independent Seeds (Line Is Mean, Dots Are Seeds)",
                 y=1.10, fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig_replication.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
