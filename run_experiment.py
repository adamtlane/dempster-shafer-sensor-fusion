"""
Entry point for the operable code deliverable.

    python run_experiment.py            # full 6 x 6 grid, 2,000 trials per cell
    python run_experiment.py --quick    # 200 trials per cell, for a fast check

Steps
-----
1. Print a worked trace of one simulated contact, showing every intermediate
   quantity: raw and discounted masses, the combined mass, the conflict K,
   Bel and Pl per class, the pignistic probability, and both Bayesian
   posteriors.
2. Run the reliability x conflict sweep with a fixed seed.
3. Write results/grid_summary.csv and three figures to results/.
4. Print a headline table and, for each reliability level, the first
   conflict level at which DS and plain Bayes separate (95 percent paired
   interval on the Brier difference excludes zero).
5. Run the misspecified discount study (the fusion assumes a discount that
   differs from the true unreliability), write results/misspec_summary.csv
   and results/fig_misspecified_discount.png, and print a robustness summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dsfusion.bayes import naive_bayes_fuse  # noqa: E402
from dsfusion.core import (  # noqa: E402
    belief, combine_all, discount, full_frame, label, pignistic, plausibility, total_conflict,
)
from dsfusion.experiment import (  # noqa: E402
    ALPHAS, CONFLICTS, MISSPEC_CONFLICTS, MISSPEC_TRUE, SEED, TRIALS,
    run_grid, run_misspec, summarise, write_csv,
)
from dsfusion.simulator import CLASS_NAMES, N_CLASSES, SENSORS, simulate_contact  # noqa: E402

RESULTS = os.path.join(HERE, "results")

# Categorical palette (validated: CVD and normal-vision separation pass).
# Marker shape and the legend give a second encoding beyond colour.
STYLE = {
    "DS":        dict(color="#2a78d6", marker="o", label="Dempster-Shafer (Discounted, BetP)"),
    "Bayes":     dict(color="#eb6834", marker="s", label="Naive Bayes (Plain)"),
    "BayesDisc": dict(color="#1baf7a", marker="^", label="Naive Bayes (Discounted Likelihoods)"),
}


def fmt(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def worked_trace(alpha=0.30, conflict=0.30, seed=SEED):
    """Print every intermediate quantity for one contact with real disagreement."""
    rng = np.random.default_rng(seed)
    n, TH = N_CLASSES, full_frame(N_CLASSES)
    # Draw until the contact has a decoy and the discounted reports genuinely
    # disagree (overall K > 0.2), so the trace shows conflict being handled
    # rather than an easy agreeing case.
    while True:
        t, reports, kinds = simulate_contact(alpha, conflict, rng)
        _, ks = combine_all([discount(m, alpha, n) for m in reports])
        if "decoy" in kinds and total_conflict(ks) > 0.2:
            break
    print("=" * 74)
    print(f"WORKED TRACE   alpha = {alpha}, conflict c = {conflict}, true class = {CLASS_NAMES[t]}")
    print("=" * 74)
    disc = []
    for sen, m, kind in zip(SENSORS, reports, kinds):
        d = discount(m, alpha, n)
        disc.append(d)
        raw = ", ".join(f"m({label(A, CLASS_NAMES, n)})={v:.3f}" for A, v in m.items())
        dd = ", ".join(f"m({label(A, CLASS_NAMES, n)})={v:.3f}" for A, v in d.items())
        print(f"{sen.name:8s} [{kind:10s}] raw: {raw}")
        print(f"{'':8s} {'':12s} discounted: {dd}")
    m, ks = combine_all(disc)
    print(f"\nStep conflicts K_i = {[round(k, 4) for k in ks]}, overall K = {total_conflict(ks):.4f}")
    print("Combined mass (Dempster's rule, normalised):")
    for A, v in sorted(m.items(), key=lambda kv: -kv[1]):
        print(f"   m({label(A, CLASS_NAMES, n)}) = {v:.4f}")
    print("\n class    Bel     Pl    BetP  Bayes  BayesDisc")
    betp = pignistic(m, n)
    pb = naive_bayes_fuse(reports, n)
    pbd = naive_bayes_fuse(reports, n, alphas=[alpha] * len(reports))
    for i, c in enumerate(CLASS_NAMES):
        s = 1 << i
        print(f"   {c}    {belief(m, s):.4f} {plausibility(m, s):.4f} {betp[i]:.4f} {pb[i]:.4f}  {pbd[i]:.4f}")
    print(f"Ignorance kept on Theta: m(Theta) = {m.get(TH, 0.0):.4f}")
    print()


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    })
    show_alphas = [0.0, 0.30, 0.60]

    # Figures 1 and 2: metric vs conflict, one panel per reliability level
    for metric, ylab, fname, ftitle in [
            ("acc", "Accuracy (Mean, 95% CI)", "fig_accuracy_vs_conflict.png",
             "Classification Accuracy versus Sensor Conflict at Three Discount Levels"),
            ("brier", "Brier Score (Mean, 95% CI, Lower Is Better)", "fig_brier_vs_conflict.png",
             "Brier Score versus Sensor Conflict at Three Discount Levels")]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, a in zip(axes, show_alphas):
            sub = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
            x = [r["conflict"] for r in sub]
            for k, st in STYLE.items():
                y = np.array([r[f"{metric}_{k}"] for r in sub])
                e = np.array([r[f"{metric}_{k}_ci"] for r in sub])
                ax.plot(x, y, lw=2, ms=6, color=st["color"], marker=st["marker"], label=st["label"])
                ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.15, lw=0)
            ax.set_title(f"Discount Alpha = {a:.2f}", color="#0b0b0b")
            ax.set_xlabel("Conflict c (Decoy Probability)")
        axes[0].set_ylabel(ylab)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(ftitle, y=1.12, fontsize=12, fontweight="bold", color="#0b0b0b")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(RESULTS, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure 3: paired Brier difference heat maps, DS minus each Bayes baseline.
    # Each panel has its own symmetric colour scale because the two gaps differ
    # by almost an order of magnitude; a shared scale would wash out panel b.
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cell = lambda a, c: next(r for r in rows if abs(r["alpha"] - a) < 1e-9 and abs(r["conflict"] - c) < 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for ax, base, title in [
        (axes[0], "Bayes", "(a) DS Minus Plain Naive Bayes"),
        (axes[1], "BayesDisc", "(b) DS Minus Naive Bayes with the Same Discount"),
    ]:
        grid = np.array([[cell(a, c)[f"dbrier_DS_{base}"] for c in CONFLICTS] for a in ALPHAS])
        ci = np.array([[cell(a, c)[f"dbrier_DS_{base}_ci"] for c in CONFLICTS] for a in ALPHAS])
        lim = float(np.abs(grid).max())
        ax.grid(False)
        im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(len(ALPHAS)):
            for j in range(len(CONFLICTS)):
                sig = abs(grid[i, j]) > ci[i, j]
                ax.text(j, i, f"{grid[i, j]:+.3f}" + ("" if sig else "\n(ns)"),
                        ha="center", va="center", fontsize=8, color="#0b0b0b")
        ax.set_xticks(range(len(CONFLICTS)), [f"{c:.1f}" for c in CONFLICTS])
        ax.set_yticks(range(len(ALPHAS)), [f"{a:.2f}" for a in ALPHAS])
        ax.set_xlabel("Conflict c (Decoy Probability)")
        ax.set_ylabel("Discount / Unreliability Alpha")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Paired Brier Difference")
    fig.suptitle("Paired Brier Score Difference by Discount and Conflict",
                 fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.text(0.5, 0.01, "Negative (blue) = DS better calibrated. ns = 95% paired CI spans zero. "
             "Colour scales differ between panels.", ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig_brier_difference_heatmap.png"), dpi=150)
    plt.close(fig)


def make_misspec_figure(mrows):
    """Brier score against the ASSUMED discount, one panel per (true alpha, conflict)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show_true = [0.30, 0.60]
    fig, axes = plt.subplots(len(show_true), len(MISSPEC_CONFLICTS), figsize=(12, 7.2))
    for r_i, a_true in enumerate(show_true):
        for c_i, c in enumerate(MISSPEC_CONFLICTS):
            ax = axes[r_i, c_i]
            sub = [r for r in mrows if abs(r["alpha_true"] - a_true) < 1e-9 and abs(r["conflict"] - c) < 1e-9]
            x = [r["alpha_assumed"] for r in sub]
            for k, st in STYLE.items():
                y = np.array([r[f"brier_{k}"] for r in sub])
                e = np.array([r[f"brier_{k}_ci"] for r in sub])
                ax.plot(x, y, lw=2, ms=6, color=st["color"], marker=st["marker"], label=st["label"])
                ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.15, lw=0)
            ax.axvline(a_true, color="#52514e", lw=1, ls="--")
            ax.text(a_true, 1.0, " True Alpha", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color="#52514e")
            ax.set_title(f"True Alpha = {a_true:.2f}, Conflict c = {c:.1f}", fontsize=10)
            if r_i == len(show_true) - 1:
                ax.set_xlabel("Assumed Discount Used by the Fusion")
            if c_i == 0:
                ax.set_ylabel("Brier Score (Lower Is Better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Brier Score versus Assumed Discount When the Discount Is Misspecified",
                 y=1.07, fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.text(0.5, -0.01, "Shaded bands are unpaired 95% CIs of each mean. All methods fuse the same contacts at every "
             "assumed discount, so the paired differences in misspec_summary.csv are several times tighter.",
             ha="center", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig_misspecified_discount.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def misspec_report(mrows):
    """Print the robustness of each method to a misjudged discount."""
    print("\nMisspecified reliability (assumption 3). Brier at the assumed discount minus Brier when")
    print("the discount is correct, averaged over true alpha in {0.15, 0.30, 0.45, 0.60} and all conflicts:")
    for off in (-0.30, -0.15, 0.15, 0.30):
        out = []
        for k in ("DS", "BayesDisc"):
            diffs = []
            for r in mrows:
                if abs(r["misspec"] - off) < 1e-9:
                    ref = next(q for q in mrows if q["alpha_true"] == r["alpha_true"]
                               and q["conflict"] == r["conflict"] and abs(q["misspec"]) < 1e-9)
                    diffs.append(r[f"brier_{k}"] - ref[f"brier_{k}"])
            out.append(f"{k} {np.mean(diffs):+.3f}")
        print(f"   assumed - true = {off:+.2f}:  " + ",  ".join(out))
    n = len(mrows)
    win = sum(r["dbrier_DS_Bayes"] + r["dbrier_DS_Bayes_ci"] < 0 for r in mrows)
    lose = sum(r["dbrier_DS_Bayes"] - r["dbrier_DS_Bayes_ci"] > 0 for r in mrows)
    print(f"   DS vs plain Bayes over all {n} misspecified and correct cells: "
          f"better in {win}, worse in {lose}, indistinguishable in {n - win - lose}")
    win = sum(r["dbrier_DS_BayesDisc"] + r["dbrier_DS_BayesDisc_ci"] < 0 for r in mrows)
    lose = sum(r["dbrier_DS_BayesDisc"] - r["dbrier_DS_BayesDisc_ci"] > 0 for r in mrows)
    print(f"   DS vs discounted Bayes over the same cells: better in {win}, worse in {lose}, "
          f"indistinguishable in {n - win - lose}")
    for a_true in MISSPEC_TRUE:
        for c in MISSPEC_CONFLICTS:
            sub = [r for r in mrows if r["alpha_true"] == a_true and r["conflict"] == c]
            best = min(sub, key=lambda r: r["brier_DS"])
            print(f"   true alpha {a_true:.2f}, c {c:.1f}: DS Brier is lowest at assumed alpha {best['alpha_assumed']:.2f}")


def headline(rows):
    print("Headline results (mean over trials, 95% CI half width in parentheses)")
    print(f"{'alpha':>5} {'c':>4} | {'acc DS':>14} {'acc Bayes':>14} | "
          f"{'Brier DS':>14} {'Brier Bayes':>14} {'Brier BDisc':>14} | {'mean K':>6}")
    for r in rows:
        if r["alpha"] in (0.0, 0.30, 0.60) and r["conflict"] in (0.0, 0.2, 0.4, 0.5):
            print(f"{r['alpha']:5.2f} {r['conflict']:4.1f} | "
                  f"{r['acc_DS']:.3f} ({r['acc_DS_ci']:.3f}) {r['acc_Bayes']:.3f} ({r['acc_Bayes_ci']:.3f}) | "
                  f"{r['brier_DS']:.3f} ({r['brier_DS_ci']:.3f}) {r['brier_Bayes']:.3f} ({r['brier_Bayes_ci']:.3f}) "
                  f"{r['brier_BayesDisc']:.3f} ({r['brier_BayesDisc_ci']:.3f}) | {r['K_mean']:.3f}")
    print("\nSeparation (paired 95% CI on Brier DS minus plain Bayes excludes zero, DS better):")
    for a in ALPHAS:
        sub = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
        first = next((r["conflict"] for r in sub
                      if r["dbrier_DS_Bayes"] + r["dbrier_DS_Bayes_ci"] < 0), None)
        print(f"   alpha = {a:.2f}: first separating conflict level = {first}")
    print("\nCounter-explanation check, DS versus naive Bayes with the SAME discounting:")
    better = sum(r["dbrier_DS_BayesDisc"] + r["dbrier_DS_BayesDisc_ci"] < 0 for r in rows)
    worse = sum(r["dbrier_DS_BayesDisc"] - r["dbrier_DS_BayesDisc_ci"] > 0 for r in rows)
    print(f"   DS significantly better Brier in {better} of {len(rows)} cells, "
          f"significantly worse in {worse}, indistinguishable in {len(rows) - better - worse}.")
    big = max(rows, key=lambda r: -r["dbrier_DS_BayesDisc"])
    print(f"   Largest DS advantage: {big['dbrier_DS_BayesDisc']:+.3f} "
          f"(+/- {big['dbrier_DS_BayesDisc_ci']:.3f}) at alpha={big['alpha']:.2f}, c={big['conflict']:.1f}")
    acc_gap = max(abs(r["dacc_DS_Bayes"]) for r in rows)
    print(f"   Largest absolute accuracy gap DS vs plain Bayes over all cells: {acc_gap:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 trials per cell")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    trials = args.trials or (200 if args.quick else TRIALS)
    os.makedirs(RESULTS, exist_ok=True)

    worked_trace(seed=args.seed)
    print(f"Running grid: {len(ALPHAS)} x {len(CONFLICTS)} cells, {trials} trials each, seed {args.seed}")
    t0 = time.perf_counter()
    rows = summarise(run_grid(trials=trials, seed=args.seed))
    print(f"Grid finished in {time.perf_counter() - t0:.1f} s\n")
    write_csv(rows, os.path.join(RESULTS, "grid_summary.csv"))
    make_figures(rows)
    headline(rows)

    print(f"\nRunning misspecified discount study: 4 true x 6 assumed x 3 conflict cells, {trials} trials each")
    t0 = time.perf_counter()
    mrows = run_misspec(trials=trials, seed=args.seed)
    print(f"Misspecification study finished in {time.perf_counter() - t0:.1f} s")
    write_csv(mrows, os.path.join(RESULTS, "misspec_summary.csv"))
    make_misspec_figure(mrows)
    misspec_report(mrows)
    print(f"\nWrote grid_summary.csv, misspec_summary.csv and four figures to {RESULTS}")


if __name__ == "__main__":
    main()
