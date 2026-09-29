"""
plot_results.py
===============
Turn the per-level evaluation CSVs into figures.

    python plot_results.py <results_folder> [<figures_folder>]

Reads every  <level>_new_evaluation_metrics_raw.csv  and
              <level>_complex_structure_emergence.csv
in the results folder and writes one PNG per figure.

Levels where a batch lost more than QUALITY_MIN of its generations are
excluded from the metric comparisons - when most generations fail the
survivors are the easy cases, so their means are not comparable. They are
still shown in the failure figure, which is where they belong.

Complex counts a structure as reproduced when it appears in the output by
any route, stamped or emergent (complex_exact_count > 0). Counting stamps
alone undercounts Complex.
"""

import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, wilcoxon

QUALITY_MIN = 0.70
SIZE_BANDS = [(2, "1-2"), (4, "3-4"), (9, "5-9"), (16, "10-16"), (10**9, "17+")]
CORE_C, COMP_C = "#4C72B0", "#DD8452"


def load(folder):
    levels = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_new_evaluation_metrics_raw.csv"))):
        stem = os.path.basename(path)[:-len("_new_evaluation_metrics_raw.csv")]
        rows = list(csv.DictReader(open(path)))
        rows = [r for r in rows if r.get("Solidity") in (None, "", "non_background")] or rows
        emg_path = os.path.join(folder, f"{stem}_complex_structure_emergence.csv")
        emg = list(csv.DictReader(open(emg_path))) if os.path.exists(emg_path) else []
        # a macro declared twice is one structure, not two
        emg = [r for r in emg if r.get("duplicate_of") in (None, "", "0")]
        levels[stem] = {"metrics": rows, "emergence": emg}
    return levels


def split(rows):
    return ([r for r in rows if r["Batch"] == "Core"],
            [r for r in rows if r["Batch"] == "Complex"])


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "nan", None):
            try: out.append(float(v))
            except ValueError: pass
    return np.array(out)


def infer_requested(levels):
    """
    How many generations were asked for per batch. Taken as the largest batch
    seen across all levels, since at least one batch somewhere usually
    completes. Override with the command line if that is not true of your run.
    """
    n = 0
    for d in levels.values():
        c, x = split(d["metrics"])
        n = max(n, len(c), len(x))
    return n or 1


def usable(levels, requested):
    """
    Levels where BOTH batches retained at least QUALITY_MIN of the requested
    generations. Measured against the request, not against each other, so a
    level that failed heavily in both batches is still excluded - when most
    generations fail the survivors are the easy cases and their means are not
    comparable to a complete batch.
    """
    ok = []
    for lv, d in levels.items():
        c, x = split(d["metrics"])
        if min(len(c), len(x)) >= QUALITY_MIN * requested:
            ok.append(lv)
    return sorted(ok)


def wilson(k, n, z=1.96):
    """Wilson score interval - honest error bars on a proportion at small n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*p, 100*max(0.0, c-h), 100*min(1.0, c+h)


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


# ==================== NEW HELPERS ====================
def short(lv):
    return (lv.replace("SuperMarioBros2_J_-World", "SMB2J ")
              .replace("SuperMarioBros2(J)-World", "SMB2J "))


def band(n):
    return next(l for lim, l in SIZE_BANDS if n <= lim)


def structures(levels, multi=None):
    """
    (size, core_yes, complex_yes, placed) for every declared structure.
    complex_yes counts a structure that appears in Complex output by any
    route, stamped or emergent; counting stamps alone undercounts Complex.
    """
    out = []
    for d in levels.values():
        for r in d["emergence"]:
            if not r.get("specified_cells") or r.get("distinct_tiles") in (None, ""):
                continue
            if multi is not None and (r["distinct_tiles"] != "1") != multi:
                continue
            out.append((int(r["specified_cells"]),
                        int(r["core_emerges"]) > 0,
                        int(r.get("complex_exact_count") or 0) > 0,
                        int(r.get("solver_placements") or 0) > 0))
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: only the pairs where Core and Complex disagree."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals):
    p = np.asarray(pvals, float)
    out, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def rank_biserial(d):
    """Paired effect size in [-1, 1]; matches the Wilcoxon signed-rank test."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if d.size < 2 or not np.any(d != 0):
        return 1.0
    try:
        return wilcoxon(d).pvalue
    except ValueError:
        return 1.0


def boot_diff(a, b, n_boot=2000, seed=0):
    """mean(b) - mean(a) with a bootstrap 95% CI; runs are independent."""
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    diff = (rng.choice(b, (n_boot, b.size)).mean(1)
            - rng.choice(a, (n_boot, a.size)).mean(1))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return b.mean() - a.mean(), lo, hi


def fit_logistic(x, y):
    """Maximum-likelihood logistic on size. Returns (x0, k) for _logistic."""
    if x.size < 8 or len(set(y)) < 2:
        return None

    def nll(b):
        z = np.clip(b[0] + b[1] * x, -30, 30)
        return np.sum(np.log1p(np.exp(z)) - y * z)

    b0, b1 = minimize(nll, [0.0, -0.1], method="BFGS").x
    return (-b0 / b1, b1) if b1 != 0 else None


def p_text(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


# ==================== FIG 1 ====================
def fig_emergence_size(levels, out):
    """Multi-tile structures: Core by chance vs Complex by any route, paired."""
    S = structures(levels, multi=True)
    labs = [l for _, l in SIZE_BANDS if any(band(s[0]) == l for s in S)]
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, l in enumerate(labs):
        b = [s for s in S if band(s[0]) == l]
        n, top = len(b), 0.0
        for off, idx, c in ((-w/2, 1, "#C44E52"), (w/2, 2, "#4C72B0")):
            k = sum(s[idx] for s in b)
            m, lo, hi = wilson(k, n)
            ax.bar(i + off, m, w, color=c, yerr=[[m - lo], [hi - m]], capsize=3,
                   error_kw=dict(lw=1, ecolor="#444444"))
            ax.text(i + off, hi + 2, f"{k}/{n}", ha="center", fontsize=7.5)
            top = max(top, hi)
        core_only = sum(s[1] and not s[2] for s in b)
        cplx_only = sum(s[2] and not s[1] for s in b)
        p = mcnemar_exact(core_only, cplx_only)
        ax.text(i, top + 10, f"{cplx_only} vs {core_only}\n{p_text(p)}", ha="center",
                fontsize=7.5, fontweight="bold" if p < 0.05 else "normal")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_xlabel("structure size (specified cells)")
    ax.set_ylabel("% of structures reproduced")
    ax.set_ylim(0, 132); ax.set_yticks(range(0, 101, 20))
    ax.legend(handles=[Patch(color="#C44E52", label="Core (by chance)"),
                       Patch(color="#4C72B0", label="Complex (stamped or emergent)")],
              frameon=False, loc="upper right", fontsize=8)
    ax.set_title("Multi-tile structures reproduced, by size\n"
                 "bars: 95% Wilson intervals; above: Complex-only vs Core-only "
                 "structures, exact McNemar", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 2 ====================
def fig_emergence_curves(levels, out):
    """Old figs 8 and 10 on one axis: fitted reproduction rate against size."""
    multi, slab = structures(levels, True), structures(levels, False)
    series = [(multi, 1, "#C44E52", "-", "Core, multi-tile (chance)"),
              (slab, 1, "#8C8C8C", "--", "Core, single-tile slabs (control)"),
              (multi, 2, "#4C72B0", "-", "Complex, multi-tile")]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for S, idx, c, ls, name in series:
        if not S:
            continue
        xs = np.array([s[0] for s in S], float)
        ys = np.array([s[idx] for s in S], float)
        tag = ""
        fit = fit_logistic(xs, ys)
        if fit:
            x0, k = fit
            grid = np.linspace(1, xs.max(), 300)
            ax.plot(grid, 100 * _logistic(grid, x0, k), color=c, ls=ls, lw=2)
            if k < 0 and 0 < x0 < xs.max():
                ax.axvline(x0, color=c, ls=":", lw=1)
                tag = f", 50% at {x0:.1f} cells"
        for _, l in SIZE_BANDS:                 # binned rates, to judge the fit
            b = [(s[0], s[idx]) for s in S if band(s[0]) == l]
            if len(b) >= 3:
                ax.plot(np.median([v[0] for v in b]), 100 * np.mean([v[1] for v in b]),
                        "o", color=c, alpha=0.6, ms=4 + min(len(b), 60) / 10)
        ax.plot([], [], color=c, ls=ls, lw=2,
                label=f"{name}  ({int(ys.sum())}/{ys.size}{tag})")
    ax.axhline(50, color="#BBBBBB", lw=0.8)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("structure size (specified cells)")
    ax.set_ylabel("% reproduced")
    ax.set_title("Reproduction against structure size\n"
                 "lines: maximum-likelihood logistic fit; dots: binned rate, sized by n",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 3 (unchanged) ====================
def fig_partial(levels, out, k=2):
    """
    For each structure, the fraction of its kxk sub-blocks that occur in Core,
    split by whether the whole structure ever did. Structures at coverage 1.0
    that were never reproduced are the strong case: Core has every local piece
    and every local combination, and still never assembles the whole.
    """
    key, nkey = f"core_cov_{k}x{k}", f"n_sub_{k}x{k}"
    em, no = [], []
    for d in levels.values():
        for r in d["emergence"]:
            v = r.get(key, "")
            if v in ("", None) or int(r.get(nkey) or 0) == 0:
                continue
            (em if r["core_emerges"] == "1" else no).append(float(v))
    if not em and not no:
        return
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist([no, em], bins=bins, stacked=True, rwidth=0.9,
            color=["#C44E52", "#8C8C8C"],
            label=[f"structure NOT reproduced (n={len(no)})",
                   f"structure reproduced (n={len(em)})"])
    full_not = sum(1 for v in no if v >= 0.999)
    ax.set_xlabel(f"fraction of the structure's {k}x{k} sub-blocks present in Core")
    ax.set_ylabel("structures")
    ax.set_title("Local pieces available vs whole structure reproduced")
    if full_not:
        ax.annotate(f"{full_not} structures: every {k}x{k} piece\npresent, "
                    f"never assembled",
                    xy=(0.98, ax.get_ylim()[1]*0.55), xytext=(0.55, ax.get_ylim()[1]*0.8),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=1))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 4 ====================
def fig_overlap(levels, out):
    order = ["both", "complex_only", "core_only", "neither"]
    colours = {"both": "#8C8C8C", "complex_only": "#55A868",
               "core_only": "#4C72B0", "neither": "#C44E52"}
    counts = {lv: Counter(r.get("where", "") for r in d["emergence"])
              for lv, d in levels.items() if d["emergence"]}

    def frac(c):
        v = np.array([c[k] for k in order], float)
        return 100 * v / max(1.0, v.sum())

    names = sorted(counts, key=lambda lv: (frac(counts[lv])[3], -frac(counts[lv])[0]))
    pooled = sum(counts.values(), Counter())
    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(7.4, 1.4 + 0.19 * len(names)),
        gridspec_kw={"height_ratios": [1.3, len(names)], "hspace": 0.15})
    for ax, rows, labels in ((a0, [pooled], ["all levels"]),
                             (a1, [counts[lv] for lv in names],
                              [short(lv) for lv in names])):
        F = np.array([frac(c) for c in rows])
        left = np.zeros(len(rows))
        for j, k in enumerate(order):
            ax.barh(range(len(rows)), F[:, j], left=left, height=0.8,
                    color=colours[k], label=k.replace("_", " "))
            left += F[:, j]
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 100); ax.invert_yaxis()
        for i, c in enumerate(rows):
            ax.text(101, i, f"n={sum(c[k] for k in order)}", va="center", fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    a0.set_xticks([])
    a0.legend(frameon=False, ncol=4, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    a1.set_xlabel("% of declared macro structures")
    a0.set_title("Where each declared structure appears\n"
                 "levels sorted by % reproduced by neither", fontsize=10, pad=22)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


# ==================== FIG 5 ====================
def fig_placement(levels, out):
    """Placed = the solver stamped the macro at least once."""
    groups = {"interior holes": [0, 0], "no interior holes": [0, 0],
              "contradictory rules": [0, 0], "consistent rules": [0, 0]}
    for d in levels.values():
        for r in d["emergence"]:
            raw = r.get("solver_placements")
            if raw in (None, ""):
                raw = r.get("complex_exact_count", "0")
            placed = int(raw or 0) > 0
            if r.get("interior_holes") not in (None, ""):
                k = "interior holes" if r["interior_holes"] != "0" else "no interior holes"
                groups[k][0] += 1; groups[k][1] += placed
            if r.get("start_invariant") in ("True", "False"):
                k = ("contradictory rules" if r["start_invariant"] == "False"
                     else "consistent rules")
                groups[k][0] += 1; groups[k][1] += placed
    labs = [k for k in groups if groups[k][0]]
    pct = [100 * groups[k][1] / groups[k][0] for k in labs]
    colour = {"interior holes": "#C44E52", "contradictory rules": "#C44E52"}
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.barh(range(len(labs)), pct, height=0.7,
            color=[colour.get(k, "#55A868") for k in labs])
    for i, k in enumerate(labs):
        ax.text(pct[i] + 1.5, i, f"{groups[k][1]}/{groups[k][0]}", va="center",
                fontsize=8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("% of macros placed at least once")
    ax.set_title("Which macros the solver can place", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 6 ====================
FOREST = [("Time (s)", "Time (s)"), ("KL-Div (1x1)", "KL 1x1"),
          ("KL-Div (2x2)", "KL 2x2"), ("KL-Div (3x3)", "KL 3x3"),
          ("Components", "Components"), ("Avg Plat Thickness", "Plat thickness"),
          ("Hazard Density /10col", "Hazards /10col"), ("Completable", "Completable")]


def fig_forest(levels, out, use=None):
    """Complex minus Core per level, one panel per metric."""
    use = sorted(use or levels, key=short)
    panels = []
    for m, title in FOREST:
        per, diffs = [], []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            d = boot_diff(col(c, m), col(x, m))
            per.append(d)
            if not np.isnan(d[0]):
                diffs.append(d[0])
        if diffs:
            panels.append((title, per, diffs, wilcoxon_p(diffs)))
    if not panels:
        return
    p_adj = holm([p for *_, p in panels])
    ncol = 4
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, sharey=True, squeeze=False,
                             figsize=(8.4, 1.0 + nrow * (0.9 + 0.14 * len(use))))
    for j, ((title, per, diffs, _), pa) in enumerate(zip(panels, p_adj)):
        ax = axes[j // ncol, j % ncol]
        for i, (d, lo, hi) in enumerate(per):
            if np.isnan(d):
                continue
            shade = "#333333" if (lo > 0 or hi < 0) else "#BBBBBB"
            ax.plot([lo, hi], [i, i], color=shade, lw=1)
            ax.plot(d, i, "o", ms=3, color=shade)
        ax.axvline(0, color="#C44E52", lw=0.8)
        ax.set_title(f"{title}\nmedian {np.median(diffs):+.3g}   "
                     f"r={rank_biserial(diffs):+.2f}\n{p_text(pa)}", fontsize=7.5,
                     fontweight="bold" if pa < 0.05 else "normal")
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    axes[0, 0].set_yticks(range(len(use)))
    axes[0, 0].set_yticklabels([short(lv) for lv in use])
    axes[0, 0].invert_yaxis()
    fig.suptitle("Complex minus Core, per level  (dark: 95% CI excludes 0; "
                 "p: Wilcoxon over levels, Holm-corrected; r: rank-biserial)",
                 fontsize=8.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 7 ====================
def fig_kl(levels, out, use=None):
    """KL(generated || source) at each window size, per-level means paired."""
    use = use or sorted(levels)
    panels = []
    for m in ("KL-Div (1x1)", "KL-Div (2x2)", "KL-Div (3x3)"):
        pairs = []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            a, b = col(c, m), col(x, m)
            if a.size and b.size:
                pairs.append((max(a.mean(), 1e-6), max(b.mean(), 1e-6)))
        if pairs:
            d = [b - a for a, b in pairs]
            panels.append((m, pairs, d, wilcoxon_p(d)))
    if not panels:
        return
    p_adj = holm([p for *_, p in panels])
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(2.6 * len(panels) + 1.2, 3.4))
    for ax, (m, pairs, d, _), pa in zip(axes[0], panels, p_adj):
        for a, b in pairs:
            ax.plot([0, 1], [a, b], "-o", ms=3, lw=1, alpha=0.6,
                    color="#55A868" if b < a else "#C44E52")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Core", "Complex"])
        ax.set_xlim(-0.3, 1.3); ax.set_yscale("log")
        lower = sum(1 for v in d if v < 0)
        ax.set_title(f"{m[8:-1]} patterns\nComplex lower in {lower}/{len(d)}, "
                     f"{p_text(pa)}", fontsize=8.5,
                     fontweight="bold" if pa < 0.05 else "normal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("KL(generated || source), per-level mean")
    note = ("" if len(panels) == 3 else
            "\n2x2/3x3 absent: re-run the learner to write source_grid.json")
    fig.suptitle("Distance from the source level's tile patterns, by window size\n"
                 "green: Complex closer; red: Complex further; "
                 f"p: Wilcoxon over levels, Holm across sizes{note}", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 8 ====================
def fig_runtime(levels, out, use=None):
    use = use or sorted(levels)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    if any(col(levels[lv]["metrics"], "Vocab Size").size for lv in use):
        allx, ally = [], []
        for batch, c in (("Core", CORE_C), ("Complex", COMP_C)):
            xs, ys = [], []
            for lv in use:
                rows = [r for r in levels[lv]["metrics"] if r["Batch"] == batch]
                v, t = col(rows, "Vocab Size"), col(rows, "Time (s)")
                if v.size and t.size and v[0] > 0:
                    xs.append(v[0]); ys.append(t.mean())
            ax.scatter(xs, ys, s=20, color=c, alpha=0.85, label=batch)
            allx += xs; ally += ys
        x, y = np.array(allx), np.array(ally)
        if x.size >= 3:
            k, c0 = np.polyfit(np.log(x), np.log(y), 1)
            g = np.geomspace(x.min(), x.max(), 50)
            ax.plot(g, np.exp(c0) * g ** k, "--", color="#333333", lw=1,
                    label=f"fit: time ~ vocab^{k:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("vocabulary size (tile ids the solver chooses among)")
        ax.set_ylabel("mean generation time (s)")
        ax.set_title("Generation time against vocabulary size", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    else:
        xs, ys = [], []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            tc, tx = col(c, "Time (s)"), col(x, "Time (s)")
            if tc.size and tx.size and levels[lv]["emergence"]:
                xs.append(len(levels[lv]["emergence"])); ys.append(tx.mean() / tc.mean())
        ax.scatter(xs, ys, color=COMP_C, s=22)
        ax.axhline(1, color="#BBBBBB", lw=0.8); ax.set_yscale("log")
        ax.set_xlabel("declared macros in the level")
        ax.set_ylabel("Complex / Core generation time")
        ax.set_title("Generation cost against macro vocabulary\n(Vocab Size column "
                     "absent: re-run the evaluator for the full version)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 9 ====================
SPREAD = ["Linearity", "Hazard Density /10col", "Solid Density", "Elevation Std",
          "Components", "Avg Plat Width"]


def _ellipse_area(rows, xm, ym):
    pts = [(float(r[xm]), float(r[ym])) for r in rows
           if r.get(xm) not in ("", "nan", None) and r.get(ym) not in ("", "nan", None)]
    if len(pts) < 3:
        return np.nan
    det = np.linalg.det(np.cov(np.array(pts).T))
    return np.pi * np.sqrt(det) if det > 0 else np.nan


def _iqr(a):
    return float(np.subtract(*np.percentile(a, [75, 25]))) if a.size >= 3 else np.nan


def fig_expressive(levels, out, xm="Linearity", ym="Hazard Density /10col", use=None):
    """
    Spread of the levels generated from ONE source, Core against Complex.
    Pooling every source into one cloud shows how different the source levels
    are from each other, not anything the generator does.
    """
    use = use or sorted(levels)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    pts = []
    for lv in use:
        c, x = split(levels[lv]["metrics"])
        ac, axx = _ellipse_area(c, xm, ym), _ellipse_area(x, xm, ym)
        if ac > 0 and axx > 0:
            pts.append((ac, axx))
    if pts:
        c, x = np.array(pts).T
        lim = [min(c.min(), x.min()) * 0.7, max(c.max(), x.max()) * 1.4]
        a0.scatter(c, x, color=COMP_C, s=22)
        a0.plot(lim, lim, color="#BBBBBB", lw=0.8)
        a0.set_xscale("log"); a0.set_yscale("log"); a0.set_xlim(lim); a0.set_ylim(lim)
        lr = np.log(x / c)
        a0.set_title(f"{xm} x {ym.replace(' /10col', '')} spread\nComplex narrower "
                     f"in {int((lr < 0).sum())}/{lr.size} levels, "
                     f"{p_text(wilcoxon_p(lr))}", fontsize=8.5)
    a0.set_xlabel("Core: area of 1-sigma ellipse")
    a0.set_ylabel("Complex: area of 1-sigma ellipse")
    a0.spines[["top", "right"]].set_visible(False)

    data, labels = [], []
    for m in SPREAD:
        r = []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            ic, ix = _iqr(col(c, m)), _iqr(col(x, m))
            if ic > 0 and ix > 0:
                r.append(np.log2(ix / ic))
        if r:
            data.append(r); labels.append(m.replace(" /10col", ""))
    if data:
        a1.boxplot(data, vert=False, widths=0.5, showfliers=False, orientation="horizontal")
        rng = np.random.default_rng(0)
        for i, r in enumerate(data, 1):
            a1.plot(r, i + rng.uniform(-0.12, 0.12, len(r)), "o", ms=2.5,
                    color=COMP_C, alpha=0.6)
        a1.axvline(0, color="#C44E52", lw=0.8)
        a1.set_yticks(range(1, len(labels) + 1)); a1.set_yticklabels(labels, fontsize=8)
        a1.set_xlabel("log2( Complex IQR / Core IQR )   <0: Complex narrower")
        a1.set_title("Per-metric spread, one point per level", fontsize=8.5)
    a1.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does Complex narrow the range of levels generated from one source?",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ==================== FIG 10 ====================
def fig_failures(levels, out, requested):
    """Only levels that lost at least one generation; the rest are counted."""
    rows = []
    for lv in sorted(levels, key=short):
        c, x = split(levels[lv]["metrics"])
        rows.append((lv, requested - len(c), requested - len(x)))
    hit = [r for r in rows if r[1] or r[2]]
    lim = requested * (1 - QUALITY_MIN)
    fig, ax = plt.subplots(figsize=(6.0, 0.24 * max(len(hit), 1) + 1.4))
    if hit:
        y = np.arange(len(hit)); h = 0.4
        ax.barh(y - h/2, [r[1] for r in hit], h, color=CORE_C, label="Core")
        ax.barh(y + h/2, [r[2] for r in hit], h, color=COMP_C, label="Complex")
        ax.set_yticks(y); ax.set_yticklabels([short(r[0]) for r in hit], fontsize=7.5)
        ax.invert_yaxis()
    ax.axvline(lim, color="k", ls=":", lw=1, label=f"exclusion threshold ({lim:g})")
    ax.set_xlim(0, requested)
    ax.set_xlabel(f"failed generations (of {requested} requested)")
    ax.set_title(f"Generation failures: Core {sum(r[1] for r in rows)}, Complex "
                 f"{sum(r[2] for r in rows)} of {requested * len(rows)} each\n"
                 f"{len(rows) - len(hit)} of {len(rows)} levels had none and "
                 "are not shown", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def placement_size_table(levels):
    """Numbers behind fig 1, for the text."""
    S = structures(levels, multi=True)
    print(f"\n{'size':>7}{'n':>5}{'Core':>15}{'Complex':>15}{'stamped':>9}"
          f"{'Core-only':>11}{'Cplx-only':>11}{'McNemar':>10}")
    for _, lab in SIZE_BANDS + [(None, "all")]:
        b = S if lab == "all" else [s for s in S if band(s[0]) == lab]
        if not b:
            continue
        n = len(b)
        e, a, st = (sum(s[i] for s in b) for i in (1, 2, 3))
        co = sum(s[1] and not s[2] for s in b)
        xo = sum(s[2] and not s[1] for s in b)
        print(f"{lab:>7}{n:>5}{f'{e}/{n} ({100*e/n:.0f}%)':>15}"
              f"{f'{a}/{n} ({100*a/n:.0f}%)':>15}{st:>9}{co:>11}{xo:>11}"
              f"{mcnemar_exact(co, xo):>10.4f}")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    figs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(folder, "figures")
    os.makedirs(figs, exist_ok=True)
    levels = load(folder)
    if not levels:
        print(f"No *_new_evaluation_metrics_raw.csv found in {folder}")
        return
    requested = int(sys.argv[3]) if len(sys.argv) > 3 else infer_requested(levels)
    good = usable(levels, requested)
    print(f"{len(levels)} levels loaded; {len(good)} usable for metric comparison")
    for lv in sorted(levels):
        c, x = split(levels[lv]["metrics"])
        print(f"   {lv:<28} core {len(c):>3}/{requested}  complex {len(x):>3}/{requested}"
              f"   {'' if lv in good else '<- excluded from metrics'}")

    p = lambda n: os.path.join(figs, n)
    fig_emergence_size(levels, p("1_emergence_by_size.png"))
    fig_emergence_curves(levels, p("2_emergence_curves.png"))
    fig_partial(levels, p("3_partial_emergence.png"), k=2)
    fig_overlap(levels, p("4_structure_overlap.png"))
    fig_placement(levels, p("5_placement_barriers.png"))
    fig_forest(levels, p("6_level_differences.png"), use=good)
    fig_kl(levels, p("7_kl_by_pattern_size.png"), use=good)
    fig_runtime(levels, p("8_runtime_vs_vocab.png"), use=good)
    fig_expressive(levels, p("9_expressive_spread.png"), use=good)
    fig_failures(levels, p("10_generation_failures.png"), requested)
    placement_size_table(levels)
    print(f"\nwrote 10 figures to {figs}")


if __name__ == "__main__":
    main()


#python demo/plot.py Generation/Baselines/all_levels