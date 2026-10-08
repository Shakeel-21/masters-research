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


# ==================== SHARED DRAWING ====================
SRC_C = "#8C8C8C"          # original level
GOOD_C, BAD_C = "#55A868", "#C44E52"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _num(r, key, default=0.0):
    try:
        return float(r.get(key, ""))
    except (TypeError, ValueError):
        return default


def _boxes(ax, groups, colours, positions=None, jitter=0.12):
    """Box plots with every data point drawn on top."""
    rng = np.random.default_rng(0)
    data = [np.asarray(g, float) for g in groups]
    positions = positions or list(range(1, len(data) + 1))
    bp = ax.boxplot(data, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.3); patch.set_edgecolor(c)
    for p, g, c in zip(positions, data, colours):
        ax.plot(p + rng.uniform(-jitter, jitter, g.size), g, "o",
                ms=3.5, color=c, alpha=0.85)
    _style(ax)


def level_means(levels, use, metric, scale=1.0):
    """Per-level mean of a metric for Core and Complex (levels with both)."""
    core, comp = [], []
    for lv in use:
        c, x = split(levels[lv]["metrics"])
        a, b = col(c, metric), col(x, metric)
        if a.size and b.size:
            core.append(a.mean() * scale); comp.append(b.mean() * scale)
    return np.array(core), np.array(comp)


def _higher_text(core, comp, word="higher"):
    up, n = int(np.sum(comp > core)), len(core)
    return f"Complex {word} in {up}/{n} levels"


def has_source_counts(levels):
    return any(r.get("source_count") not in (None, "")
               for d in levels.values() for r in d["emergence"])


def is_multi(r):
    return r.get("distinct_tiles") not in (None, "", "1")


def placed(r):
    raw = r.get("solver_placements")
    if raw in (None, ""):
        raw = r.get("complex_exact_count", "0")
    return _num({"v": raw}, "v") > 0


# ==================== FIG 1: reuse in the original level ====================
REPEAT_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10**9, "6+")]


def fig_reuse_in_source(levels, out):
    """How often each macro structure repeats in the level it came from."""
    if not has_source_counts(levels):
        print("skip fig 1: no source_count column - re-run the evaluator")
        return
    multi, slab = Counter(), Counter()
    for d in levels.values():
        for r in d["emergence"]:
            if r.get("source_count") in (None, ""):
                continue
            n = int(_num(r, "source_count"))
            lab = next(l for lo, hi, l in REPEAT_BINS if lo <= n <= hi)
            (multi if is_multi(r) else slab)[lab] += 1
    labs = [l for *_, l in REPEAT_BINS]
    m = np.array([multi[l] for l in labs])
    s = np.array([slab[l] for l in labs])
    rep, total = int(m[2:].sum()), int(m.sum())

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.bar(labs, m, color=COMP_C, label="multi-tile structures")
    ax.bar(labs, s, bottom=m, color="#CCCCCC", label="single-tile slabs")
    for i in range(len(labs)):
        if m[i]:
            ax.text(i, m[i] / 2, str(m[i]), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.axvspan(1.5, len(labs) - 0.5, color=GOOD_C, alpha=0.08, lw=0)
    ax.text(len(labs) - 0.6, ax.get_ylim()[1] * 0.95, "repeated", ha="right",
            va="top", color=GOOD_C, fontsize=9, fontweight="bold")
    ax.set_xlabel("times the structure appears in its original level\n"
                  "(0 = the blueprint shape never appears exactly as built)",
                  fontsize=8.5)
    ax.set_ylabel("number of structures")
    pct = 100 * rep / total if total else 0
    ax.set_title(f"{rep} of {total} multi-tile structures ({pct:.0f}%) repeat "
                 f"in their original level\nrepeated structures are the ones "
                 f"worth storing as a reusable blueprint", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    _style(ax)
    _save(fig, out)


# ==================== FIG 2: structures per level ====================
def fig_structures_per_level(levels, out):
    """Multi-tile macro structures per level: original vs Core vs Complex."""
    if not has_source_counts(levels):
        print("skip fig 2: no source_count column - re-run the evaluator")
        return
    orig, core, comp = [], [], []
    for d in levels.values():
        rows = [r for r in d["emergence"] if is_multi(r)
                and r.get("source_count") not in (None, "")]
        c, x = split(d["metrics"])
        if not rows or not c or not x:
            continue
        orig.append(sum(_num(r, "source_count") for r in rows))
        core.append(sum(_num(r, "core_exact_count") for r in rows) / len(c))
        comp.append(sum(_num(r, "complex_exact_count") for r in rows) / len(x))
    if not orig:
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _boxes(ax, [orig, core, comp], [SRC_C, CORE_C, COMP_C])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Original\nlevel", "Core\ngenerations", "Complex\ngenerations"])
    ax.set_ylabel("multi-tile macro structures per level")
    ax.set_title(f"Macro structures per level (median over {len(orig)} levels)\n"
                 f"original {np.median(orig):.0f}   |   Core {np.median(core):.1f}"
                 f"   |   Complex {np.median(comp):.1f}", fontsize=10)
    _save(fig, out)


# ==================== FIG 3: reproduced by size ====================
def fig_reproduced_by_size(levels, out):
    """Share of multi-tile structures that appear at least once in the output."""
    S = structures(levels, multi=True)
    if not S:
        return
    labs = [l for _, l in SIZE_BANDS if any(band(s[0]) == l for s in S)]
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ticks = []
    for i, l in enumerate(labs):
        b = [s for s in S if band(s[0]) == l]
        n = len(b)
        ticks.append(f"{l} cells\n(n={n})")
        for off, idx, c in ((-w / 2, 1, CORE_C), (w / 2, 2, COMP_C)):
            pct = 100 * sum(s[idx] for s in b) / n
            ax.bar(i + off, pct, w, color=c)
            ax.text(i + off, pct + 1.5, f"{pct:.0f}%", ha="center", fontsize=8)
    n = len(S)
    pc = 100 * sum(s[1] for s in S) / n
    px = 100 * sum(s[2] for s in S) / n
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_xlabel("structure size")
    ax.set_ylabel("% of structures that appear")
    ax.set_ylim(0, 110); ax.set_yticks(range(0, 101, 20))
    ax.legend(handles=[Patch(color=CORE_C, label="Core"),
                       Patch(color=COMP_C, label="Complex")],
              frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title(f"Multi-tile structures that appear in the generated levels\n"
                 f"overall: Core {pc:.0f}%  vs  Complex {px:.0f}%", fontsize=10)
    _style(ax)
    _save(fig, out)


# ==================== FIG 4: placement rate ====================
def fig_placement_rate(levels, out):
    """Per level: share of declared macros the solver stamped at least once."""
    rows = [(lv, sum(placed(r) for r in d["emergence"]), len(d["emergence"]))
            for lv, d in levels.items() if d["emergence"]]
    if not rows:
        return
    rows.sort(key=lambda r: -r[1] / r[2])
    k, n = sum(r[1] for r in rows), sum(r[2] for r in rows)
    pct = [100 * a / b for _, a, b in rows]

    fig, ax = plt.subplots(figsize=(6.6, 0.22 * len(rows) + 1.6))
    ax.barh(range(len(rows)), pct, height=0.75, color=COMP_C)
    for i, (_, a, b) in enumerate(rows):
        ax.text(pct[i] + 1, i, f"{a}/{b}", va="center", fontsize=7)
    ax.axvline(100 * k / n, color="black", ls="--", lw=1,
               label=f"all levels: {100 * k / n:.0f}%")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r[0]) for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 112); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("% of declared macros placed at least once")
    ax.set_title(f"The solver placed {k} of {n} declared macros "
                 f"({100 * k / n:.0f}%) at least once", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, out)


# ==================== FIG 5: why macros are not placed ====================
def fig_why_not_placed(levels, out):
    tests = [("Gaps inside the shape", "interior_holes",
              lambda v: v != "0", "has gaps", "no gaps"),
             ("Adjacency rules", "start_invariant",
              lambda v: v == "False", "contradictory", "consistent")]
    panels = []
    for title, key, is_bad, bad_lab, good_lab in tests:
        g = {"bad": [0, 0], "good": [0, 0]}
        for d in levels.values():
            for r in d["emergence"]:
                v = r.get(key)
                if v in (None, "", "None"):
                    continue
                side = "bad" if is_bad(v) else "good"
                g[side][0] += 1; g[side][1] += placed(r)
        if g["bad"][0] or g["good"][0]:
            panels.append((title, [(bad_lab, *g["bad"], BAD_C),
                                   (good_lab, *g["good"], GOOD_C)]))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             squeeze=False, sharey=True)
    for ax, (title, bars) in zip(axes[0], panels):
        for i, (lab, n, k, c) in enumerate(bars):
            pct = 100 * k / n if n else 0
            ax.bar(i, pct, 0.6, color=c)
            ax.text(i, pct + 2, f"{k}/{n}", ha="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars])
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 110)
        _style(ax)
    axes[0, 0].set_ylabel("% of macros placed at least once")
    fig.suptitle("What stops the solver placing a macro?", fontsize=10.5)
    _save(fig, out)


# ==================== FIG 6: level features, Core vs Complex ====================
FEATURES = [("Components", "Separate structures", 1),
            ("Avg Plat Width", "Platform width (tiles)", 1),
            ("Avg Plat Thickness", "Platform thickness (tiles)", 1),
            ("Hazard Density /10col", "Hazards per 10 columns", 1),
            ("Reachable %", "Reachable %", 1),
            ("Completable", "Completable (%)", 100)]


def fig_features(levels, out, use=None):
    use = use or sorted(levels)
    panels = [(lab, *level_means(levels, use, m, s)) for m, lab, s in FEATURES]
    panels = [p for p in panels if p[1].size]
    if not panels:
        return
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4, 3.0 * nrow), squeeze=False)
    for ax, (lab, c, x) in zip(axes.flat, panels):
        _boxes(ax, [c, x], [CORE_C, COMP_C])
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Core", "Complex"])
        ax.set_title(f"{lab}\n{_higher_text(c, x)}", fontsize=9)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Core vs Complex level features\n"
                 "one dot = one source level, averaged over its generations",
                 fontsize=10.5)
    _save(fig, out)


# ==================== FIG 7: similarity to the original ====================
PATTERNS = [("KL-Div (1x1)", "single tiles"), ("KL-Div (2x2)", "2x2 blocks"),
            ("KL-Div (3x3)", "3x3 blocks")]


def fig_similarity(levels, out, use=None):
    use = use or sorted(levels)
    groups = []
    for m, lab in PATTERNS:
        c, x = level_means(levels, use, m)
        if c.size:
            groups.append((lab, np.maximum(c, 1e-6), np.maximum(x, 1e-6)))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(1.0 + 2.2 * len(groups), 4.0))
    ticks, notes = [], []
    for i, (lab, c, x) in enumerate(groups):
        base = 3 * i
        _boxes(ax, [c, x], [CORE_C, COMP_C], positions=[base + 1, base + 2])
        ticks.append((base + 1.5, lab))
        notes.append(f"{lab}: {int(np.sum(x < c))}/{len(c)}")
    ax.set_yscale("log")
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks])
    ax.set_xlabel("pattern size compared")
    ax.set_ylabel("difference from original (KL)\nlower = closer")
    ax.legend(handles=[Patch(color=CORE_C, alpha=0.6, label="Core"),
                       Patch(color=COMP_C, alpha=0.6, label="Complex")],
              frameon=False, fontsize=8.5)
    ax.set_title("How closely the generated levels match the original's patterns\n"
                 "Complex closer in " + ",  ".join(notes) + " levels", fontsize=9.5)
    _save(fig, out)


# ==================== FIG 8: variety between generations ====================
SPREAD = [("Linearity", "Ground linearity"),
          ("Elevation Std", "Ground height variation"),
          ("Solid Density", "Solid density"),
          ("Components", "Separate structures"),
          ("Avg Plat Width", "Platform width"),
          ("Hazard Density /10col", "Hazards per 10 columns")]


def _iqr(a):
    return float(np.subtract(*np.percentile(a, [75, 25]))) if a.size >= 3 else np.nan


def fig_variety(levels, out, use=None):
    """Spread of the levels generated from ONE source, Core against Complex."""
    use = use or sorted(levels)
    panels = []
    for m, lab in SPREAD:
        c_out, x_out = [], []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            ic, ix = _iqr(col(c, m)), _iqr(col(x, m))
            if not (np.isnan(ic) or np.isnan(ix)):
                c_out.append(ic); x_out.append(ix)
        if c_out:
            panels.append((lab, np.array(c_out), np.array(x_out)))
    if not panels:
        return
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4, 3.0 * nrow), squeeze=False)
    for ax, (lab, c, x) in zip(axes.flat, panels):
        _boxes(ax, [c, x], [CORE_C, COMP_C])
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Core", "Complex"])
        ax.set_title(f"{lab}\n{_higher_text(c, x, 'more varied')}", fontsize=9)
    axes[0, 0].set_ylabel("spread between generations (IQR)")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("How varied are the levels generated from one source?\n"
                 "one dot = one source level; higher = more variety", fontsize=10.5)
    _save(fig, out)


# ==================== FIG 9: generation time ====================
def fig_runtime(levels, out, use=None):
    use = use or sorted(levels)
    c, x = level_means(levels, use, "Time (s)")
    if not c.size:
        return
    ratio = np.median(x / np.maximum(c, 1e-9))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    _boxes(ax, [c, x], [CORE_C, COMP_C])
    ax.set_yscale("log")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Core", "Complex"])
    ax.set_ylabel("mean generation time per level (s)")
    ax.set_title(f"Complex takes {ratio:.1f}x as long as Core\n"
                 f"(median over {len(c)} levels)", fontsize=10)
    _save(fig, out)


# ==================== FIG 10: generation failures ====================
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
    ax.axvline(lim, color="k", ls=":", lw=1, label=f"excluded beyond {lim:g}")
    ax.set_xlim(0, requested)
    ax.set_xlabel(f"failed generations (of {requested} requested)")
    ax.set_title(f"Failed generations: Core {sum(r[1] for r in rows)}, Complex "
                 f"{sum(r[2] for r in rows)} (of {requested * len(rows)} each)\n"
                 f"{len(rows) - len(hit)} of {len(rows)} levels had no failures "
                 "and are not shown", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, out)


def placement_size_table(levels):
    """Numbers behind fig 3, with the significance test, for the text."""
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
    # the macro story: worth reusing -> Complex keeps it -> how often it's placed
    fig_reuse_in_source(levels, p("1_reuse_in_original.png"))
    fig_structures_per_level(levels, p("2_structures_per_level.png"))
    fig_reproduced_by_size(levels, p("3_reproduced_by_size.png"))
    fig_placement_rate(levels, p("4_placement_rate.png"))
    fig_why_not_placed(levels, p("5_why_not_placed.png"))
    # the level-quality story: what Complex changes in the output
    fig_features(levels, p("6_level_features.png"), use=good)
    fig_similarity(levels, p("7_similarity_to_original.png"), use=good)
    fig_variety(levels, p("8_variety.png"), use=good)
    fig_runtime(levels, p("9_generation_time.png"), use=good)
    fig_failures(levels, p("10_generation_failures.png"), requested)
    placement_size_table(levels)
    print(f"\nwrote figures to {figs}")


if __name__ == "__main__":
    main()


#python demo/plot.py Generation/Baselines/all_levels