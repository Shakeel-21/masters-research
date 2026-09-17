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
"""

import csv
import glob
import os
import sys
from collections import Counter, defaultdict
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import mannwhitneyu
 
QUALITY_MIN = 0.70          # a batch must retain this fraction to be comparable
SIZE_BANDS = [(2, "1-2"), (4, "3-4"), (9, "5-9"), (16, "10-16"), (10**9, "17+")]
 
CORE_C, COMP_C = "#4C72B0", "#DD8452"
 
 
# --------------------------------------------------------------------------
def load(folder):
    levels = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_new_evaluation_metrics_raw.csv"))):
        stem = os.path.basename(path)[:-len("_new_evaluation_metrics_raw.csv")]
        rows = list(csv.DictReader(open(path)))
        rows = [r for r in rows if r.get("Solidity") in (None, "", "non_background")] or rows
        emg_path = os.path.join(folder, f"{stem}_complex_structure_emergence.csv")
        emg = list(csv.DictReader(open(emg_path))) if os.path.exists(emg_path) else []
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
 
 
def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]; matches the Mann-Whitney test."""
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = sum((b[:, None] > a[None, :]).sum(axis=0))
    lt = sum((b[:, None] < a[None, :]).sum(axis=0))
    return (gt - lt) / (a.size * b.size)
 
 
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
 
 
# --------------------------------------------------------------------------
# FIG 1 - the headline: emergence vs structure size, slabs as control
# --------------------------------------------------------------------------
def fig_emergence_size(levels, out):
    multi = defaultdict(lambda: [0, 0])
    slab = defaultdict(lambda: [0, 0])
    for d in levels.values():
        for r in d["emergence"]:
            if not r.get("specified_cells"):
                continue
            n = int(r["specified_cells"])
            lab = next(l for lim, l in SIZE_BANDS if n <= lim)
            tgt = slab if r.get("distinct_tiles") == "1" else multi
            tgt[lab][0] += 1
            tgt[lab][1] += int(r["core_emerges"])
 
    labs = [l for _, l in SIZE_BANDS if l in multi or l in slab]
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for off, data, c, name in ((-w/2, multi, "#C44E52", "multi-tile structures"),
                               (+w/2, slab, "#8C8C8C", "single-tile slabs (control)")):
        pct, txt, lo, hi = [], [], [], []
        for l in labs:
            t, e = data.get(l, [0, 0])
            m, a, b = wilson(e, t)
            pct.append(m); lo.append(m - a); hi.append(b - m)
            txt.append(f"{e}/{t}" if t else "")
        bars = ax.bar(x + off, pct, w, color=c, label=name,
                      yerr=[lo, hi], capsize=3,
                      error_kw=dict(lw=1, ecolor="#444444"))
        for b, t, u in zip(bars, txt, hi):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + u + 2.5, t,
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel("structure size (specified cells)")
    ax.set_ylabel("% reproduced by the Core generator")
    ax.set_ylim(0, 125)
    ax.set_title("Do complex-tile structures arise in Core generations?\n"
                 "bars are 95% Wilson intervals", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 2 - where each declared structure ends up
# --------------------------------------------------------------------------
def fig_overlap(levels, out):
    order = ["both", "complex_only", "core_only", "neither"]
    colours = {"both": "#8C8C8C", "complex_only": "#55A868",
               "core_only": "#4C72B0", "neither": "#C44E52"}
    names = sorted(levels)
    counts = {lv: Counter(r.get("where", "") for r in levels[lv]["emergence"])
              for lv in names}
    pooled = Counter()
    for c in counts.values():
        pooled.update(c)
 
    labels = names + ["POOLED"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottoms = np.zeros(len(labels))
    for k in order:
        vals = []
        for lv in names:
            tot = sum(counts[lv][o] for o in order) or 1
            vals.append(100 * counts[lv][k] / tot)
        tot = sum(pooled[o] for o in order) or 1
        vals.append(100 * pooled[k] / tot)
        vals = np.array(vals)
        ax.barh(labels, vals, left=bottoms, color=colours[k],
                label=k.replace("_", " "))
        bottoms += vals
    ax.set_xlabel("% of declared macro structures")
    ax.set_xlim(0, 100)
    ax.set_title("Where each declared structure appears")
    ax.legend(frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.3), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for i, lv in enumerate(labels):
        n = sum(pooled[o] for o in order) if lv == "POOLED" else sum(counts[lv][o] for o in order)
        ax.text(101, i, f"n={n}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 3 - which metric differences replicate across levels
# --------------------------------------------------------------------------
METRICS = ["Time (s)", "KL-Div (1x1)", "Avg Plat Thickness", "Hazard Density /10col",
           "Obstacles", "Avg Plat Width", "Mean Obstacle Width", "Max Obstacle Height",
           "Unjumpable Obstacles", "Platforms", "Components", "Solid Density",
           "Roughness (std)", "Linearity", "Elevation Std", "Gaps",
           "Unreachable Platforms", "Reachable %", "N-Gram Entropy",
           "Compression Ratio", "Repetition Peak"]
 
 
def fig_effect_heatmap(levels, out, use=None):
    use = use or sorted(levels)
    M, S = [], []
    keep = []
    for m in METRICS:
        row, sig = [], []
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            a, b = col(c, m), col(x, m)
            if a.size < 3 or b.size < 3:
                row.append(np.nan); sig.append(False); continue
            row.append(cliffs_delta(a, b))
            try: p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError: p = 1.0
            sig.append(p < 0.05)
        if not all(np.isnan(row)):
            M.append(row); S.append(sig); keep.append(m)
 
    M = np.array(M, dtype=float)
    fig, ax = plt.subplots(figsize=(1.05*len(use)+6.0, 0.42*len(keep)+2))
    im = ax.imshow(M, cmap="coolwarm", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax.set_xticks(range(len(use)))
    ax.set_xticklabels(use, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=8)
    for i in range(len(keep)):
        for j in range(len(use)):
            if S[i][j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=11)
    # verdict column
    for i, m in enumerate(keep):
        vals = [M[i][j] for j in range(len(use)) if not np.isnan(M[i][j])]
        up = sum(1 for j in range(len(use)) if S[i][j] and M[i][j] > 0)
        dn = sum(1 for j in range(len(use)) if S[i][j] and M[i][j] < 0)
        tag = "robust" if (up == len(vals) or dn == len(vals)) else \
              "consistent" if min(up, dn) == 0 and max(up, dn) >= len(vals) - 1 else \
              "conflicts" if up and dn else "weak"
        ax.text(len(use) - 0.35, i, "  " + tag, va="center", fontsize=8,
                color={"robust": "#2A6F2A", "consistent": "#2A6F2A",
                       "conflicts": "#A02020", "weak": "#777777"}[tag])
    ax.set_xlim(-0.5, len(use) - 0.5 + 2.4)
    ax.set_title("Complex vs Core effect direction per level\n"
                 "Cliff's delta, red = higher with complex tiles, * = p<0.05",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label="Cliff's delta")
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 4 - generation reliability
# --------------------------------------------------------------------------
def fig_failures(levels, out, requested):
    names = sorted(levels)
    core = [requested - len(split(levels[lv]["metrics"])[0]) for lv in names]
    comp = [requested - len(split(levels[lv]["metrics"])[1]) for lv in names]
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.45*len(names)+2))
    ax.barh(y - h/2, core, h, color=CORE_C, label="Core")
    ax.barh(y + h/2, comp, h, color=COMP_C, label="Complex")
    ax.axvline(requested * (1 - QUALITY_MIN), color="k", ls=":", lw=1,
               label=f"exclusion threshold ({int(requested*(1-QUALITY_MIN))} losses)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"failed generations (of {requested} requested)")
    ax.set_title("Generation reliability per level")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 5 - paired slopes for the metrics that do replicate
# --------------------------------------------------------------------------
def fig_paired(levels, out, metrics=("Time (s)", "KL-Div (1x1)",
                                     "Avg Plat Thickness", "Hazard Density /10col"),
               use=None):
    use = use or sorted(levels)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1*len(metrics), 4.2))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for lv in use:
            c, x = split(levels[lv]["metrics"])
            a, b = col(c, m), col(x, m)
            if a.size < 3 or b.size < 3: continue
            try: p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError: p = 1.0
            ax.plot([0, 1], [a.mean(), b.mean()],
                    "-o", ms=4, lw=1.6 if p < 0.05 else 0.8,
                    color="#333333" if p < 0.05 else "#BBBBBB", zorder=2)
            ax.annotate(lv, (1.02, b.mean()), fontsize=6, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Core", "Complex"])
        ax.set_xlim(-0.15, 1.5)
        ax.set_title(m, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-level means, Core to Complex  (bold = p<0.05)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 6 - expressive range
# --------------------------------------------------------------------------
def fig_expressive(levels, out, xm="Linearity", ym="Hazard Density /10col", use=None):
    use = use or sorted(levels)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for rows, c, name in ((None, CORE_C, "Core"), (None, COMP_C, "Complex")):
        xs, ys = [], []
        for lv in use:
            cr, cx = split(levels[lv]["metrics"])
            src = cr if name == "Core" else cx
            a, b = col(src, xm), col(src, ym)
            n = min(a.size, b.size)
            xs += list(a[:n]); ys += list(b[:n])
        ax.scatter(xs, ys, s=22, alpha=0.55, color=c, label=name, edgecolor="none")
    ax.set_xlabel(xm); ax.set_ylabel(ym)
    ax.set_title("Expressive range: each point is one generated level")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 7 - what stops a macro being placed
# --------------------------------------------------------------------------
def fig_placement(levels, out):
    groups = {"interior holes": [0, 0], "no interior holes": [0, 0],
              "contradictory rules": [0, 0], "consistent rules": [0, 0]}
    for d in levels.values():
        for r in d["emergence"]:
            placed = int(r.get("complex_exact_count") or 0) > 0
            if r.get("interior_holes") not in (None, ""):
                k = "interior holes" if r["interior_holes"] != "0" else "no interior holes"
                groups[k][0] += 1; groups[k][1] += placed
            if r.get("start_invariant") in ("True", "False"):
                k = "contradictory rules" if r["start_invariant"] == "False" else "consistent rules"
                groups[k][0] += 1; groups[k][1] += placed
    labs = [k for k in groups if groups[k][0]]
    pct = [100*groups[k][1]/groups[k][0] for k in labs]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(labs, pct, color=["#C44E52", "#55A868", "#C44E52", "#55A868"][:len(labs)])
    for b, k in zip(bars, labs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                f"{groups[k][1]}/{groups[k][0]}", ha="center", fontsize=9)
    ax.set_ylabel("% of macros placed at least once")
    ax.set_ylim(0, 108)
    ax.set_title("Which macros the solver can actually place")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
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
 
 
# --------------------------------------------------------------------------
# FIG 8 - every structure individually, no binning
# --------------------------------------------------------------------------
def fig_emergence_scatter(levels, out):
    """
    One dot per declared structure: size against whether Core reproduced it.
    Binning into size bands is a presentation choice, so showing the raw 125
    points guards against the claim being an artefact of where the bin edges
    fell. A fitted logistic gives the 50% crossover size.
    """
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
 
    for is_slab, colour, name in ((False, "#C44E52", "multi-tile"),
                                  (True, "#8C8C8C", "single-tile slab")):
        xs, ys = [], []
        for d in levels.values():
            for r in d["emergence"]:
                if not r.get("specified_cells") or r.get("distinct_tiles") in (None, ""):
                    continue
                if (r["distinct_tiles"] == "1") != is_slab:
                    continue
                xs.append(int(r["specified_cells"]))
                ys.append(int(r["core_emerges"]))
        if not xs:
            continue
        xs, ys = np.array(xs, float), np.array(ys, float)
        ax.scatter(xs + rng.normal(0, 0.16, xs.size),
                   ys + rng.normal(0, 0.022, ys.size),
                   s=26, alpha=0.6, color=colour, edgecolor="none",
                   label=f"{name} (n={xs.size})")
        if len(set(ys)) > 1 and xs.size >= 8:
            try:
                (x0, k), _ = curve_fit(_logistic, xs, ys, p0=[8, -0.4], maxfev=8000)
                grid = np.linspace(xs.min(), xs.max(), 200)
                ax.plot(grid, _logistic(grid, x0, k), color=colour, lw=2)
                if 0 < x0 < xs.max() and k < 0:
                    ax.axvline(x0, color=colour, ls=":", lw=1)
                    ax.text(x0, 1.09, f" 50% at {x0:.1f} cells",
                            color=colour, fontsize=8, ha="left")
            except Exception:
                pass
 
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not reproduced", "reproduced"])
    ax.set_ylim(-0.12, 1.16)
    ax.set_xlabel("structure size (specified cells)")
    ax.set_title("Every declared structure, unbinned")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
# --------------------------------------------------------------------------
# FIG 9 - ingredients present but never assembled
# --------------------------------------------------------------------------
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
 
 
# --------------------------------------------------------------------------
# FIG 10 - the mirror of fig 8: can the solver PLACE what it declares?
# --------------------------------------------------------------------------
def fig_placement_vs_size(levels, out):
    """
    Two different failures, plotted on one axis against structure size.
 
      Core reproduced   - the structure arose by chance from single tiles.
                          Falling with size is the point of the whole method.
      Complex placed    - the solver stamped the macro at least once, having
                          been given it explicitly. Falling with size would
                          mean WFC cannot satisfy large multi-cell constraints,
                          so the macros do not deliver what they promise.
 
    The vertical gap between the two curves is the value the complex tileset
    actually adds: sizes where a structure is placeable but not reproducible.
    Where both curves are low the macro is declared and never appears at all.
 
    Multi-tile structures only; single-tile slabs are excluded because they
    are trivially reproducible at any size and would flatten the comparison.
    """
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(1)
 
    xs, y_core, y_plac = [], [], []
    for d in levels.values():
        for r in d["emergence"]:
            if not r.get("specified_cells") or r.get("distinct_tiles") in (None, ""):
                continue
            if r["distinct_tiles"] == "1":
                continue
            placed_raw = r.get("solver_placements")
            if placed_raw in (None, ""):
                placed_raw = r.get("complex_exact_count", "0")
            xs.append(int(r["specified_cells"]))
            y_core.append(int(r["core_emerges"]))
            y_plac.append(1 if int(placed_raw or 0) > 0 else 0)
    if not xs:
        return
    xs = np.array(xs, float)
 
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    series = ((np.array(y_core, float), "#C44E52", "reproduced by Core (chance)"),
              (np.array(y_plac, float), "#4C72B0", "placed by the Complex solver"))
    crossovers = []
    for si, (ys, colour, name) in enumerate(series):
        ax.scatter(xs + rng.normal(0, 0.16, xs.size),
                   ys + rng.normal(0, 0.022, ys.size),
                   s=24, alpha=0.5, color=colour, edgecolor="none",
                   label=f"{name}  ({int(ys.sum())}/{ys.size})")
        if len(set(ys)) > 1 and xs.size >= 8:
            try:
                (x0, k), _ = curve_fit(_logistic, xs, ys, p0=[8, -0.4], maxfev=8000)
                grid = np.linspace(xs.min(), xs.max(), 300)
                ax.plot(grid, _logistic(grid, x0, k), color=colour, lw=2)
                if xs.min() < x0 < xs.max() and k < 0:
                    ax.axvline(x0, color=colour, ls=":", lw=1)
                    crossovers.append((x0, colour, name))
            except Exception:
                pass
 
    # stack the crossover labels so they cannot overlap each other
    for i, (x0, colour, name) in enumerate(sorted(crossovers)):
        ax.text(x0 + 0.3, 1.13 - 0.075 * i, f"50% at {x0:.1f} cells",
                color=colour, fontsize=8, ha="left", va="center")
 
    ax.set_yticks([0, 1]); ax.set_yticklabels(["absent", "present"])
    ax.set_ylim(-0.12, 1.22)
    ax.set_xlabel("structure size (specified cells)")
    ax.set_title("Reproducible by chance vs placeable by the solver\n"
                 "multi-tile structures only; the gap is what the macros add",
                 fontsize=11)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
 
 
def placement_size_table(levels):
    """Numbers behind fig 10, for the text."""
    bands = defaultdict(lambda: [0, 0, 0])   # n, core-emerged, solver-placed
    for d in levels.values():
        for r in d["emergence"]:
            if not r.get("specified_cells") or r.get("distinct_tiles") in (None, ""):
                continue
            if r["distinct_tiles"] == "1":
                continue
            n = int(r["specified_cells"])
            lab = next(l for lim, l in SIZE_BANDS if n <= lim)
            placed_raw = r.get("solver_placements")
            if placed_raw in (None, ""):
                placed_raw = r.get("complex_exact_count", "0")
            b = bands[lab]
            b[0] += 1; b[1] += int(r["core_emerges"])
            b[2] += 1 if int(placed_raw or 0) > 0 else 0
    print(f"\n{'size':>7}{'n':>5}{'Core reproduced':>18}{'Complex placed':>17}"
          f"{'gap':>7}")
    for _, lab in SIZE_BANDS:
        if lab not in bands: continue
        n, e, p = bands[lab]
        print(f"{lab:>7}{n:>5}{f'{e}/{n} ({100*e/n:.0f}%)':>18}"
              f"{f'{p}/{n} ({100*p/n:.0f}%)':>17}{f'{100*(p-e)/n:+.0f}pt':>7}")
 
 
# --------------------------------------------------------------------------
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
    print(f"{len(levels)} levels loaded; {requested} generations requested per batch; "
          f"{len(good)} usable for metric comparison "
          f"(needs >={QUALITY_MIN:.0%} of {requested} in both batches)")
    for lv in sorted(levels):
        c, x = split(levels[lv]["metrics"])
        print(f"   {lv:<28} core {len(c):>3}/{requested}  complex {len(x):>3}/{requested}"
              f"   {'' if lv in good else '<- excluded from metrics'}")
 
    p = lambda n: os.path.join(figs, n)
    fig_emergence_size(levels, p("1_emergence_by_size.png"))
    fig_overlap(levels, p("2_structure_overlap.png"))
    fig_effect_heatmap(levels, p("3_effect_heatmap.png"), use=good)
    fig_failures(levels, p("4_generation_failures.png"), requested)
    fig_paired(levels, p("5_paired_means.png"), use=good)
    fig_expressive(levels, p("6_expressive_range.png"), use=good)
    fig_placement(levels, p("7_placement_barriers.png"))
    fig_emergence_scatter(levels, p("8_emergence_unbinned.png"))
    fig_partial(levels, p("9_partial_emergence.png"), k=2)
    fig_placement_vs_size(levels, p("10_placed_vs_reproduced.png"))
    placement_size_table(levels)
    print(f"\nwrote 10 figures to {figs}")
 
 
if __name__ == "__main__":
    main()


#python demo/plot.py Generation/Baselines/all_levels