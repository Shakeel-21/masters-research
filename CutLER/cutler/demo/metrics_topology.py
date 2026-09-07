"""
metrics_topology.py
===================
Corrected structural / topological metrics for evaluating WFC-generated
platformer levels (tested against Super Mario Bros tilesets).

WHY THIS EXISTS
---------------
The original `extract_topological_metrics` derived every structural metric from
a 1D "skyline" (the topmost non-background tile per column). A 1D projection
cannot distinguish a ground-attached block from a floating platform above it, so:

  * a 3-wide block on the floor followed by a 4-wide platform 5 tiles up
    registered as ONE obstacle run,
  * roughness saw a fictitious 5-tile cliff between them,
  * "average platform width" was dominated by the floor itself.

This module works on the 2D occupancy grid instead:

  every non-background tile is solid
             -> connected components (4-connectivity)
             -> components touching the bottom row  = GROUND structure
             -> everything else                     = FLOATING platforms

Ground surface, roughness, gaps and obstacles are computed from the ground
structure only. Platform metrics are computed from the floating components only.

USAGE
-----
    from metrics_topology import structural_metrics

    metrics, ngrams, debug = structural_metrics(grid, id_to_tile)
"""

from __future__ import annotations

import os
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# --------------------------------------------------------------------------
# Tile name handling
# --------------------------------------------------------------------------

# Same pattern used by the generator, but with the local offsets captured and
# the tail anchored so partial names cannot match.
CLONE_RE = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+?)_y(\d+)_x(\d+)$")

BACKGROUND = "B"
PADDING = "P"
UNCOLLAPSED = "UNCOLLAPSED"


def parse_cell_name(raw_name: str):
    """
    Fold a grid cell's tile name to (base_image_name, local_y, local_x, is_clone).

    Atomic tile   : "tile_12.png"                 -> ("tile_12.png", 0, 0, False)
    Complex clone : "lvl_seg_3_tile_5_y1_x2"      -> ("tile_5.png",   1, 2, True)
    Specials      : "B" / "P"                     -> unchanged
    """
    m = CLONE_RE.match(raw_name)
    if m:
        return f"{m.group(2)}.png", int(m.group(3)), int(m.group(4)), True
    if raw_name in (BACKGROUND, PADDING, UNCOLLAPSED):
        return raw_name, 0, 0, False
    base = raw_name if raw_name.endswith(".png") else f"{raw_name}.png"
    return base, 0, 0, False


def grid_to_arrays(grid, id_to_tile):
    """
    Convert the raw WFC grid into (names, local_y, local_x) object/int arrays
    with the padding ring removed.

    Padding is detected (rows/columns that are entirely "P") rather than assumed
    to be exactly one cell thick.
    """
    h, w = len(grid), len(grid[0])
    names = np.empty((h, w), dtype=object)
    lys = np.zeros((h, w), dtype=int)
    lxs = np.zeros((h, w), dtype=int)

    for y in range(h):
        row = grid[y]
        for x in range(w):
            cell = row[x]
            if isinstance(cell, (set, frozenset)):
                names[y, x] = UNCOLLAPSED
                continue
            base, ly, lx, _ = parse_cell_name(id_to_tile.get(cell, str(cell)))
            names[y, x] = base
            lys[y, x] = ly
            lxs[y, x] = lx

    pad = names == PADDING
    keep_r = ~pad.all(axis=1)
    keep_c = ~pad.all(axis=0)
    if keep_r.any() and keep_c.any():
        sel = np.ix_(keep_r, keep_c)
        names, lys, lxs = names[sel], lys[sel], lxs[sel]
    return names, lys, lxs


# --------------------------------------------------------------------------
# Solidity: which tiles are collidable geometry?
# --------------------------------------------------------------------------

@dataclass


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class LevelConfig:
    """SMB-ish physics limits, in tiles."""
    jump_height: int = 4        # max upward reach of a full jump
    jump_run: int = 5           # max horizontal distance cleared
    body_height: int = 2        # Mario is 2 tiles tall (big)
    obstacle_threshold: int = 1 # ground rise > this counts as an obstacle
    compute_reachability: bool = True


# --------------------------------------------------------------------------
# Core structural analysis
# --------------------------------------------------------------------------

def is_solid(name):
    """
    Occupancy rule: every tile except the background is an object.

    "B" is background and the padding/uncollapsed markers are not tiles, so
    everything else contributes geometry. Complex clones fold to the core tile
    occupying that cell, so both batches are measured in exactly the same way.
    """
    return name not in (BACKGROUND, PADDING, UNCOLLAPSED)


def surface_rows(mask):
    """Topmost occupied row per column; -1 where the column is empty."""
    has = mask.any(axis=0)
    top = np.argmax(mask, axis=0)
    return np.where(has, top, -1)


def _runs(flags):
    """Yield (start, length) for each run of True in a 1D bool array."""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def structural_metrics(grid, id_to_tile, cfg: LevelConfig = LevelConfig(),
                       ngram_size=3):
    """
    Returns (metrics_dict, ngram_counter, debug_dict).

    debug_dict carries the masks and profiles so results can be visually
    audited (see dump_occupancy_ascii) and so callers can compute their own
    derived statistics without re-deriving the masks.
    """
    names, _, _ = grid_to_arrays(grid, id_to_tile)
    H, W = names.shape

    # ---- 1. occupancy ---------------------------------------------------
    uncollapsed = names == UNCOLLAPSED
    solid = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            solid[y, x] = is_solid(names[y, x])

    # ---- 2. components: ground vs floating ------------------------------
    # 4-connectivity: diagonal-only contact is not a shared surface.
    labels, n_comp = ndimage.label(solid)
    bottom_labels = set(np.unique(labels[-1, :])) - {0}
    ground = np.isin(labels, sorted(bottom_labels)) if bottom_labels else np.zeros_like(solid)
    floating = solid & ~ground

    # ---- 3. ground surface profile --------------------------------------
    g_top = surface_rows(ground)                      # -1 = pit
    has_ground = g_top >= 0
    g_elev = np.where(has_ground, H - g_top, 0)       # tiles above grid bottom

    if has_ground.any():
        vals = g_elev[has_ground]
        ground_level = int(np.bincount(vals).argmax())  # modal floor height
    else:
        ground_level = 0

    # ---- 4. roughness (gaps excluded) -----------------------------------
    # Only compare columns that BOTH have ground. A pit is a gap, not a cliff,
    # and floating platforms no longer contribute at all.
    pair_ok = has_ground[:-1] & has_ground[1:]
    steps = np.abs(np.diff(g_elev))[pair_ok] if W > 1 else np.array([])
    roughness_std = float(np.std(steps)) if steps.size else 0.0
    roughness_mean = float(np.mean(steps)) if steps.size else 0.0
    elev_std = float(np.std(g_elev[has_ground])) if has_ground.any() else 0.0

    # linearity: mean |residual| from a least-squares fit of the ground profile
    if has_ground.sum() >= 2:
        xs = np.flatnonzero(has_ground)
        m, b = np.polyfit(xs, g_elev[xs], 1)
        linearity = float(np.mean(np.abs(g_elev[xs] - (m * xs + b))))
        slope = float(m)
    else:
        linearity, slope = 0.0, 0.0

    # ---- 5. gaps (pits) --------------------------------------------------
    gap_runs = _runs(~has_ground)
    gap_widths = [ln for _, ln in gap_runs]
    unjumpable_gaps = sum(1 for wd in gap_widths if wd > cfg.jump_run)

    # ---- 6. obstacles: ground-attached rises above the modal floor -------
    # Component-labelled in 2D, so two adjacent-but-separate structures are
    # two obstacles, and a floating platform above one is neither.
    rows = np.arange(H)[:, None]
    above = ground & (rows < (H - ground_level - cfg.obstacle_threshold + 1))
    obs_labels, n_obs = ndimage.label(above)
    obs_heights, obs_widths = [], []
    for i in range(1, n_obs + 1):
        ys, xs = np.nonzero(obs_labels == i)
        obs_heights.append(int((H - ys.min()) - ground_level))
        obs_widths.append(int(len(np.unique(xs))))
    unjumpable_obs = sum(1 for h in obs_heights if h > cfg.jump_height)

    # ---- 7. floating platforms only --------------------------------------
    fl_labels, n_float = ndimage.label(floating)
    plat_widths, plat_clearance, plat_thickness = [], [], []
    unreachable_plats = 0
    for i in range(1, n_float + 1):
        ys, xs = np.nonzero(fl_labels == i)
        cols = np.unique(xs)
        plat_widths.append(int(len(cols)))
        plat_thickness.append(int(ys.max() - ys.min() + 1))
        under = g_elev[cols]
        base_elev = float(np.median(under[under > 0])) if (under > 0).any() else float(ground_level)
        clearance = (H - ys.max()) - base_elev      # tiles of air beneath it
        plat_clearance.append(float(clearance))
        if clearance - 1 > cfg.jump_height:
            unreachable_plats += 1

    # ---- 8. n-grams on the folded name grid ------------------------------
    ngrams = Counter()
    k = ngram_size
    for y in range(H - k + 1):
        for x in range(W - k + 1):
            ngrams[tuple(names[y:y + k, x:x + k].flatten())] += 1
    total = sum(ngrams.values())
    ng_probs = np.array(list(ngrams.values()), dtype=float) / total if total else np.array([])
    ng_entropy = float(entropy(ng_probs)) if ng_probs.size else 0.0

    # ---- 9. reachability -------------------------------------------------
    reach = {"Reachable %": np.nan, "Completable": np.nan, "Standable Cells": 0}
    if cfg.compute_reachability:
        reach = reachability(solid, cfg)

    metrics = {
        # occupancy
        "Solid Density": round(float(solid.mean()), 4),
        "Uncollapsed": int(uncollapsed.sum()),
        "Components": int(n_comp),
        # ground surface
        "Ground Level": ground_level,
        "Ground Coverage": round(float(has_ground.mean()), 4),
        "Roughness (std)": round(roughness_std, 4),
        "Roughness (mean step)": round(roughness_mean, 4),
        "Elevation Std": round(elev_std, 4),
        "Linearity": round(linearity, 4),
        "Slope": round(slope, 5),
        # gaps
        "Gaps": len(gap_widths),
        "Max Gap": int(max(gap_widths)) if gap_widths else 0,
        "Mean Gap": round(float(np.mean(gap_widths)), 2) if gap_widths else 0.0,
        "Unjumpable Gaps": unjumpable_gaps,
        # obstacles (ground-attached only)
        "Obstacles": n_obs,
        "Mean Obstacle Height": round(float(np.mean(obs_heights)), 2) if obs_heights else 0.0,
        "Max Obstacle Height": int(max(obs_heights)) if obs_heights else 0,
        "Mean Obstacle Width": round(float(np.mean(obs_widths)), 2) if obs_widths else 0.0,
        "Unjumpable Obstacles": unjumpable_obs,
        # floating platforms only
        "Platforms": n_float,
        "Avg Plat Width": round(float(np.mean(plat_widths)), 2) if plat_widths else 0.0,
        "Avg Plat Clearance": round(float(np.mean(plat_clearance)), 2) if plat_clearance else 0.0,
        "Avg Plat Thickness": round(float(np.mean(plat_thickness)), 2) if plat_thickness else 0.0,
        "Unreachable Platforms": unreachable_plats,
        # rhythm / complexity
        "N-Gram Entropy": round(ng_entropy, 4),
        "Repetition Peak": round(profile_repetition(g_elev), 4),
        "Compression Ratio": round(compression_ratio(names), 4),
        "Hazard Density /10col": round(
            10.0 * (len(gap_widths) + n_obs) / W if W else 0.0, 3),
    }
    metrics.update(reach)

    debug = {
        "names": names, "solid": solid, "ground": ground, "floating": floating,
        "ground_elev": g_elev, "has_ground": has_ground,
        "ground_level": ground_level,
        "obstacle_heights": obs_heights, "plat_widths": plat_widths,
        "gap_widths": gap_widths,
    }
    return metrics, ngrams, debug


# --------------------------------------------------------------------------
# Reachability (structural playability proxy)
# --------------------------------------------------------------------------

def reachability(solid, cfg: LevelConfig = LevelConfig()):
    """
    Flood-fill of standable tiles under a coarse jump model.

    A cell is standable if it is free, has solid directly below, and has
    `body_height` free cells of headroom. Two standable cells are connected if
    the horizontal distance <= jump_run, the rise <= jump_height, and every
    intervening column offers a free body-height window somewhere in the
    vertical band spanned by the jump.

    This is an OPTIMISTIC upper bound - it ignores momentum, run-up distance,
    enemies and moving platforms. For publication-grade playability numbers,
    run an actual agent (the Mario AI Framework A* agent is the field standard);
    use this as a fast structural screen.
    """
    H, W = solid.shape
    free = ~solid
    below = np.zeros_like(solid)
    below[:-1, :] = solid[1:, :]
    head = free.copy()
    for d in range(1, cfg.body_height):
        shifted = np.ones_like(free)
        shifted[d:, :] = free[:-d, :]
        head &= shifted
    standable = free & below & head

    nodes = [(int(y), int(x)) for y, x in zip(*np.nonzero(standable))]
    if not nodes:
        return {"Reachable %": 0.0, "Completable": 0, "Standable Cells": 0}

    by_col = defaultdict(list)
    for y, x in nodes:
        by_col[x].append(y)

    def corridor_ok(x0, y0, x1, y1):
        lo = min(y0, y1) - cfg.jump_height
        hi = max(y0, y1)
        step = 1 if x1 > x0 else -1
        for xi in range(x0 + step, x1 + step, step):
            ok = False
            for r in range(max(cfg.body_height - 1, lo), hi + 1):
                if r < 0 or r >= H:
                    continue
                if free[r, xi] and all(free[r - d, xi] for d in range(1, cfg.body_height)
                                       if r - d >= 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    # spawn: lowest standable tile in the leftmost column that has one
    start = None
    for x in sorted(by_col):
        start = (max(by_col[x]), x)
        break

    seen = {start}
    stack = [start]
    while stack:
        y, x = stack.pop()
        for dx in range(-cfg.jump_run, cfg.jump_run + 1):
            if dx == 0:
                continue
            xt = x + dx
            if xt < 0 or xt >= W:
                continue
            for yt in by_col.get(xt, ()):
                if (yt, xt) in seen:
                    continue
                rise = y - yt                     # >0 means jumping up
                if rise > cfg.jump_height:
                    continue
                if abs(dx) > 1 and rise > 0 and (abs(dx) / cfg.jump_run + rise / cfg.jump_height) > 1.5:
                    continue
                if abs(dx) > 1 and not corridor_ok(x, y, xt, yt):
                    continue
                seen.add((yt, xt))
                stack.append((yt, xt))

    last_col = max(by_col)
    completable = any(x >= last_col for _, x in seen)
    return {
        "Reachable %": round(100.0 * len(seen) / len(nodes), 2),
        "Completable": int(completable),
        "Standable Cells": len(nodes),
    }


# --------------------------------------------------------------------------
# Repetition / complexity
# --------------------------------------------------------------------------

def profile_repetition(profile, min_lag=4):
    """
    Strongest non-trivial autocorrelation peak of the ground profile.
    High values indicate periodic, visibly tiled output - a classic WFC
    artefact when the pattern vocabulary is small.
    """
    p = np.asarray(profile, dtype=float)
    if p.size < 2 * min_lag or np.std(p) == 0:
        return 0.0
    p = p - p.mean()
    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    ac /= ac[0]
    tail = ac[min_lag:max(min_lag + 1, p.size // 2)]
    return float(tail.max()) if tail.size else 0.0


def compression_ratio(name_grid):
    """
    zlib-compressed size / raw size of the tile grid. A cheap proxy for
    structural complexity: low = repetitive, high = noisy.
    """
    vocab = {v: i for i, v in enumerate(sorted(set(name_grid.flatten())))}
    raw = bytes(vocab[v] % 256 for v in name_grid.flatten())
    if not raw:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


# --------------------------------------------------------------------------
# Distribution comparison helpers
# --------------------------------------------------------------------------


def js_distance(counts_a, counts_b):
    """Jensen-Shannon *distance* (sqrt of divergence), 0 = identical."""
    keys = sorted(set(counts_a) | set(counts_b), key=lambda t: str(t))
    if not keys:
        return 0.0
    a = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    b = np.array([counts_b.get(k, 0) for k in keys], dtype=float)
    a /= a.sum() or 1
    b /= b.sum() or 1
    d = jensenshannon(a, b)
    return 0.0 if np.isnan(d) else float(d)


def pairwise_diversity(name_grids):
    """
    Mean pairwise normalised Hamming distance between levels of equal size.
    0 = every level identical, 1 = no shared cell. Measures intra-batch
    diversity far more reliably than comparing perceptual hashes for equality.
    """
    grids = [g for g in name_grids if g is not None]
    if len(grids) < 2:
        return float("nan")
    shape = grids[0].shape
    grids = [g for g in grids if g.shape == shape]
    ds = []
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            ds.append(float((grids[i] != grids[j]).mean()))
    return round(float(np.mean(ds)), 4) if ds else float("nan")


# --------------------------------------------------------------------------
# Debug output
# --------------------------------------------------------------------------

def dump_occupancy_ascii(debug, path, max_width=None):
    """
    Write an ASCII map of the classification so it can be diffed against the
    rendered PNG. Always eyeball this once per dataset before trusting numbers.

      '#' ground-connected solid   'o' floating solid
      '.' passable / background    '?' uncollapsed
      '_' ground surface marker    'v' pit column (no ground at all)
    """
    names = debug["names"]
    solid, ground, floating = debug["solid"], debug["ground"], debug["floating"]
    H, W = names.shape
    W = min(W, max_width or W)
    lines = []
    for y in range(H):
        row = []
        for x in range(W):
            if names[y, x] == UNCOLLAPSED:
                row.append("?")
            elif ground[y, x]:
                row.append("#")
            elif floating[y, x]:
                row.append("o")
            else:
                row.append(".")
        lines.append("".join(row))
    marks = "".join("v" if not debug["has_ground"][x] else "_" for x in range(W))
    lines.append(marks)
    lines.append(f"ground_level={debug['ground_level']}  "
                 f"gaps={debug['gap_widths']}  "
                 f"obstacle_heights={debug['obstacle_heights']}  "
                 f"plat_widths={debug['plat_widths']}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print(__doc__)