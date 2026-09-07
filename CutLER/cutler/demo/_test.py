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

  solid mask -> connected components (4-connectivity)
             -> components touching the bottom row  = GROUND structure
             -> everything else                     = FLOATING platforms

Ground surface, roughness, gaps and obstacles are computed from the ground
structure only. Platform metrics are computed from the floating components only.

USAGE
-----
    from metrics_topology import SolidityModel, structural_metrics

    solidity = SolidityModel(mode="non_background")          # drop-in default
    # or, much better:
    solidity = SolidityModel(mode="classes",
                             class_map=json.load(open("tile_classes.json")))

    metrics, ngrams, debug = structural_metrics(grid, id_to_tile, solidity)

Generate a stub class map to fill in:
    python metrics_topology.py --stub path/to/dataset/tiles > tile_classes.json
"""

from __future__ import annotations

import json
import os
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

# Classes that Mario can stand on / is blocked by.
SOLID_CLASSES = {"solid", "ground", "block", "breakable", "question", "pipe", "platform"}
# Classes that occupy a cell but are walked through (decor, pickups, enemies).
PASSABLE_CLASSES = {"background", "decoration", "coin", "reward", "enemy", "hazard", "empty"}


@dataclass
class SolidityModel:
    """
    Decides, per grid cell, whether that cell is collidable geometry.

    mode="non_background"
        Everything except "B" is solid. Reproduces the original behaviour.
        Cheap, but counts coins, bushes, clouds and enemies as platforms, and
        (see caveat below) inflates the Complex batch.

    mode="classes"
        Uses `class_map`: {base_tile_name: class_string}. Recommended.

    mode="image"
        Per-cell occupancy read out of the tile PNGs: for a clone cell at
        (local_y, local_x) the corresponding sub-cell of the parent image is
        cropped and tested for non-empty pixels. This is the only mode that
        measures complex (multi-cell) tiles fairly - see CAVEAT.

    CAVEAT (important for the Core vs Complex comparison)
    -----------------------------------------------------
    Every cell of a complex tile folds to the same base image name, so any
    name-based rule marks all N cells of an NxM chunk identically. If that chunk
    is mostly sky, the Complex batch is measured as far denser and rougher than
    it really is, and the Core/Complex comparison is biased. mode="image" fixes
    this; verify it with `dump_occupancy_ascii` against the rendered PNG.
    """

    mode: str = "non_background"
    class_map: dict = field(default_factory=dict)
    tiles_dir: Optional[str] = None
    cell_size: Optional[Tuple[int, int]] = None          # (w, h) in px; auto-detected if None
    bg_color: Optional[Tuple[int, int, int]] = None           # e.g. (92, 148, 252) Mario sky
    alpha_threshold: int = 16
    color_tolerance: int = 24
    coverage_threshold: float = 0.10        # frac of non-empty px to call solid
    y_from_top: bool = True                 # local_y=0 is the TOP row of a clone
    default_solid: bool = True              # unknown tile -> solid

    def __post_init__(self):
        self._cache: dict = {}
        self._images: dict = {}
        self._warned = set()
        if self.mode == "image":
            self._load_images()

    # -- image backend ----------------------------------------------------
    def _load_images(self):
        from PIL import Image

        if not self.tiles_dir or not os.path.isdir(self.tiles_dir):
            raise ValueError("mode='image' requires a valid tiles_dir")
        ws, hs = [], []
        for f in sorted(os.listdir(self.tiles_dir)):
            if not f.lower().endswith(".png"):
                continue
            img = Image.open(os.path.join(self.tiles_dir, f)).convert("RGBA")
            self._images[f] = img
            ws.append(img.width)
            hs.append(img.height)
        if not self._images:
            raise ValueError(f"no PNGs in {self.tiles_dir}")
        if self.cell_size is None:
            self.cell_size = (Counter(ws).most_common(1)[0][0],
                              Counter(hs).most_common(1)[0][0])
        if self.bg_color is None:
            self.bg_color = self._estimate_bg_color()

    def _estimate_bg_color(self):
        """Most common opaque colour across the tileset (usually the sky)."""
        counts = Counter()
        for img in self._images.values():
            a = np.asarray(img)
            opaque = a[a[..., 3] > 200][:, :3]
            if opaque.size:
                # subsample for speed
                for px in opaque[::7]:
                    counts[tuple(int(v) for v in px)] += 1
        return counts.most_common(1)[0][0] if counts else None

    def _image_solid(self, base, ly, lx):
        img = self._images.get(base)
        if img is None:
            return self.default_solid
        cw, ch = self.cell_size
        cols = max(1, img.width // cw)
        rows = max(1, img.height // ch)
        ly = min(ly, rows - 1)
        lx = min(lx, cols - 1)
        row = ly if self.y_from_top else (rows - 1 - ly)
        crop = np.asarray(img.crop((lx * cw, row * ch, (lx + 1) * cw, (row + 1) * ch)))
        if crop.size == 0:
            return self.default_solid
        empty = crop[..., 3] <= self.alpha_threshold
        if self.bg_color is not None:
            diff = np.abs(crop[..., :3].astype(int) - np.array(self.bg_color)).sum(axis=-1)
            empty |= diff <= self.color_tolerance
        return float((~empty).mean()) >= self.coverage_threshold

    # -- public API -------------------------------------------------------
    def is_solid(self, base, ly=0, lx=0) -> bool:
        if base in (BACKGROUND, PADDING, UNCOLLAPSED):
            return False
        key = (base, ly, lx) if self.mode == "image" else base
        if key in self._cache:
            return self._cache[key]

        if self.mode == "non_background":
            val = True
        elif self.mode == "classes":
            cls = self.class_map.get(base)
            if cls is None:
                if base not in self._warned:
                    self._warned.add(base)
                    print(f"[solidity] no class for {base!r} -> "
                          f"{'solid' if self.default_solid else 'passable'}")
                val = self.default_solid
            else:
                val = cls.lower() in SOLID_CLASSES
        elif self.mode == "image":
            val = self._image_solid(base, ly, lx)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")

        self._cache[key] = val
        return val

    def classify(self, base) -> str:
        return self.class_map.get(base, "unknown")


def make_class_map_stub(tiles_dir) -> dict:
    """Emit {tile_name: "TODO"} for every tile so it can be hand-labelled."""
    names = sorted(f for f in os.listdir(tiles_dir) if f.lower().endswith(".png"))
    stub = {BACKGROUND: "background"}
    stub.update({n: "TODO" for n in names})
    return stub


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


def structural_metrics(grid, id_to_tile, solidity: SolidityModel,
                       cfg: LevelConfig = LevelConfig(), ngram_size=3):
    """
    Returns (metrics_dict, ngram_counter, debug_dict).

    debug_dict carries the masks and profiles so results can be visually
    audited (see dump_occupancy_ascii) and so callers can compute their own
    derived statistics without re-deriving the masks.
    """
    names, lys, lxs = grid_to_arrays(grid, id_to_tile)
    H, W = names.shape

    # ---- 1. occupancy ---------------------------------------------------
    solid = np.zeros((H, W), dtype=bool)
    uncollapsed = names == UNCOLLAPSED
    for y in range(H):
        for x in range(W):
            if uncollapsed[y, x]:
                continue
            solid[y, x] = solidity.is_solid(names[y, x], lys[y, x], lxs[y, x])

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

def ngram_counts(name_grid, k=3):
    c = Counter()
    H, W = name_grid.shape
    for y in range(H - k + 1):
        for x in range(W - k + 1):
            c[tuple(name_grid[y:y + k, x:x + k].flatten())] += 1
    return c


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


def pattern_precision_recall(gen_counts, corpus_counts):
    """
    Treats the training corpus as ground truth for local structure.

      precision : share of generated pattern instances that occur in the corpus
                  -> low means the generator invents structures Mario never has
      recall    : share of distinct corpus patterns that the generator produced
                  -> low means mode collapse / limited vocabulary use
      novel_rate: share of DISTINCT generated patterns unseen in the corpus

    This catches both failure directions that a single KL number hides.
    """
    if not gen_counts:
        return {"pattern_precision": 0.0, "pattern_recall": 0.0, "novel_rate": 0.0}
    total = sum(gen_counts.values())
    in_corpus = sum(v for k, v in gen_counts.items() if k in corpus_counts)
    distinct_novel = sum(1 for k in gen_counts if k not in corpus_counts)
    recall = (sum(1 for k in corpus_counts if k in gen_counts) / len(corpus_counts)
              if corpus_counts else 0.0)
    return {
        "pattern_precision": round(in_corpus / total, 4),
        "pattern_recall": round(recall, 4),
        "novel_rate": round(distinct_novel / len(gen_counts), 4),
    }


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


def longest_verbatim_run(name_grid, corpus_grid, max_len=64):
    """
    Longest full-height column run that appears verbatim in the corpus.
    Detects memorisation: WFC with large overlapping patterns often reproduces
    long stretches of a source level, which inflates every fidelity metric.
    """
    if corpus_grid is None or name_grid.shape[0] != corpus_grid.shape[0]:
        return 0
    gen_cols = ["|".join(map(str, name_grid[:, x])) for x in range(name_grid.shape[1])]
    cor_cols = ["|".join(map(str, corpus_grid[:, x])) for x in range(corpus_grid.shape[1])]
    corpus_str = "\u0001".join(cor_cols)
    best = 0
    for i in range(len(gen_cols)):
        for L in range(best + 1, min(max_len, len(gen_cols) - i) + 1):
            if "\u0001".join(gen_cols[i:i + L]) in corpus_str:
                best = L
            else:
                break
    return best


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
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--stub":
        print(json.dumps(make_class_map_stub(sys.argv[2]), indent=2))
    else:
        print(__doc__)