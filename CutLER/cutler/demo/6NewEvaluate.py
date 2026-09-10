"""
6NewEvaluate.py
===============
Evaluation harness for the WFC level generator, run over every level.

Levels are discovered from LEVELS_DIR. For each one, the generation grid is
taken from the source level image itself - width and height in pixels divided
by CELL_SIZE, rounded up - so the generated levels are the same size as the
original and the ratio counts are comparable.

Occupancy rule: "B" is background, every other tile is an object. That is the
only rule, it needs no labels, and it applies identically to both batches
(complex clones fold to the core tile occupying their cell).

Requires, in the same folder:
    _55generate.py                    the solver
    metrics_topology.py               per-level structural metrics
    complex_structure.py              macro emergence analysis

Writes four files per level into OUTPUT_FOLDER:
    <level>_new_evaluation_metrics_raw.csv    per-level metrics, both batches
    <level>_new_comparison_summary.txt        Core vs Complex with bootstrap CIs
    <level>_complex_structure_emergence.csv   per-macro emergence detail
    <level>_complex_structure_emergence.txt   emergence summary
"""

import os
import re
import glob
import json
import math
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from scipy.stats import entropy, mannwhitneyu

from _55generate import run_generation
from metrics_topology import (
    LevelConfig, structural_metrics, pairwise_diversity, js_distance,
    parse_cell_name,
)
from complex_structure import (
    load_macro_templates, fold_grid, emergence_report, layout_diagnostics,
    cross_check_ratios,
)

# ===========================================================================
# CONFIG
# ===========================================================================
LEVELS_DIR = os.path.join("demo", "imgs", "test")
CORE_ROOT = os.path.join("Generation", "0CoreDataset")
COMPLEX_ROOT = os.path.join("Generation", "0ComplexDataset")
OUTPUT_FOLDER = os.path.join("Generation", "Baselines", "all_levels")

DATASET_SUFFIX = "_data"        # <level stem> + this = dataset folder name
LEVEL_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
CELL_SIZE = 16                  # px per tile; grid size = ceil(level px / this)

NUM_LEVELS = 20
NGRAM_K = 3                     # window size for the n-gram statistics
EMERGENCE_K = (2, 3)            # sub-block sizes for partial emergence
MACRO_LAYOUT = "blueprint"      # shape the walk reconstructs and the solver
                                # stamps; "name" for raw source offsets

ONLY_LEVELS = ()                # e.g. ("mario-1-1",) to restrict a run
SKIP_EXISTING = False           # True to skip levels already written

LEVEL_CFG = LevelConfig(
    jump_height=4,              # SMB: ~4 tiles of upward reach
    jump_run=5,                 # ~5 tiles cleared horizontally
    body_height=2,              # big Mario is 2 tiles tall
    obstacle_threshold=1,       # a rise of >1 tile above the floor is an obstacle
    compute_reachability=True,
)

CLONE_PATTERN = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")


# ===========================================================================
# Level discovery
# ===========================================================================
def discover_levels():
    """
    Every level image in LEVELS_DIR that has both a Core and a Complex dataset.

    Returns a list of dicts with the level stem, its grid size derived from the
    image, and the two dataset paths. Levels missing a dataset are reported and
    skipped rather than failing the run.
    """
    found, skipped = [], []
    paths = []
    for ext in LEVEL_EXTS:
        paths += glob.glob(os.path.join(LEVELS_DIR, f"*{ext}"))

    for path in sorted(set(paths)):
        stem = os.path.splitext(os.path.basename(path))[0]
        if ONLY_LEVELS and stem not in ONLY_LEVELS:
            continue
        try:
            with Image.open(path) as im:
                px_w, px_h = im.size
        except Exception as e:
            skipped.append((stem, f"unreadable image: {e}"))
            continue

        core = os.path.join(CORE_ROOT, f"{stem}{DATASET_SUFFIX}")
        comp = os.path.join(COMPLEX_ROOT, f"{stem}{DATASET_SUFFIX}")
        missing = [n for n, d in (("core", core), ("complex", comp))
                   if not os.path.isdir(d)]
        if missing:
            skipped.append((stem, f"no {'/'.join(missing)} dataset"))
            continue

        found.append({
            "stem": stem,
            "image": path,
            "px": (px_w, px_h),
            "width": math.ceil(px_w / CELL_SIZE),
            "height": math.ceil(px_h / CELL_SIZE),
            "core": core,
            "complex": comp,
        })

    return found, skipped


# ===========================================================================
# Per-batch evaluation
# ===========================================================================
def evaluate_batch(dataset_dir, batch_label, num_levels, grid_width, grid_height):
    print(f"\n--- Generating {batch_label} Batch "
          f"({grid_width}x{grid_height}) ---")

    generated_data = run_generation(
        root_dir=dataset_dir,
        grid_width=grid_width,
        grid_height=grid_height,
        num_levels=num_levels,
        filename="newMetrics.png",
    )

    failed_generations = num_levels - len(generated_data)

    with open(os.path.join(dataset_dir, "ratios.json")) as f:
        target_ratios = json.load(f)

    # ---- 1x1 target distribution over folded core-tile names ---------------
    base_target_ratios = defaultdict(float)
    for k, v in target_ratios.items():
        base, _, _, _ = parse_cell_name(k)
        base_target_ratios[base] += float(v)

    vocab = list(base_target_ratios.keys())
    target_probs = np.array([float(base_target_ratios.get(k, 1e-5)) for k in vocab])
    target_probs /= target_probs.sum()

    batch_results = []
    batch_global_ngrams = Counter()
    folded_grids = []
    raw_grids = []

    print(f"--- Evaluating {batch_label} Batch Matrices ---")
    for data in generated_data:
        grid = data["grid"]
        id_to_tile = data["id_to_tile"]

        # ---- 1x1 tile distribution ----------------------------------------
        complex_count = 0
        collapses = 0
        actual_counts = {k: 1e-5 for k in vocab}

        for row in grid:
            for cell in row:
                if isinstance(cell, set):
                    continue
                tile_name = id_to_tile.get(cell, str(cell))
                if tile_name == "P":
                    continue
                match = CLONE_PATTERN.match(tile_name)
                if match:
                    base_name = f"{match.group(2)}.png"
                    if "_y0_x0" in tile_name:
                        complex_count += 1
                        collapses += 1
                else:
                    base_name = tile_name
                    collapses += 1
                if base_name in actual_counts:
                    actual_counts[base_name] += 1

        actual_probs = np.array([actual_counts[k] for k in vocab])
        actual_probs /= actual_probs.sum()
        kl_div = entropy(actual_probs, target_probs)

        # ---- structural / topological metrics ------------------------------
        topo, level_ngrams, _debug = structural_metrics(
            grid, id_to_tile, LEVEL_CFG, ngram_size=NGRAM_K)

        batch_global_ngrams.update(level_ngrams)
        folded_grids.append(fold_grid(grid, id_to_tile))
        raw_grids.append((grid, id_to_tile))

        row_out = {
            "Batch": batch_label,
            "File": os.path.basename(data["path"]),
            "Path": data["path"],
            "Time (s)": round(data["time"], 3),
            "Retries": data["retries"],
            "Collapses": collapses,
            "Complex Pieces": complex_count,
            "KL-Div (1x1)": round(kl_div, 4),
            "pHash": str(imagehash.phash(Image.open(data["path"]))),
        }
        row_out.update(topo)
        batch_results.append(row_out)

    diversity = pairwise_diversity(folded_grids)
    print(f"{batch_label} intra-batch diversity "
          f"(mean pairwise Hamming): {diversity}")

    extras = {"diversity": diversity, "folded_grids": folded_grids,
              "raw_grids": raw_grids}
    return batch_results, failed_generations, batch_global_ngrams, extras


# ===========================================================================
# Cross-batch statistics
# ===========================================================================
def _ci95(series, n_boot=2000, seed=0):
    """Bootstrap 95% CI of the mean - point estimates alone mislead at n=20."""
    vals = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy(float)
    if vals.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _compare(core_df, comp_df, col):
    if col not in core_df or col not in comp_df:
        return None
    a = pd.to_numeric(core_df[col], errors="coerce").dropna()
    b = pd.to_numeric(comp_df[col], errors="coerce").dropna()
    if a.empty or b.empty:
        return None
    a_lo, a_hi = _ci95(a)
    b_lo, b_hi = _ci95(b)
    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
        p = "identical"                      # avoids reporting a nan p-value
    else:
        try:
            p = f"p={mannwhitneyu(a, b, alternative='two-sided').pvalue:.4f}"
        except ValueError:
            p = "p=n/a"
    return (f"{col:<24} Core {a.mean():9.3f} [{a_lo:.2f}, {a_hi:.2f}]   "
            f"Complex {b.mean():9.3f} [{b_lo:.2f}, {b_hi:.2f}]   {p}")


SECTIONS = (
    ("1. PERFORMANCE & STABILITY",
     ["Time (s)", "Retries", "Collapses", "Complex Pieces", "Uncollapsed"]),
    ("2. STATISTICAL DISTRIBUTIONS",
     ["KL-Div (1x1)", "N-Gram Entropy", "Compression Ratio", "Repetition Peak"]),
    ("3. GROUND TOPOLOGY (ground-connected structure only)",
     ["Ground Coverage", "Ground Level", "Roughness (std)",
      "Roughness (mean step)", "Elevation Std", "Linearity", "Slope"]),
    ("4. FEATURES & RHYTHM",
     ["Solid Density", "Gaps", "Mean Gap", "Max Gap", "Unjumpable Gaps",
      "Obstacles", "Mean Obstacle Height", "Max Obstacle Height",
      "Mean Obstacle Width", "Unjumpable Obstacles", "Platforms",
      "Avg Plat Width", "Avg Plat Clearance", "Avg Plat Thickness",
      "Unreachable Platforms", "Hazard Density /10col", "Components"]),
    ("5. STRUCTURAL PLAYABILITY (heuristic jump model, optimistic)",
     ["Reachable %", "Standable Cells"]),
)


def cross_batch_analysis(level, core_results, complex_results, out_dir,
                         core_fails, complex_fails, core_ngrams, complex_ngrams,
                         num_levels, core_extras=None, complex_extras=None):
    print("\n--- Running Cross-Batch Analysis ---")

    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)
    core_extras = core_extras or {}
    complex_extras = complex_extras or {}

    S = [f"=== GENERATION EVALUATION SUMMARY: {level['stem']} ===",
         f"grid {level['width']}x{level['height']} tiles, taken from the source "
         f"level ({level['px'][0]}x{level['px'][1]} px / {CELL_SIZE}).",
         f"n={num_levels} requested per batch; "
         f"[lo, hi] = bootstrap 95% CI of the mean; "
         f"p from two-sided Mann-Whitney U.",
         "Occupancy rule: B is background, every other tile is an object.",
         "",
         "Reachability and platform clearance assume every non-background tile",
         "can be stood on, so decorative tiles act as floors. Read those as",
         "relative comparisons between batches, not absolute playability."]

    if core_df.empty or comp_df.empty:
        S.append("\nOne batch is empty - no comparison possible.")
    else:
        for title, cols in SECTIONS:
            S.append("")
            S.append(title)
            for col in cols:
                line = _compare(core_df, comp_df, col)
                if line:
                    S.append(line)

            if title.startswith("1."):
                t_core = core_df["Time (s)"].mean()
                t_comp = comp_df["Time (s)"].mean()
                S.append(f"Compute cost multiplier: Complex is "
                         f"{(t_comp / t_core if t_core else 0):.2f}x slower")
                S.append(f"Failed generations: Core {core_fails}/{num_levels} | "
                         f"Complex {complex_fails}/{num_levels}")

            if title.startswith("2."):
                S.append(f"Core vs Complex n-gram ({NGRAM_K}x{NGRAM_K}) "
                         f"JS-distance: {js_distance(core_ngrams, complex_ngrams):.4f}")
                S.append(f"Intra-batch diversity (mean pairwise Hamming): "
                         f"Core {core_extras.get('diversity')} | "
                         f"Complex {complex_extras.get('diversity')}")

            if title.startswith("5.") and "Completable" in core_df:
                S.append(f"Completable: Core {int(core_df['Completable'].sum())}"
                         f"/{len(core_df)} | Complex "
                         f"{int(comp_df['Completable'].sum())}/{len(comp_df)}")

    text = "\n".join(S)
    print("\n" + text)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{level['stem']}_new_comparison_summary.txt")
    with open(path, "w") as f:
        f.write(text)


# ===========================================================================
# Macro emergence
# ===========================================================================
def run_emergence(level, core_grids, complex_grids, out_dir, complex_raw=None):
    """
    How many of the complex macros does the Core generator build unaided?

    The macro list comes from the complex dataset's declared vocabulary
    (adjacency_rules.txt, cross-checked against ratios.json), so it covers
    every macro the solver was permitted to place - not only those that
    appeared in the output.
    """
    print("\n--- Complex Structure Emergence ---")
    complex_folder = level["complex"]
    try:
        templates = load_macro_templates(complex_folder, layout=MACRO_LAYOUT)
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping emergence analysis: {e}")
        return None
    if not templates:
        print("Skipping emergence analysis: no macros found in the vocabulary.")
        return None

    return emergence_report(
        templates, core_grids, complex_grids,
        k_values=EMERGENCE_K,
        diagnostics=layout_diagnostics(complex_folder),
        ratio_check=cross_check_ratios(complex_folder, templates),
        complex_raw=complex_raw,
        out_dir=out_dir,
        out_prefix=f"{level['stem']}_",
        write_audit=False,
    )


# ===========================================================================
def evaluate_level(level, out_dir):
    """Run both batches, the comparison and the emergence analysis for one level."""
    print("\n" + "=" * 74)
    print(f"LEVEL: {level['stem']}   "
          f"source {level['px'][0]}x{level['px'][1]} px -> "
          f"grid {level['width']}x{level['height']} tiles")
    print("=" * 74)

    all_results = []

    core_results, core_fails, core_ngrams, core_extras = evaluate_batch(
        level["core"], "Core", NUM_LEVELS, level["width"], level["height"])
    all_results.extend(core_results)

    complex_results, complex_fails, complex_ngrams, complex_extras = evaluate_batch(
        level["complex"], "Complex", NUM_LEVELS, level["width"], level["height"])
    all_results.extend(complex_results)

    if core_results and complex_results:
        cross_batch_analysis(
            level, core_results, complex_results, out_dir,
            core_fails, complex_fails, core_ngrams, complex_ngrams,
            NUM_LEVELS, core_extras, complex_extras)

    if core_extras.get("folded_grids"):
        run_emergence(level,
                      core_extras["folded_grids"],
                      complex_extras.get("folded_grids"),
                      out_dir,
                      complex_raw=complex_extras.get("raw_grids"))

    if all_results:
        df = pd.DataFrame(all_results).drop(columns=["Path"])
        out = os.path.join(out_dir,
                           f"{level['stem']}_new_evaluation_metrics_raw.csv")
        df.to_csv(out, index=False)
        print(f"\nRaw metrics exported to {out}")

    return {"level": level["stem"],
            "core_ok": len(core_results), "core_failed": core_fails,
            "complex_ok": len(complex_results), "complex_failed": complex_fails}


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    levels, skipped = discover_levels()

    print(f"Found {len(levels)} level(s) with both datasets in {LEVELS_DIR}")
    for lv in levels:
        print(f"   {lv['stem']:<44} {lv['width']}x{lv['height']} tiles")
    for stem, why in skipped:
        print(f"   SKIP {stem:<39} {why}")
    if not levels:
        print("Nothing to do.")
        return

    done, failed = [], []
    for i, level in enumerate(levels, 1):
        target = os.path.join(
            OUTPUT_FOLDER, f"{level['stem']}_new_evaluation_metrics_raw.csv")
        if SKIP_EXISTING and os.path.exists(target):
            print(f"\n[{i}/{len(levels)}] {level['stem']}: already done, skipping")
            continue
        print(f"\n[{i}/{len(levels)}] {level['stem']}")
        try:
            done.append(evaluate_level(level, OUTPUT_FOLDER))
        except Exception:
            # one bad level must not lose the whole run
            print(f"!! {level['stem']} failed:\n{traceback.format_exc()}")
            failed.append(level["stem"])

    print("\n" + "=" * 74)
    print("RUN COMPLETE")
    print("=" * 74)
    print(f"{'level':<44}{'core':>10}{'complex':>10}")
    for d in done:
        core_txt = "{}/{}".format(d["core_ok"], NUM_LEVELS)
        comp_txt = "{}/{}".format(d["complex_ok"], NUM_LEVELS)
        print(f"{d['level']:<44}{core_txt:>10}{comp_txt:>10}")
    if failed:
        print(f"\nfailed levels: {failed}")
    print(f"\nAll output in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()