"""
6evaluate.py
============
Evaluation harness for the WFC level generator.

Occupancy rule: "B" is background, every other tile is an object. That is the
only rule, it needs no labels, and it applies identically to both batches
(complex clones fold to the core tile occupying their cell).

Requires, in the same folder:
    _55generate.py                    the solver
    metrics_topology.py               per-level structural metrics
    complex_structure_emergence.py    macro emergence analysis

Writes into BASELINE_FOLDER:
    new_evaluation_metrics_raw.csv    per-level metrics, both batches
    new_comparison_summary.txt        Core vs Complex with bootstrap CIs
    occupancy_<batch>.txt             ASCII map of the first level
    complex_structure_emergence.csv   per-macro emergence detail
    complex_structure_emergence.txt   emergence summary
    macro_geometry_audit.csv          macro layout consistency audit
"""

import os
import re
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from scipy.stats import entropy, mannwhitneyu

from _55generate import run_generation
from metrics_topology import (
    LevelConfig, structural_metrics, dump_occupancy_ascii,
    pairwise_diversity, js_distance, parse_cell_name,
)
from complex_structure import (
    load_macro_templates, fold_grid, emergence_report, layout_diagnostics,
    cross_check_ratios,
)

# ===========================================================================
# CONFIG
# ===========================================================================
CORE_FOLDER = os.path.join("Generation", "0CoreDataset/mario-1-2_data")
COMPLEX_FOLDER = os.path.join("Generation", "0ComplexDataset/mario-1-2_data")
BASELINE_FOLDER = os.path.join("Generation", "Baselines/mario 1-2")

GRID_WIDTH = 160
GRID_HEIGHT = 13
NUM_LEVELS = 20

NGRAM_K = 3                     # window size for the n-gram statistics
EMERGENCE_K = (2, 3)            # sub-block sizes for partial emergence
MACRO_LAYOUT = "blueprint"      # shape the walk reconstructs and the solver
                                # stamps; "name" for raw source offsets

LEVEL_CFG = LevelConfig(
    jump_height=4,              # SMB: ~4 tiles of upward reach
    jump_run=5,                 # ~5 tiles cleared horizontally
    body_height=2,              # big Mario is 2 tiles tall
    obstacle_threshold=1,       # a rise of >1 tile above the floor is an obstacle
    compute_reachability=True,
)

CLONE_PATTERN = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")


# ===========================================================================
# Per-batch evaluation
# ===========================================================================
def evaluate_batch(dataset_dir, batch_label, num_levels, grid_width, grid_height,
                   debug_dir=None):
    print(f"\n--- Generating {batch_label} Batch ---")

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
    for idx, data in enumerate(generated_data):
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
        topo, level_ngrams, debug = structural_metrics(
            grid, id_to_tile, LEVEL_CFG, ngram_size=NGRAM_K)

        batch_global_ngrams.update(level_ngrams)
        folded_grids.append(fold_grid(grid, id_to_tile))
        raw_grids.append((grid, id_to_tile))

        if idx == 0 and debug_dir:
            dump_occupancy_ascii(
                debug, os.path.join(debug_dir, f"occupancy_{batch_label}.txt"))

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


def cross_batch_analysis(core_results, complex_results, baseline_dir,
                         core_fails, complex_fails, core_ngrams, complex_ngrams,
                         num_levels, core_extras=None, complex_extras=None):
    print("\n--- Running Cross-Batch Analysis ---")

    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)
    core_extras = core_extras or {}
    complex_extras = complex_extras or {}

    S = ["=== GENERATION EVALUATION SUMMARY ===",
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
    os.makedirs(baseline_dir, exist_ok=True)
    with open(os.path.join(baseline_dir, "new_comparison_summary.txt"), "w") as f:
        f.write(text)


# ===========================================================================
# Macro emergence
# ===========================================================================
def run_emergence(complex_folder, core_grids, complex_grids, baseline_dir,
                  complex_raw=None):
    """
    How many of the complex macros does the Core generator build unaided?

    The macro list comes from the complex dataset's declared vocabulary
    (adjacency_rules.txt, cross-checked against ratios.json), so it covers
    every macro the solver was permitted to place - not only those that
    appeared in the output.
    """
    print("\n--- Complex Structure Emergence ---")
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
        out_dir=baseline_dir,
    )


# ===========================================================================
def main():
    os.makedirs(BASELINE_FOLDER, exist_ok=True)

    all_results = []
    core_results, complex_results = [], []
    core_fails = complex_fails = 0
    core_ngrams, complex_ngrams = Counter(), Counter()
    core_extras, complex_extras = {}, {}

    if os.path.exists(CORE_FOLDER):
        core_results, core_fails, core_ngrams, core_extras = evaluate_batch(
            CORE_FOLDER, "Core", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT,
            BASELINE_FOLDER)
        all_results.extend(core_results)
    else:
        print(f"Warning: Core directory {CORE_FOLDER} not found.")

    if os.path.exists(COMPLEX_FOLDER):
        complex_results, complex_fails, complex_ngrams, complex_extras = evaluate_batch(
            COMPLEX_FOLDER, "Complex", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT,
            BASELINE_FOLDER)
        all_results.extend(complex_results)
    else:
        print(f"Warning: Complex directory {COMPLEX_FOLDER} not found.")

    if core_results and complex_results:
        cross_batch_analysis(
            core_results, complex_results, BASELINE_FOLDER,
            core_fails, complex_fails, core_ngrams, complex_ngrams,
            NUM_LEVELS, core_extras, complex_extras)

    if core_extras.get("folded_grids") and os.path.exists(COMPLEX_FOLDER):
        run_emergence(COMPLEX_FOLDER,
                      core_extras["folded_grids"],
                      complex_extras.get("folded_grids"),
                      BASELINE_FOLDER,
                      complex_raw=complex_extras.get("raw_grids"))

    if all_results:
        df = pd.DataFrame(all_results).drop(columns=["Path"])
        out = os.path.join(BASELINE_FOLDER, "new_evaluation_metrics_raw.csv")
        df.to_csv(out, index=False)
        print(f"\nRaw metrics exported to {out}")


if __name__ == "__main__":
    main()