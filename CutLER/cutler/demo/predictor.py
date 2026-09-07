import os
import json
import re
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict, Counter
from scipy.stats import entropy, mannwhitneyu

from _55generate import run_generation
from metrics_topology import (
    SolidityModel, LevelConfig, structural_metrics, dump_occupancy_ascii,
    ngram_counts, js_distance, pattern_precision_recall, pairwise_diversity,
    longest_verbatim_run, parse_cell_name,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# "non_background" reproduces the old behaviour (every non-B tile is solid).
# "classes" uses tile_classes.json ({tile_name: "solid"|"coin"|"enemy"|...}).
#     generate a stub with:  python metrics_topology.py --stub <dataset>/tiles
# "image" reads per-cell occupancy out of the tile PNGs. This is the only mode
#     that measures multi-cell (complex) tiles fairly - see the caveat in
#     metrics_topology.SolidityModel.
SOLIDITY_MODE = "non_background"
LEVEL_CFG = LevelConfig(jump_height=4, jump_run=5, body_height=2,
                        obstacle_threshold=1, compute_reachability=True)
NGRAM_K = 3
DUMP_DEBUG_MAPS = True          # ASCII occupancy dump for the first level of each batch

CLONE_PATTERN = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")


# ---------------------------------------------------------------------------
# Optional: training-corpus patterns for fidelity / memorisation checks
# ---------------------------------------------------------------------------
def load_corpus_grids(dataset_dir):
    """
    Loads the source levels the rules were learned from, if available, as
    numpy string grids. Expected layout:

        <dataset_dir>/corpus_grids/*.txt
        one row per line, cells separated by '|', background written as 'B'

    Dump these from whichever earlier pipeline step parses the original levels.
    Returns [] if absent - every corpus-dependent metric is then skipped.
    """
    folder = os.path.join(dataset_dir, "corpus_grids")
    if not os.path.isdir(folder):
        return []
    grids = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(folder, fname)) as f:
            rows = [ln.rstrip("\n").split("|") for ln in f if ln.strip()]
        if rows:
            grids.append(np.array(rows, dtype=object))
    return grids


# ---------------------------------------------------------------------------
def evaluate_batch(dataset_dir, batch_label, num_levels, grid_width, grid_height,
                   debug_dir=None):
    print(f"\n--- Generating {batch_label} Batch ---")

    generated_data = run_generation(
        root_dir=dataset_dir,
        grid_width=grid_width,
        grid_height=grid_height,
        num_levels=num_levels,
        filename="optTest.png"
    )

    failed_generations = num_levels - len(generated_data)

    with open(os.path.join(dataset_dir, "ratios.json"), 'r') as f:
        target_ratios = json.load(f)

    # ---- solidity model -------------------------------------------------
    tiles_dir = os.path.join(dataset_dir, "tiles")
    class_path = os.path.join(dataset_dir, "tile_classes.json")
    class_map = {}
    if os.path.exists(class_path):
        with open(class_path) as f:
            class_map = json.load(f)
    solidity = SolidityModel(mode=SOLIDITY_MODE, class_map=class_map,
                             tiles_dir=tiles_dir if os.path.isdir(tiles_dir) else None)

    # ---- 1x1 target distribution over base tile names -------------------
    base_target_ratios = defaultdict(float)
    for k, v in target_ratios.items():
        base, _, _, _ = parse_cell_name(k)
        base_target_ratios[base] += float(v)

    vocab = list(base_target_ratios.keys())
    target_probs = np.array([float(base_target_ratios.get(k, 1e-5)) for k in vocab])
    target_probs /= target_probs.sum()

    # ---- corpus patterns (optional) -------------------------------------
    corpus_grids = load_corpus_grids(dataset_dir)
    corpus_ngrams = Counter()
    for cg in corpus_grids:
        corpus_ngrams.update(ngram_counts(cg, NGRAM_K))
    corpus_concat = (np.concatenate(corpus_grids, axis=1)
                     if corpus_grids and len({g.shape[0] for g in corpus_grids}) == 1
                     else None)
    if corpus_ngrams:
        print(f"Loaded {len(corpus_grids)} corpus levels "
              f"({len(corpus_ngrams)} distinct {NGRAM_K}x{NGRAM_K} patterns)")

    batch_results = []
    batch_global_ngrams = Counter()
    name_grids = []

    print(f"--- Evaluating {batch_label} Batch Matrices ---")
    for idx, data in enumerate(generated_data):
        grid = data['grid']
        id_to_tile = data['id_to_tile']

        # ---- 1x1 tile distribution (unchanged logic) --------------------
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

        img_hash = str(imagehash.phash(Image.open(data['path'])))

        # ---- corrected structural metrics --------------------------------
        topo, level_ngrams, debug = structural_metrics(
            grid, id_to_tile, solidity, LEVEL_CFG, ngram_size=NGRAM_K)

        batch_global_ngrams.update(level_ngrams)
        name_grids.append(debug["names"])

        if DUMP_DEBUG_MAPS and idx == 0 and debug_dir:
            dump_occupancy_ascii(
                debug, os.path.join(debug_dir, f"occupancy_{batch_label}.txt"))

        row_out = {
            'Batch': batch_label,
            'File': os.path.basename(data['path']),
            'Path': data['path'],
            'Time (s)': round(data['time'], 3),
            'Retries': data['retries'],
            'Collapses': collapses,
            'Complex Pieces': complex_count,
            'KL-Div (1x1)': round(kl_div, 4),
            'pHash': img_hash,
        }
        row_out.update(topo)

        # ---- corpus fidelity / memorisation ------------------------------
        if corpus_ngrams:
            row_out.update(pattern_precision_recall(level_ngrams, corpus_ngrams))
            row_out['ngram_JS_vs_corpus'] = round(
                js_distance(level_ngrams, corpus_ngrams), 4)
            row_out['longest_verbatim_cols'] = longest_verbatim_run(
                debug["names"], corpus_concat)

        batch_results.append(row_out)

    diversity = pairwise_diversity(name_grids)
    print(f"{batch_label} intra-batch diversity (mean pairwise Hamming): {diversity}")

    extras = {'diversity': diversity, 'corpus_ngrams': corpus_ngrams}
    return batch_results, failed_generations, batch_global_ngrams, extras


# ---------------------------------------------------------------------------
def _ci95(series):
    """Bootstrap 95% CI of the mean - n=10 means point estimates alone mislead."""
    vals = np.asarray(series, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0)
    boot = rng.choice(vals, size=(2000, vals.size), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _compare(core_df, comp_df, col):
    """Formatted 'core vs complex' line with CIs and a rank-sum p-value."""
    if col not in core_df or col not in comp_df:
        return None
    a = pd.to_numeric(core_df[col], errors='coerce').dropna()
    b = pd.to_numeric(comp_df[col], errors='coerce').dropna()
    if a.empty or b.empty:
        return None
    a_lo, a_hi = _ci95(a)
    b_lo, b_hi = _ci95(b)
    try:
        p = mannwhitneyu(a, b, alternative='two-sided').pvalue
        p_txt = f"p={p:.4f}"
    except ValueError:
        p_txt = "p=n/a"
    return (f"{col:<24} Core {a.mean():8.3f} [{a_lo:.2f}, {a_hi:.2f}]   "
            f"Complex {b.mean():8.3f} [{b_lo:.2f}, {b_hi:.2f}]   {p_txt}")


def cross_batch_analysis(core_results, complex_results, baseline_dir,
                         core_fails, complex_fails, core_ngrams, complex_ngrams,
                         num_levels, core_extras=None, complex_extras=None):
    print("\n--- Running Topological Cross-Batch Analysis ---")

    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)
    core_extras = core_extras or {}
    complex_extras = complex_extras or {}

    js_dist = js_distance(core_ngrams, complex_ngrams)

    summary = ["=== GENERATION EVALUATION SUMMARY ===",
               f"(n={num_levels} per batch; [lo, hi] = bootstrap 95% CI of the mean;",
               " p from two-sided Mann-Whitney U. Treat n=10 as exploratory.)"]

    if not core_df.empty and not comp_df.empty:
        summary.append("\n1. PERFORMANCE & STABILITY")
        for col in ['Time (s)', 'Retries', 'Collapses', 'Uncollapsed']:
            line = _compare(core_df, comp_df, col)
            if line:
                summary.append(line)
        t_core, t_comp = core_df['Time (s)'].mean(), comp_df['Time (s)'].mean()
        summary.append(f"Compute Cost Multiplier: Complex is "
                       f"{(t_comp / t_core if t_core else 0):.2f}x slower")
        summary.append(f"Failed Generations: Core {core_fails}/{num_levels} | "
                       f"Complex {complex_fails}/{num_levels}")

        summary.append("\n2. STATISTICAL DISTRIBUTIONS")
        for col in ['KL-Div (1x1)', 'N-Gram Entropy', 'Compression Ratio',
                    'ngram_JS_vs_corpus', 'pattern_precision', 'pattern_recall',
                    'novel_rate', 'longest_verbatim_cols']:
            line = _compare(core_df, comp_df, col)
            if line:
                summary.append(line)
        summary.append(f"Core vs Complex N-Gram ({NGRAM_K}x{NGRAM_K}) "
                       f"JS-Distance: {js_dist:.4f}")
        summary.append(f"Intra-batch diversity (pairwise Hamming): "
                       f"Core {core_extras.get('diversity')} | "
                       f"Complex {complex_extras.get('diversity')}")

        summary.append("\n3. GROUND TOPOLOGY (ground-connected structure only)")
        for col in ['Ground Coverage', 'Ground Level', 'Roughness (std)',
                    'Roughness (mean step)', 'Elevation Std', 'Linearity',
                    'Repetition Peak']:
            line = _compare(core_df, comp_df, col)
            if line:
                summary.append(line)

        summary.append("\n4. FEATURES & RHYTHM")
        for col in ['Gaps', 'Mean Gap', 'Max Gap', 'Unjumpable Gaps',
                    'Obstacles', 'Mean Obstacle Height', 'Max Obstacle Height',
                    'Unjumpable Obstacles', 'Platforms', 'Avg Plat Width',
                    'Avg Plat Clearance', 'Unreachable Platforms',
                    'Solid Density', 'Hazard Density /10col']:
            line = _compare(core_df, comp_df, col)
            if line:
                summary.append(line)

        summary.append("\n5. STRUCTURAL PLAYABILITY (heuristic jump model)")
        for col in ['Reachable %', 'Standable Cells']:
            line = _compare(core_df, comp_df, col)
            if line:
                summary.append(line)
        if 'Completable' in core_df:
            summary.append(f"Completable: Core {int(core_df['Completable'].sum())}"
                           f"/{len(core_df)} | Complex "
                           f"{int(comp_df['Completable'].sum())}/{len(comp_df)}")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(os.path.join(baseline_dir, "new_comparison_summary.txt"), "w") as f:
        f.write(summary_text)


def main():
    CORE_FOLDER = os.path.join("Generation", "0CoreDataset/mario_7_data")
    COMPLEX_FOLDER = os.path.join("Generation", "0ComplexDataset/mario_7_data")
    BASELINE_FOLDER = os.path.join("Generation", "Baselines/mario 7 full")

    os.makedirs(BASELINE_FOLDER, exist_ok=True)

    GRID_WIDTH = 214
    GRID_HEIGHT = 14
    NUM_LEVELS = 10

    all_results = []
    core_results, complex_results = [], []
    core_fails = complex_fails = 0
    core_ngrams, complex_ngrams = Counter(), Counter()
    core_extras = complex_extras = {}

    if os.path.exists(CORE_FOLDER):
        core_results, core_fails, core_ngrams, core_extras = evaluate_batch(
            CORE_FOLDER, "Core", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT, BASELINE_FOLDER)
        all_results.extend(core_results)
    else:
        print(f"Warning: Core directory {CORE_FOLDER} not found.")

    if os.path.exists(COMPLEX_FOLDER):
        complex_results, complex_fails, complex_ngrams, complex_extras = evaluate_batch(
            COMPLEX_FOLDER, "Complex", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT, BASELINE_FOLDER)
        all_results.extend(complex_results)
    else:
        print(f"Warning: Complex directory {COMPLEX_FOLDER} not found.")

    if core_results and complex_results:
        cross_batch_analysis(core_results, complex_results, BASELINE_FOLDER,
                             core_fails, complex_fails, core_ngrams, complex_ngrams,
                             NUM_LEVELS, core_extras, complex_extras)

    if all_results:
        df = pd.DataFrame(all_results).drop(columns=['Path'])
        output_file = os.path.join(BASELINE_FOLDER, "new_evaluation_metrics_raw.csv")
        df.to_csv(output_file, index=False)
        print(f"\nRaw metrics exported to {output_file}")


if __name__ == "__main__":
    main()