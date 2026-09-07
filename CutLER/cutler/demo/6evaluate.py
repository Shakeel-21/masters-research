import os
import json
import re
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from _55generate import run_generation

def extract_topological_metrics(grid, id_to_tile):
    """
    Analyzes the raw 2D grid to extract platformer-specific rhythm and structure metrics.
    """
    playable_grid = []
    clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")
    
    # 1. Normalize the grid (strip padding and fold complex tiles to atomic base)
    for y in range(1, len(grid) - 1):
        row = []
        for x in range(1, len(grid[0]) - 1):
            cell = grid[y][x]
            if isinstance(cell, set):
                val = "UNCOLLAPSED"
            else:
                t_name = id_to_tile.get(cell, str(cell))
                match = clone_pattern.match(t_name)
                if match:
                    val = f"{match.group(2)}.png"
                else:
                    val = t_name if t_name.endswith('.png') or t_name == 'B' else f"{t_name}.png"
                    if t_name == 'B': val = 'B'
            row.append(val)
        playable_grid.append(row)
        
    playable_grid = np.array(playable_grid)
    height, width = playable_grid.shape
    
    # --- 1. N-Gram Frequencies (3x3 blocks) ---
    ngrams = defaultdict(int)
    for y in range(height - 2):
        for x in range(width - 2):
            chunk = tuple(playable_grid[y:y+3, x:x+3].flatten())
            ngrams[chunk] += 1
            
    total_ngrams = sum(ngrams.values())
    ngram_probs = np.array(list(ngrams.values())) / total_ngrams
    ngram_entropy = entropy(ngram_probs) if len(ngram_probs) > 0 else 0
    
    # --- 2. Heightmap Skyline Variance ---
    skyline = np.zeros(width)
    for x in range(width):
        col = playable_grid[:, x]
        # Find highest solid block (first non-Background tile from the top)
        non_b_indices = np.where(col != 'B')[0]
        if len(non_b_indices) > 0:
            skyline[x] = height - non_b_indices[0] # Height measured from floor
        else:
            skyline[x] = 0 # Pit / Gap
            
    diffs = np.abs(np.diff(skyline))
    roughness = np.std(diffs)
    
    # --- 3. Feature Density & Rhythm ---
    is_gap = (skyline == 0)
    gap_changes = np.diff(is_gap.astype(int))
    num_gaps = np.sum(gap_changes == 1) + (1 if is_gap[0] else 0)
    
    # Calculate base ground level (most common height, ignoring gaps)
    non_zero_sky = skyline[skyline > 0]
    ground_level = float(pd.Series(non_zero_sky).mode()[0]) if len(non_zero_sky) > 0 else 0
        
    # Obstacles are vertical spikes above the base ground level
    is_obstacle = (skyline > ground_level)
    obs_changes = np.diff(is_obstacle.astype(int))
    num_obstacles = np.sum(obs_changes == 1) + (1 if is_obstacle[0] else 0)
    
    # Platforms are continuous non-gap segments
    is_platform = (skyline > 0)
    plat_changes = np.diff(is_platform.astype(int))
    num_platforms = np.sum(plat_changes == 1) + (1 if is_platform[0] else 0)
    avg_plat_width = np.sum(is_platform) / num_platforms if num_platforms > 0 else 0
    
    return {
        'N-Gram Entropy': round(ngram_entropy, 4),
        'Roughness (std)': round(roughness, 4),
        'Gaps': num_gaps,
        'Obstacles': num_obstacles,
        'Avg Plat Width': round(avg_plat_width, 2)
    }, ngrams

def evaluate_batch(dataset_dir, batch_label, num_levels, grid_width, grid_height):
    print(f"\n--- Generating {batch_label} Batch ---")
    
    generated_data = run_generation(
        root_dir=dataset_dir,
        grid_width=grid_width,
        grid_height=grid_height,
        num_levels=num_levels,
        filename= "optTest.png"
    )
    
    failed_generations = num_levels - len(generated_data)
    
    with open(os.path.join(dataset_dir, "ratios.json"), 'r') as f:
        target_ratios = json.load(f)
        
    clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")
    
    base_target_ratios = defaultdict(float)
    for k, v in target_ratios.items():
        match = clone_pattern.match(k)
        if match:
            base_name = f"{match.group(2)}.png"
        else:
            base_name = k if k.endswith('.png') or k == 'B' else f"{k}.png"
            if k == 'B': base_name = 'B'
            
        base_target_ratios[base_name] += float(v)
        
    vocab = list(base_target_ratios.keys())
    target_probs = np.array([float(base_target_ratios.get(k, 1e-5)) for k in vocab])
    target_probs /= target_probs.sum()
    
    batch_results = []
    batch_global_ngrams = defaultdict(int)
    
    print(f"--- Evaluating {batch_label} Batch Matrices ---")
    for data in generated_data:
        grid = data['grid']
        id_to_tile = data['id_to_tile']
        
        complex_count = 0
        collapses = 0
        actual_counts = {k: 1e-5 for k in vocab} 
        
        for row in grid:
            for cell in row:
                if not isinstance(cell, set):
                    tile_name = id_to_tile.get(cell, str(cell))
                    if tile_name == "P": continue 
                        
                    match = clone_pattern.match(tile_name)
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
        
        # Extract new topological metrics
        topo_metrics, level_ngrams = extract_topological_metrics(grid, id_to_tile)
        
        # Accumulate global N-grams for inter-batch JS-Divergence
        for k, v in level_ngrams.items():
            batch_global_ngrams[k] += v
        
        batch_results.append({
            'Batch': batch_label,
            'File': os.path.basename(data['path']),
            'Path': data['path'],
            'Time (s)': round(data['time'], 3),
            'Retries': data['retries'],
            'Collapses': collapses,
            'Complex Pieces': complex_count,
            'KL-Div (1x1)': round(kl_div, 4),
            'N-Gram Entropy': topo_metrics['N-Gram Entropy'],
            'Roughness (std)': topo_metrics['Roughness (std)'],
            'Gaps': topo_metrics['Gaps'],
            'Obstacles': topo_metrics['Obstacles'],
            'Avg Plat Width': topo_metrics['Avg Plat Width'],
            'pHash': img_hash
        })

    return batch_results, failed_generations, batch_global_ngrams

def cross_batch_analysis(core_results, complex_results, baseline_dir, core_fails, complex_fails, core_ngrams, complex_ngrams, num_levels):
    print("\n--- Running Topological Cross-Batch Analysis ---")

    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)

    # Calculate Inter-Batch N-Gram JS Distance
    all_ngram_keys = list(set(core_ngrams.keys()).union(set(complex_ngrams.keys())))
    if all_ngram_keys:
        core_vec = np.array([core_ngrams.get(k, 0) for k in all_ngram_keys], dtype=float)
        comp_vec = np.array([complex_ngrams.get(k, 0) for k in all_ngram_keys], dtype=float)
        
        core_vec /= (core_vec.sum() or 1)
        comp_vec /= (comp_vec.sum() or 1)
        
        js_dist = jensenshannon(core_vec, comp_vec)
    else:
        js_dist = 0.0

    summary = []
    summary.append("=== GENERATION EVALUATION SUMMARY ===")
    
    if not core_df.empty and not comp_df.empty:
        t_core, t_comp = core_df['Time (s)'].mean(), comp_df['Time (s)'].mean()
        time_mult = t_comp / t_core if t_core > 0 else 0
        col_core, col_comp = core_df['Collapses'].mean(), comp_df['Collapses'].mean()
        
        summary.append(f"\n1. PERFORMANCE & STABILITY")
        summary.append(f"Core Avg Time: {t_core:.2f}s | Complex Avg Time: {t_comp:.2f}s")
        summary.append(f"Compute Cost Multiplier: Complex is {time_mult:.2f}x slower")
        summary.append(f"Core Avg Collapses: {col_core:.1f} | Complex Avg Collapses: {col_comp:.1f}")
        
        r_core, r_comp = core_df['Retries'].mean(), comp_df['Retries'].mean()
        summary.append(f"Core Avg Retries: {r_core:.1f} | Complex Avg Retries: {r_comp:.1f}")
        summary.append(f"Core Failed Generations: {core_fails}/{num_levels} | Complex Failed Generations: {complex_fails}/{num_levels}")

        kl_core, kl_comp = core_df['KL-Div (1x1)'].mean(), comp_df['KL-Div (1x1)'].mean()
        summary.append(f"\n2. STATISTICAL DISTRIBUTIONS")
        summary.append(f"Core Base KL-Div: {kl_core:.4f} | Complex Base KL-Div: {kl_comp:.4f}")
        summary.append(f"Core vs Complex N-Gram (3x3) JS-Distance: {js_dist:.4f}")
        
        summary.append(f"\n3. TOPOLOGICAL SPATIAL RHYTHM")
        summary.append(f"Core Avg N-Gram Entropy: {core_df['N-Gram Entropy'].mean():.4f} | Complex Avg N-Gram Entropy: {comp_df['N-Gram Entropy'].mean():.4f}")
        summary.append(f"Core Avg Roughness (std): {core_df['Roughness (std)'].mean():.4f} | Complex Avg Roughness (std): {comp_df['Roughness (std)'].mean():.4f}")
        summary.append(f"Core Avg Gaps: {core_df['Gaps'].mean():.1f} | Complex Avg Gaps: {comp_df['Gaps'].mean():.1f}")
        summary.append(f"Core Avg Obstacles: {core_df['Obstacles'].mean():.1f} | Complex Avg Obstacles: {comp_df['Obstacles'].mean():.1f}")
        summary.append(f"Core Avg Platform Width: {core_df['Avg Plat Width'].mean():.1f} | Complex Avg Platform Width: {comp_df['Avg Plat Width'].mean():.1f}")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    summary_file_path = os.path.join(baseline_dir, "new_comparison_summary.txt")
    with open(summary_file_path, "w") as f:
        f.write(summary_text)

def main():
    CORE_FOLDER = os.path.join("Generation", "0CoreDataset/mario_8_data")
    COMPLEX_FOLDER = os.path.join("Generation", "0ComplexDataset/mario_8_data")
    BASELINE_FOLDER = os.path.join("Generation","Baselines/mario_8")
    
    os.makedirs(BASELINE_FOLDER, exist_ok=True)
    
    GRID_WIDTH = 214
    GRID_HEIGHT = 14
    NUM_LEVELS = 10
    
    all_results = []
    core_results = []
    complex_results = []
    
    core_fails = 0
    complex_fails = 0
    core_ngrams = {}
    complex_ngrams = {}
    
    if os.path.exists(CORE_FOLDER):
        core_results, core_fails, core_ngrams = evaluate_batch(CORE_FOLDER, "Core", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(core_results)
    else:
        print(f"Warning: Core directory {CORE_FOLDER} not found.")

    if os.path.exists(COMPLEX_FOLDER):
        complex_results, complex_fails, complex_ngrams = evaluate_batch(COMPLEX_FOLDER, "Complex", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(complex_results)
    else:
        print(f"Warning: Complex directory {COMPLEX_FOLDER} not found.")

    if core_results and complex_results:
        cross_batch_analysis(core_results, complex_results, BASELINE_FOLDER, core_fails, complex_fails, core_ngrams, complex_ngrams, NUM_LEVELS)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.drop(columns=['Path']) 
        
        output_file = os.path.join(BASELINE_FOLDER, "new_evaluation_metrics_raw.csv")
        df.to_csv(output_file, index=False)
        print(f"\nRaw metrics exported to {output_file}")

if __name__ == "__main__":
    main()