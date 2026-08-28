import os
import json
import re
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from _55generate import run_generation

def calculate_ssim(img_path1, img_path2):
    img1 = np.array(Image.open(img_path1).convert('L'))
    img2 = np.array(Image.open(img_path2).convert('L'))
    
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
    return ssim(img1, img2, data_range=255)

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
    
    # Fold target ratios down to their base atomic identifiers
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
                    
                    if tile_name == "P":
                        continue # Ignore pre-placed padding
                        
                    match = clone_pattern.match(tile_name)
                    if match:
                        base_name = f"{match.group(2)}.png"
                        if "_y0_x0" in tile_name:
                            complex_count += 1
                            collapses += 1 # 1 collapse call stamped this entire macro
                    else:
                        base_name = tile_name
                        collapses += 1 # 1 collapse call stamped this atomic tile
                        
                    if base_name in actual_counts:
                        actual_counts[base_name] += 1
                        
        actual_probs = np.array([actual_counts[k] for k in vocab])
        actual_probs /= actual_probs.sum()
        
        kl_div = entropy(actual_probs, target_probs)
        shannon_ent = entropy(actual_probs)
        img_hash = str(imagehash.phash(Image.open(data['path'])))
        
        batch_results.append({
            'Batch': batch_label,
            'File': os.path.basename(data['path']),
            'Path': data['path'],
            'Time (s)': round(data['time'], 3),
            'Retries': data['retries'],
            'Collapses': collapses,
            'Complex Pieces': complex_count,
            'KL-Div': round(kl_div, 4),
            'Entropy': round(shannon_ent, 4),
            'pHash': img_hash,
            'Inter-Batch SSIM': 0.0,
            'Baseline SSIM': 0.0
        })

    print(f"--- Calculating Perceptual Variance for {batch_label} ---")
    for i, res1 in enumerate(batch_results):
        ssim_scores = []
        for j, res2 in enumerate(batch_results):
            if i != j:
                score = calculate_ssim(res1['Path'], res2['Path'])
                ssim_scores.append(score)
        
        res1['Inter-Batch SSIM'] = round(np.mean(ssim_scores), 4) if ssim_scores else 1.0

    return batch_results, failed_generations

def cross_batch_analysis(core_results, complex_results, baseline_dir, core_fails, complex_fails, num_levels):
    print("\n--- Running Cross-Batch Analysis ---")
    
    baseline_paths = []
    if os.path.exists(baseline_dir):
        baseline_paths = [os.path.join(baseline_dir, f) for f in os.listdir(baseline_dir) if f.endswith('.png')]
    else:
        print(f"Warning: Baseline directory {baseline_dir} not found. Skipping baseline SSIM.")

    if baseline_paths:
        for batch in [core_results, complex_results]:
            for res in batch:
                base_scores = [calculate_ssim(res['Path'], bp) for bp in baseline_paths]
                res['Baseline SSIM'] = round(np.mean(base_scores), 4) if base_scores else 0.0

    cross_ssim_scores = []
    for c_res in core_results:
        for comp_res in complex_results:
            cross_ssim_scores.append(calculate_ssim(c_res['Path'], comp_res['Path']))
    
    avg_cross_ssim = round(np.mean(cross_ssim_scores), 4) if cross_ssim_scores else 0.0

    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)

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

        kl_core, kl_comp = core_df['KL-Div'].mean(), comp_df['KL-Div'].mean()
        ent_core, ent_comp = core_df['Entropy'].mean(), comp_df['Entropy'].mean()
        summary.append(f"\n2. DISTRIBUTIONAL SHIFTS (Base Atomic Form)")
        summary.append(f"Core Avg KL-Div: {kl_core:.4f} | Complex Avg KL-Div: {kl_comp:.4f}")
        summary.append(f"Core Avg Entropy: {ent_core:.4f} | Complex Avg Entropy: {ent_comp:.4f}")
        
        base_core = core_df['Baseline SSIM'].mean() if 'Baseline SSIM' in core_df else 0
        base_comp = comp_df['Baseline SSIM'].mean() if 'Baseline SSIM' in comp_df else 0
        summary.append(f"\n3. PERCEPTUAL SIMILARITY")
        summary.append(f"Core vs Baseline SSIM: {base_core:.4f}")
        summary.append(f"Complex vs Baseline SSIM: {base_comp:.4f}")
        summary.append(f"Core vs Complex Cross-Batch SSIM: {avg_cross_ssim:.4f}")
        
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    summary_file_path = os.path.join(baseline_dir, "comparison_summary.txt")
    with open(summary_file_path, "w") as f:
        f.write(summary_text)

def main():
    CORE_FOLDER = os.path.join("Generation", "0CoreDataset/mario-2-1_data")
    COMPLEX_FOLDER = os.path.join("Generation", "0ComplexDataset/mario-2-1_data")
    BASELINE_FOLDER = os.path.join("Generation","Baselines/mario 2-1 full")
    
    os.makedirs(BASELINE_FOLDER, exist_ok=True)
    
    GRID_WIDTH = 197
    GRID_HEIGHT = 13
    NUM_LEVELS = 10
    
    all_results = []
    core_results = []
    complex_results = []
    
    core_fails = 0
    complex_fails = 0
    
    if os.path.exists(CORE_FOLDER):
        core_results, core_fails = evaluate_batch(CORE_FOLDER, "Core", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(core_results)
    else:
        print(f"Warning: Core directory {CORE_FOLDER} not found.")

    if os.path.exists(COMPLEX_FOLDER):
        complex_results, complex_fails = evaluate_batch(COMPLEX_FOLDER, "Complex", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(complex_results)
    else:
        print(f"Warning: Complex directory {COMPLEX_FOLDER} not found.")

    if core_results and complex_results:
        cross_batch_analysis(core_results, complex_results, BASELINE_FOLDER, core_fails, complex_fails, NUM_LEVELS)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.drop(columns=['Path']) 
        
        output_file = os.path.join(BASELINE_FOLDER, "evaluation_metrics_raw.csv")
        df.to_csv(output_file, index=False)
        print(f"\nRaw metrics exported to {output_file}")

if __name__ == "__main__":
    main()