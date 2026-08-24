import os
import json
import re
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
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
        filename= "newEval.png"
    )
    
    with open(os.path.join(dataset_dir, "ratios.json"), 'r') as f:
        target_ratios = json.load(f)
        
    vocab = list(target_ratios.keys())
    target_probs = np.array([float(target_ratios.get(k, 1e-5)) for k in vocab])
    target_probs /= target_probs.sum()
    
    clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")
    batch_results = []
    
    print(f"--- Evaluating {batch_label} Batch Matrices ---")
    for data in generated_data:
        grid = data['grid']
        id_to_tile = data['id_to_tile']
        
        complex_count = 0
        actual_counts = {k: 1e-5 for k in vocab} 
        
        for row in grid:
            for cell in row:
                if not isinstance(cell, set):
                    tile_name = id_to_tile.get(cell, str(cell))
                    if tile_name in actual_counts:
                        actual_counts[tile_name] += 1
                    
                    if clone_pattern.match(tile_name):
                        if "_y0_x0" in tile_name:
                            complex_count += 1
                        
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

    return batch_results

def cross_batch_analysis(core_results, complex_results, baseline_dir):
    print("\n--- Running Cross-Batch Analysis ---")
    
    baseline_paths = []
    if os.path.exists(baseline_dir):
        baseline_paths = [os.path.join(baseline_dir, f) for f in os.listdir(baseline_dir) if f.endswith('.png')]
    else:
        print(f"Warning: Baseline directory {baseline_dir} not found. Skipping baseline SSIM.")

    # Calculate Baseline SSIM for both batches
    if baseline_paths:
        for batch in [core_results, complex_results]:
            for res in batch:
                base_scores = [calculate_ssim(res['Path'], bp) for bp in baseline_paths]
                res['Baseline SSIM'] = round(np.mean(base_scores), 4) if base_scores else 0.0

    # Calculate Core vs Complex SSIM
    cross_ssim_scores = []
    for c_res in core_results:
        for comp_res in complex_results:
            cross_ssim_scores.append(calculate_ssim(c_res['Path'], comp_res['Path']))
    
    avg_cross_ssim = round(np.mean(cross_ssim_scores), 4) if cross_ssim_scores else 0.0

    # Compile Summary Statistics
    core_df = pd.DataFrame(core_results)
    comp_df = pd.DataFrame(complex_results)

    summary = []
    summary.append("=== GENERATION EVALUATION SUMMARY ===")
    
    if not core_df.empty and not comp_df.empty:
        # Performance
        t_core, t_comp = core_df['Time (s)'].mean(), comp_df['Time (s)'].mean()
        time_mult = t_comp / t_core if t_core > 0 else 0
        summary.append(f"\n1. PERFORMANCE")
        summary.append(f"Core Avg Time: {t_core:.2f}s | Complex Avg Time: {t_comp:.2f}s")
        summary.append(f"Compute Cost Multiplier: Complex is {time_mult:.2f}x slower")
        
        r_core, r_comp = core_df['Retries'].mean(), comp_df['Retries'].mean()
        summary.append(f"Core Avg Retries: {r_core:.1f} | Complex Avg Retries: {r_comp:.1f}")

        # Distribution
        kl_core, kl_comp = core_df['KL-Div'].mean(), comp_df['KL-Div'].mean()
        ent_core, ent_comp = core_df['Entropy'].mean(), comp_df['Entropy'].mean()
        summary.append(f"\n2. DISTRIBUTIONAL SHIFTS")
        summary.append(f"Core Avg KL-Div: {kl_core:.4f} | Complex Avg KL-Div: {kl_comp:.4f}")
        summary.append(f"Core Avg Entropy: {ent_core:.4f} | Complex Avg Entropy: {ent_comp:.4f}")
        
        # Similarity
        base_core = core_df['Baseline SSIM'].mean() if 'Baseline SSIM' in core_df else 0
        base_comp = comp_df['Baseline SSIM'].mean() if 'Baseline SSIM' in comp_df else 0
        summary.append(f"\n3. PERCEPTUAL SIMILARITY")
        summary.append(f"Core vs Baseline SSIM: {base_core:.4f}")
        summary.append(f"Complex vs Baseline SSIM: {base_comp:.4f}")
        summary.append(f"Core vs Complex Cross-Batch SSIM: {avg_cross_ssim:.4f}")
        
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    with open("comparison_summary.txt", "w") as f:
        f.write(summary_text)

def main():
    # --- CONFIGURATION ---
    CORE_FOLDER = os.path.join("Generation", "0CoreDataset/mario-2-1_data")
    COMPLEX_FOLDER = os.path.join("Generation", "0ComplexDataset/mario-2-1_data")
    BASELINE_FOLDER = os.path.join("Generation","Baselines")
    
    GRID_WIDTH = 50
    GRID_HEIGHT = 14
    NUM_LEVELS = 10
    
    all_results = []
    core_results = []
    complex_results = []
    
    if os.path.exists(CORE_FOLDER):
        core_results = evaluate_batch(CORE_FOLDER, "Core", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(core_results)
    else:
        print(f"Warning: Core directory {CORE_FOLDER} not found.")

    if os.path.exists(COMPLEX_FOLDER):
        complex_results = evaluate_batch(COMPLEX_FOLDER, "Complex", NUM_LEVELS, GRID_WIDTH, GRID_HEIGHT)
        all_results.extend(complex_results)
    else:
        print(f"Warning: Complex directory {COMPLEX_FOLDER} not found.")

    # Execute Comparisons
    if core_results and complex_results:
        cross_batch_analysis(core_results, complex_results, BASELINE_FOLDER)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.drop(columns=['Path']) 
        
        output_file = "evaluation_metrics_raw.csv"
        df.to_csv(output_file, index=False)
        print(f"\nRaw metrics exported to {output_file}")

if __name__ == "__main__":
    main()