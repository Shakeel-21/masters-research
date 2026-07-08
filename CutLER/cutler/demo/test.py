import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_SEGMENTS = 'output_segments/transRemake'
INPUT_ORIGINAL = 'demo/imgs/test'
OUTPUT_FOLDER = 'output_segments/coreV10'

# Phase 1 Config (from test.py)
SIMILARITY_THRESHOLD = 1000.0
COLOR_ROUNDING = 50
BLUR_KERNEL = (5, 5)
BUCKET_SEARCH_RANGE = 20

# Phase 2 Config (from 2getCore.py)
GRID_CANDIDATES = [8, 16, 24, 32]
MATCH_THRESHOLD = 0.05
MIN_VISIBLE_PERCENT = 0.40  
MIN_VISIBLE_PIXELS = 128 # Used in Phase 1

def load_images_from_folder(folder, pattern='*.png'):
    path_obj = Path(folder).resolve()
    if not path_obj.exists():
        return []
    return list(path_obj.rglob(pattern))

def clean_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[img[:, :, 3] == 0] = [0, 0, 0, 0]
    return img

def preprocess_for_matching(img):
    processed = img.copy()
    bgr = processed[:, :, :3]
    blurred_bgr = cv2.GaussianBlur(bgr, BLUR_KERNEL, 0)
    quantized_bgr = (blurred_bgr // COLOR_ROUNDING) * COLOR_ROUNDING + COLOR_ROUNDING // 2
    processed[:, :, :3] = quantized_bgr
    return processed.astype(np.uint8)

def pad_to_target_transparent(img, target_size):
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img.copy()
    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    copy_h = min(h, target_size)
    copy_w = min(w, target_size)
    canvas[0:copy_h, 0:copy_w] = img[0:copy_h, 0:copy_w]
    return canvas

def get_mean_brightness(img):
    return int(np.mean(img[:, :, :3]))

def shrink_wrap_alpha(img_bgra):
    alpha_channel = img_bgra[:, :, 3]
    if cv2.countNonZero(alpha_channel) < 25: 
        return None
    y_indices, x_indices = np.where(alpha_channel > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return img_bgra[y_min:y_max+1, x_min:x_max+1]

def calculate_dynamic_grid(segment_paths):
    print("\n--- Calculating Dynamic Grid Size ---")
    measurements = []
    for path in segment_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        wrapped = shrink_wrap_alpha(clean_image(img))
        if wrapped is not None:
            h, w = wrapped.shape[:2]
            if w >= 8: measurements.append(w)
            if h >= 8: measurements.append(h)
            
    if not measurements: return 16
    best_grid = 16
    lowest_error = float('inf')
    
    for cand in GRID_CANDIDATES:
        total_error = 0
        for val in measurements:
            remainder = val % cand
            error = min(remainder, cand - remainder)
            total_error += error
        avg_error = total_error / len(measurements)
        if avg_error < 2.0:
            best_grid = max(best_grid, cand)
            lowest_error = avg_error
        elif avg_error < lowest_error and lowest_error > 2.0:
            best_grid = cand
            lowest_error = avg_error
            
    print(f"Selected Grid Size: {best_grid}x{best_grid}\n")
    return best_grid

def process_dataset():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    all_segment_files = load_images_from_folder(INPUT_SEGMENTS)
    original_levels = load_images_from_folder(INPUT_ORIGINAL)
    
    if not all_segment_files:
        print("Error: Missing segment input files.")
        return

    GRID_SIZE = calculate_dynamic_grid(all_segment_files)
    min_required_pixels = int((GRID_SIZE * GRID_SIZE) * MIN_VISIBLE_PERCENT)
    
    tile_buckets = defaultdict(list)
    global_unique_tiles = [] 
    saved_count = 0
    duplicate_count_p1 = 0
    duplicate_count_p2 = 0

    print("==================================================")
    print("PHASE 1: Foundational Tileset (Top-Left Extraction)")
    print("==================================================")
    
    for idx, file_path in enumerate(all_segment_files):
        img_raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None: continue

        img = clean_image(img_raw)
        h, w = img.shape[:2]
        candidates = []

        # Strictly cut from top-left as in test.py
        for y in range(0, h, GRID_SIZE):
            for x in range(0, w, GRID_SIZE):
                chunk = img[y:y+GRID_SIZE, x:x+GRID_SIZE]
                candidates.append(pad_to_target_transparent(chunk, GRID_SIZE))

        for cand in candidates:
            if np.sum(cand[:, :, 3]) < (MIN_VISIBLE_PIXELS * 255):
                continue

            cand_fuzzy = preprocess_for_matching(cand)
            cand_mean = get_mean_brightness(cand_fuzzy)
            
            potential_duplicates = []
            start = max(0, cand_mean - BUCKET_SEARCH_RANGE)
            end = min(256, cand_mean + BUCKET_SEARCH_RANGE + 1)
            for b_key in range(start, end):
                if b_key in tile_buckets:
                    potential_duplicates.extend(tile_buckets[b_key])

            is_unique = True
            if potential_duplicates:
                cand_bgr = cand_fuzzy[:, :, :3]
                cand_mask = cand_fuzzy[:, :, 3]
                for (exist_orig, exist_fuzzy) in potential_duplicates:
                    padded_existing = cv2.copyMakeBorder(
                        exist_fuzzy[:, :, :3], 1, 1, 1, 1, 
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    try:
                        res = cv2.matchTemplate(padded_existing, cand_bgr, cv2.TM_SQDIFF, mask=cand_mask)
                        min_val, _, _, _ = cv2.minMaxLoc(res)
                        if (min_val / 256.0) < SIMILARITY_THRESHOLD:
                            is_unique = False
                            duplicate_count_p1 += 1
                            break
                    except:
                        continue
            
            if is_unique:
                tile_buckets[cand_mean].append((cand, cand_fuzzy))
                global_unique_tiles.append(cand) # Feed into Phase 2
                saved_count += 1
                filename = f"tile_{saved_count:05d}.png"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), cand)

    print(f"Phase 1 Complete. Base tiles saved: {saved_count}")

    print("\n==================================================")
    print("PHASE 2: Omni-Directional Sweep for Missed Blocks")
    print("==================================================")

    for level_path in original_levels:
        level_stem = level_path.stem
        segment_dir = Path(INPUT_SEGMENTS) / level_stem
        if not segment_dir.exists(): continue
            
        print(f"Scanning level background: {level_stem}")
        level_img_bgr = cv2.imread(str(level_path), cv2.IMREAD_COLOR)
        level_h, level_w = level_img_bgr.shape[:2]
        visited_mask = np.zeros((level_h, level_w), dtype=bool)

        # Block out areas already covered by Phase 1 tiles
        for existing_bgra in global_unique_tiles:
            existing_bgr = existing_bgra[:, :, :3]
            existing_mask = existing_bgra[:, :, 3]
            try:
                res = cv2.matchTemplate(level_img_bgr, existing_bgr, cv2.TM_SQDIFF_NORMED, mask=existing_mask)
                locs = np.where(res <= MATCH_THRESHOLD)
                for y, x in zip(*locs):
                    visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
            except cv2.error:
                continue

        segment_files = load_images_from_folder(segment_dir)
        for seg_path in segment_files:
            img_raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if img_raw is None: continue
                
            wrapped_img = shrink_wrap_alpha(clean_image(img_raw))
            if wrapped_img is None: continue
                
            template_bgr = cv2.cvtColor(wrapped_img, cv2.COLOR_BGRA2BGR)
            template_mask = wrapped_img[:, :, 3]
            
            try:
                res = cv2.matchTemplate(level_img_bgr, template_bgr, cv2.TM_SQDIFF_NORMED, mask=template_mask)
                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            except cv2.error:
                continue
                
            if min_val > MATCH_THRESHOLD: continue
                
            x_min, y_min = min_loc
            h, w = template_bgr.shape[:2]
            
            pad_bottom = max(0, GRID_SIZE - h)
            pad_right = max(0, GRID_SIZE - w)
            safe_mask = cv2.copyMakeBorder(template_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
            
            x_max = x_min + max(w, GRID_SIZE)
            y_max = y_min + max(h, GRID_SIZE)
            candidates = set()

            if w <= GRID_SIZE + 2 and h <= GRID_SIZE + 2:
                candidates.add((x_min, y_min))
            else:
                for y in range(y_min, y_max - GRID_SIZE + 1, GRID_SIZE):
                    for x in range(x_min, x_max - GRID_SIZE + 1, GRID_SIZE):
                        candidates.add((x, y))
                for y in range(y_max - GRID_SIZE, y_min - 1, -GRID_SIZE):
                    for x in range(x_max - GRID_SIZE, x_min - 1, -GRID_SIZE):
                        candidates.add((max(x_min, x), max(y_min, y)))

            scored_candidates = []
            for cx, cy in candidates:
                if cx < 0 or cy < 0 or cx + GRID_SIZE > level_w or cy + GRID_SIZE > level_h:
                    continue
                    
                local_mask = safe_mask[cy - y_min : cy - y_min + GRID_SIZE, cx - x_min : cx - x_min + GRID_SIZE]
                visible_pixels = cv2.countNonZero(local_mask)
                
                if visible_pixels < min_required_pixels:
                    continue
                    
                scored_candidates.append({
                    'x': cx,
                    'y': cy,
                    'area': visible_pixels,
                    'mask': local_mask
                })
                
            scored_candidates.sort(key=lambda item: item['area'], reverse=True)

            for cand in scored_candidates:
                cx, cy = cand['x'], cand['y']
                local_mask = cand['mask']
                
                local_visited = visited_mask[cy:cy+GRID_SIZE, cx:cx+GRID_SIZE]
                if np.count_nonzero(local_visited) >= (GRID_SIZE * GRID_SIZE * 0.85):
                    duplicate_count_p2 += 1
                    continue

                pristine_bgr = level_img_bgr[cy:cy+GRID_SIZE, cx:cx+GRID_SIZE]
                pristine_bgra = cv2.cvtColor(pristine_bgr, cv2.COLOR_BGR2BGRA)
                pristine_bgra[:, :, 3] = local_mask
                
                is_unique = True
                for existing_bgra in global_unique_tiles:
                    existing_bgr = existing_bgra[:, :, :3]
                    existing_mask = existing_bgra[:, :, 3]
                    combined_mask = cv2.bitwise_and(local_mask, existing_mask)
                    
                    if cv2.countNonZero(combined_mask) < min_required_pixels:
                        continue
                        
                    try:
                        res = cv2.matchTemplate(pristine_bgr, existing_bgr, cv2.TM_SQDIFF_NORMED, mask=combined_mask)
                        if res[0][0] <= MATCH_THRESHOLD:
                            is_unique = False
                            break
                    except cv2.error:
                        continue
                            
                if is_unique:
                    global_unique_tiles.append(pristine_bgra)
                    saved_count += 1
                    filename = f"tile_{saved_count:05d}.png"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), pristine_bgra)
                    
                    try:
                        res = cv2.matchTemplate(level_img_bgr, pristine_bgr, cv2.TM_SQDIFF_NORMED, mask=local_mask)
                        locs = np.where(res <= MATCH_THRESHOLD)
                        for y, x in zip(*locs):
                            visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
                    except cv2.error:
                        visited_mask[cy:cy+GRID_SIZE, cx:cx+GRID_SIZE] = True

    print(f"\n--- DONE ---")
    print(f"Total Unique Tiles Extracted: {saved_count}")
    print(f"P1 Duplicates Blocked (Fuzzy Check): {duplicate_count_p1}")
    print(f"P2 Duplicates Blocked (Omni Check): {duplicate_count_p2}")

if __name__ == "__main__":
    process_dataset()