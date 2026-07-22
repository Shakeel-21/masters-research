import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---
INPUT_SEGMENTS = 'output_segments/transRemake'
INPUT_ORIGINAL = 'demo/imgs/test'
OUTPUT_FOLDER = 'output_segments/test4'

GRID_CANDIDATES = [8, 16, 24, 32]
MATCH_THRESHOLD = 0.05
MIN_VISIBLE_PERCENT = 0.30 
COVERAGE_REJECTION_THRESHOLD = 0.55 # Reject tile if 55% of it is already claimed

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
    return img

def get_fuzzy_bgr(bgr_img):
    """Applies blur and quantizes colors to ignore compression artifacts during duplicate checks."""
    blurred = cv2.GaussianBlur(bgr_img, (3, 3), 0)
    return (blurred // 40) * 40 + 20

def standardize_bgra(img_bgra):
    """Zeroes out RGB channels where Alpha is 0 for downstream WFC compatibility."""
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)
    res = img_bgra.copy()
    alpha_mask = res[:, :, 3] == 0
    res[alpha_mask] = [0, 0, 0, 0]
    return res

def pad_to_grid_transparent(img, grid_size):
    h, w = img.shape[:2]
    if h == grid_size and w == grid_size:
        return img.copy()
    canvas = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    copy_h = min(h, grid_size)
    copy_w = min(w, grid_size)
    canvas[0:copy_h, 0:copy_w] = img[0:copy_h, 0:copy_w]
    return canvas

def calculate_dynamic_grid(segment_paths):
    print("\n--- Calculating Dynamic Grid Size ---")
    measurements = []
    for path in segment_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        h, w = img.shape[:2]
        if w >= 8: measurements.append(w)
        if h >= 8: measurements.append(h)
            
    if not measurements: return 16

    best_grid = 16
    lowest_error = float('inf')
    
    for cand in GRID_CANDIDATES:
        total_error = sum(min(val % cand, cand - (val % cand)) for val in measurements)
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
    
    original_levels = load_images_from_folder(INPUT_ORIGINAL)
    all_segment_files = load_images_from_folder(INPUT_SEGMENTS)
    
    if not original_levels or not all_segment_files:
        print("Error: Missing input folders.")
        return

    GRID_SIZE = calculate_dynamic_grid(all_segment_files)
    min_required_pixels = int((GRID_SIZE * GRID_SIZE) * MIN_VISIBLE_PERCENT)
    
    global_unique_tiles = [] 
    saved_count = 0
    duplicate_count = 0

    for level_path in original_levels:
        level_stem = level_path.stem
        segment_dir = Path(INPUT_SEGMENTS) / level_stem
        if not segment_dir.exists(): continue
            
        print(f"\nProcessing level: {level_stem}")
        
        level_img_raw = cv2.imread(str(level_path), cv2.IMREAD_UNCHANGED)
        level_img_bgra = clean_image(level_img_raw)
        level_img_bgr = level_img_bgra[:, :, :3]
        
        # Used strictly for evaluation to prevent artifacts from breaking duplicate detection
        fuzzy_level_bgr = get_fuzzy_bgr(level_img_bgr)
        level_h, level_w = level_img_bgra.shape[:2]
        visited_mask = np.zeros((level_h, level_w), dtype=bool)

        # ---------------------------------------------------------
        # PHASE 0: DYNAMIC GRID ANCHORING
        # ---------------------------------------------------------
        segment_files = load_images_from_folder(segment_dir)
        segment_files_sorted = sorted(segment_files, key=lambda p: os.path.getsize(p), reverse=True)
        
        x_offsets, y_offsets = [], []
        
        # Expand sample size and filter exclusively for structural geometry
        for seg_path in segment_files_sorted[:40]:
            img_raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if img_raw is None: continue
            
            seg_bgra = clean_image(img_raw)
            seg_mask = seg_bgra[:, :, 3]
            
            file_h, file_w = seg_bgra.shape[:2]
            area = file_h * file_w
            
            # Enforce solidity: Only allow segments that are >75% solid to vote.
            # This explicitly blocks floating transparent artifacts and irregular shapes.
            if area == 0 or (cv2.countNonZero(seg_mask) / area) < 0.75:
                continue

            fuzzy_seg = get_fuzzy_bgr(seg_bgra[:, :, :3])
            
            try:
                res = cv2.matchTemplate(fuzzy_level_bgr, fuzzy_seg, cv2.TM_SQDIFF_NORMED, mask=seg_mask)
                
                # Harvest ALL locations below the threshold, not just the single best match
                locs = np.where(res <= MATCH_THRESHOLD)
                
                for y, x in zip(*locs):
                    x_offsets.append(x % GRID_SIZE)
                    y_offsets.append(y % GRID_SIZE)
            except cv2.error:
                continue

        offset_x = Counter(x_offsets).most_common(1)[0][0] if x_offsets else 0
        offset_y = Counter(y_offsets).most_common(1)[0][0] if y_offsets else 0
        print(f"  -> Anchored grid phase to ({offset_x}, {offset_y}) using {len(x_offsets)} geometric data points")

        visited_mask = np.zeros((level_h, level_w), dtype=bool)

        # ---------------------------------------------------------
        # PHASE 1: GLOBAL DICTIONARY EAT-UP (FUZZY RESTORED)
        # ---------------------------------------------------------
        for existing_bgra in global_unique_tiles:
            existing_bgr = existing_bgra[:, :, :3]
            existing_mask = existing_bgra[:, :, 3]
            fuzzy_existing_bgr = get_fuzzy_bgr(existing_bgr)
            
            try:
                res = cv2.matchTemplate(fuzzy_level_bgr, fuzzy_existing_bgr, cv2.TM_SQDIFF_NORMED, mask=existing_mask)
                locs = np.where(res <= MATCH_THRESHOLD)
                for y, x in zip(*locs):
                    # Flag this area as already claimed to block misaligned overlapping extracts later
                    visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
            except cv2.error:
                continue

        # ---------------------------------------------------------
        # PHASE 2: ALIGNMENT PIPELINE & EXTRACTION
        # ---------------------------------------------------------
        for seg_path in segment_files:
            img_raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if img_raw is None: continue
                
            segment_bgra = clean_image(img_raw)
            file_h, file_w = segment_bgra.shape[:2]
            template_bgr = segment_bgra[:, :, :3]
            
            # Clean dirty segment alpha channel
            _, template_mask = cv2.threshold(segment_bgra[:, :, 3], 127, 255, cv2.THRESH_BINARY)
            
            if cv2.countNonZero(template_mask) < 25: 
                continue
            
            found_in_level = False
            file_x, file_y = 0, 0
            fuzzy_template_bgr = get_fuzzy_bgr(template_bgr)
            
            try:
                res = cv2.matchTemplate(fuzzy_level_bgr, fuzzy_template_bgr, cv2.TM_SQDIFF_NORMED, mask=template_mask)
                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                if min_val <= MATCH_THRESHOLD:
                    found_in_level = True
                    file_x, file_y = min_loc
            except cv2.error:
                pass
                
            scored_candidates = []
            
            if found_in_level:
                start_x = ((file_x - offset_x) // GRID_SIZE) * GRID_SIZE + offset_x
                start_y = ((file_y - offset_y) // GRID_SIZE) * GRID_SIZE + offset_y
                end_x = ((file_x + file_w - offset_x + GRID_SIZE - 1) // GRID_SIZE) * GRID_SIZE + offset_x
                end_y = ((file_y + file_h - offset_y + GRID_SIZE - 1) // GRID_SIZE) * GRID_SIZE + offset_y
                
                for gy in range(start_y, end_y, GRID_SIZE):
                    for gx in range(start_x, end_x, GRID_SIZE):
                        if gx < 0 or gy < 0 or gx + GRID_SIZE > level_w or gy + GRID_SIZE > level_h:
                            continue

                        overlap_x_start = max(gx, file_x)
                        overlap_y_start = max(gy, file_y)
                        overlap_x_end = min(gx + GRID_SIZE, file_x + file_w)
                        overlap_y_end = min(gy + GRID_SIZE, file_y + file_h)
                        
                        if overlap_x_start >= overlap_x_end or overlap_y_start >= overlap_y_end:
                            continue 
                            
                        local_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                        
                        lg_x1, lg_y1 = overlap_x_start - gx, overlap_y_start - gy
                        lg_x2, lg_y2 = lg_x1 + (overlap_x_end - overlap_x_start), lg_y1 + (overlap_y_end - overlap_y_start)
                        sm_x1, sm_y1 = overlap_x_start - file_x, overlap_y_start - file_y
                        sm_x2, sm_y2 = sm_x1 + (overlap_x_end - overlap_x_start), sm_y1 + (overlap_y_end - overlap_y_start)
                        
                        local_mask[lg_y1:lg_y2, lg_x1:lg_x2] = template_mask[sm_y1:sm_y2, sm_x1:sm_x2]
                        local_level_alpha = level_img_bgra[gy:gy+GRID_SIZE, gx:gx+GRID_SIZE, 3]
                        
                        combined_mask = cv2.bitwise_and(local_level_alpha, local_mask)
                        visible_pixels = cv2.countNonZero(combined_mask)
                        
                        if visible_pixels >= (GRID_SIZE * GRID_SIZE * 0.85):
                            combined_mask = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
                            visible_pixels = GRID_SIZE * GRID_SIZE
                        
                        if visible_pixels < min_required_pixels:
                            continue
                            
                        scored_candidates.append({
                            'abs_x': gx, 'abs_y': gy,
                            'area': visible_pixels,
                            'mask': combined_mask,
                            'is_fallback': False
                        })
            else:
                for sy in range(0, file_h, GRID_SIZE):
                    for sx in range(0, file_w, GRID_SIZE):
                        local_mask = template_mask[sy:sy+GRID_SIZE, sx:sx+GRID_SIZE]
                        visible_pixels = cv2.countNonZero(local_mask)
                        
                        if visible_pixels >= (GRID_SIZE * GRID_SIZE * 0.85):
                            local_mask = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
                            visible_pixels = GRID_SIZE * GRID_SIZE
                        
                        if visible_pixels < min_required_pixels:
                            continue
                            
                        scored_candidates.append({
                            'sx': sx, 'sy': sy,
                            'area': visible_pixels,
                            'mask': local_mask,
                            'is_fallback': True
                        })

            scored_candidates.sort(key=lambda item: item['area'], reverse=True)

            for cand in scored_candidates:
                if not cand['is_fallback']:
                    abs_x, abs_y = cand['abs_x'], cand['abs_y']
                    final_local_mask = cand['mask']
                    
                    # 55% Coverage Check: Rejects misaligned chunks overlapping found geometry
                    local_visited = visited_mask[abs_y:abs_y+GRID_SIZE, abs_x:abs_x+GRID_SIZE]
                    if np.count_nonzero(local_visited) >= (GRID_SIZE * GRID_SIZE * COVERAGE_REJECTION_THRESHOLD):
                        duplicate_count += 1
                        continue

                    # STRICT EXTRACTION: Pulls BGR strictly from pristine level image, ignores dirty segment colors
                    level_chunk_bgr = level_img_bgr[abs_y:abs_y+GRID_SIZE, abs_x:abs_x+GRID_SIZE]
                    pristine_bgra = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
                    pristine_bgra[:, :, :3] = level_chunk_bgr
                    pristine_bgra[:, :, 3] = final_local_mask
                    
                else:
                    sx, sy = cand['sx'], cand['sy']
                    final_local_mask = cand['mask']
                    
                    chunk_bgra = segment_bgra[sy:sy+GRID_SIZE, sx:sx+GRID_SIZE].copy()
                    pristine_bgra = pad_to_grid_transparent(chunk_bgra, GRID_SIZE)
                    pristine_bgra[:, :, 3] = pad_to_grid_transparent(cv2.cvtColor(final_local_mask, cv2.COLOR_GRAY2BGRA), GRID_SIZE)[:,:,0]

                # Deduplication sweep uses fuzzy math to ignore artifacts
                pristine_bgr = pristine_bgra[:, :, :3]
                local_alpha = pristine_bgra[:, :, 3]
                fuzzy_pristine_bgr = get_fuzzy_bgr(pristine_bgr)

                is_unique = True
                for existing_bgra in global_unique_tiles:
                    existing_bgr = existing_bgra[:, :, :3]
                    existing_mask = existing_bgra[:, :, 3]
                    fuzzy_existing_bgr = get_fuzzy_bgr(existing_bgr)
                    
                    combined_check_mask = cv2.bitwise_and(local_alpha, existing_mask)
                    if cv2.countNonZero(combined_check_mask) < min_required_pixels:
                        continue
                        
                    try:
                        res = cv2.matchTemplate(fuzzy_pristine_bgr, fuzzy_existing_bgr, cv2.TM_SQDIFF_NORMED, mask=combined_check_mask)
                        if res[0][0] <= MATCH_THRESHOLD:
                            is_unique = False
                            break
                    except cv2.error:
                        continue
                            
                if is_unique:
                    # Output conversion happens HERE, ensuring solver gets standardized tiles
                    final_output = standardize_bgra(pristine_bgra)
                    global_unique_tiles.append(final_output)
                    
                    saved_count += 1
                    filename = f"tile_{saved_count:05d}.png"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), final_output)
                    
                    try:
                        res = cv2.matchTemplate(fuzzy_level_bgr, fuzzy_pristine_bgr, cv2.TM_SQDIFF_NORMED, mask=local_alpha)
                        locs = np.where(res <= MATCH_THRESHOLD)
                        for y, x in zip(*locs):
                            visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
                    except cv2.error:
                        if not cand['is_fallback']:
                            visited_mask[abs_y:abs_y+GRID_SIZE, abs_x:abs_x+GRID_SIZE] = True

    print(f"\n--- DONE ---")
    print(f"Pristine Silhouette Tiles Extracted: {saved_count}")
    print(f"Duplicates Blocked: {duplicate_count}")

if __name__ == "__main__":
    process_dataset()