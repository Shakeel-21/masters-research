import os
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
INPUT_SEGMENTS = 'output_segments/transRemake'
INPUT_ORIGINAL = 'demo/imgs/test'
OUTPUT_FOLDER = 'output_segments/coreV2'

GRID_CANDIDATES = [8, 16, 24, 32]
MATCH_THRESHOLD = 0.05
MIN_VISIBLE_PERCENT = 0.30 

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

def pad_to_grid_transparent(img, grid_size):
    h, w = img.shape[:2]
    if h == grid_size and w == grid_size:
        return img.copy()
    canvas = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    copy_h = min(h, grid_size)
    copy_w = min(w, grid_size)
    canvas[0:copy_h, 0:copy_w] = img[0:copy_h, 0:copy_w]
    return canvas

def get_fuzzy_bgr(bgr_img):
    """Applies blur and quantizes colors to ignore compression artifacts during duplicate checks."""
    blurred = cv2.GaussianBlur(bgr_img, (3, 3), 0)
    # Block colors into groups of 40 to easily match slight hue/compression shifts
    return (blurred // 40) * 40 + 20

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
        
        # Use fuzzy level image strictly for evaluation, never for saving
        fuzzy_level_bgr = get_fuzzy_bgr(level_img_bgr)
        
        level_h, level_w = level_img_bgra.shape[:2]
        visited_mask = np.zeros((level_h, level_w), dtype=bool)

        # ---------------------------------------------------------
        # PHASE 1: GLOBAL DICTIONARY EAT-UP
        # ---------------------------------------------------------
        for existing_bgra in global_unique_tiles:
            existing_bgr = existing_bgra[:, :, :3]
            existing_mask = existing_bgra[:, :, 3]
            fuzzy_existing_bgr = get_fuzzy_bgr(existing_bgr)
            
            try:
                res = cv2.matchTemplate(fuzzy_level_bgr, fuzzy_existing_bgr, cv2.TM_SQDIFF_NORMED, mask=existing_mask)
                locs = np.where(res <= MATCH_THRESHOLD)
                for y, x in zip(*locs):
                    visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
            except cv2.error:
                continue

        # ---------------------------------------------------------
        # PHASE 2: ALIGNMENT PIPELINE
        # ---------------------------------------------------------
        segment_files = load_images_from_folder(segment_dir)
        
        for seg_path in segment_files:
            img_raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if img_raw is None: continue
                
            segment_bgra = clean_image(img_raw)
            file_h, file_w = segment_bgra.shape[:2]
            template_bgr = segment_bgra[:, :, :3]
            template_mask = segment_bgra[:, :, 3]
            
            if cv2.countNonZero(template_mask) < 25: 
                continue
            
            # Map to Level
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
                # --- ALIGNMENT MODE 1: ABSOLUTE GLOBAL GRID ---
                start_x = (file_x // GRID_SIZE) * GRID_SIZE
                start_y = (file_y // GRID_SIZE) * GRID_SIZE
                end_x = ((file_x + file_w + GRID_SIZE - 1) // GRID_SIZE) * GRID_SIZE
                end_y = ((file_y + file_h + GRID_SIZE - 1) // GRID_SIZE) * GRID_SIZE
                
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
                        
                        lg_x1 = overlap_x_start - gx
                        lg_y1 = overlap_y_start - gy
                        lg_x2 = lg_x1 + (overlap_x_end - overlap_x_start)
                        lg_y2 = lg_y1 + (overlap_y_end - overlap_y_start)
                        
                        sm_x1 = overlap_x_start - file_x
                        sm_y1 = overlap_y_start - file_y
                        sm_x2 = sm_x1 + (overlap_x_end - overlap_x_start)
                        sm_y2 = sm_y1 + (overlap_y_end - overlap_y_start)
                        
                        local_mask[lg_y1:lg_y2, lg_x1:lg_x2] = template_mask[sm_y1:sm_y2, sm_x1:sm_x2]
                        local_level_alpha = level_img_bgra[gy:gy+GRID_SIZE, gx:gx+GRID_SIZE, 3]
                        
                        combined_mask = cv2.bitwise_and(local_level_alpha, local_mask)
                        visible_pixels = cv2.countNonZero(combined_mask)
                        
                        # MASK HEALING: If tile is mostly solid (like a pipe), force it to a perfect 100% square
                        # This stops edge shards from breaking the duplicate check.
                        if visible_pixels >= (GRID_SIZE * GRID_SIZE * 0.85):
                            combined_mask = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
                            visible_pixels = GRID_SIZE * GRID_SIZE
                        
                        if visible_pixels < min_required_pixels:
                            continue
                            
                        scored_candidates.append({
                            'abs_x': gx,
                            'abs_y': gy,
                            'area': visible_pixels,
                            'mask': combined_mask,
                            'is_fallback': False
                        })
            else:
                # --- ALIGNMENT MODE 2: STRICT LOCAL GRID (NO OVERLAP SWEEPS) ---
                for sy in range(0, file_h, GRID_SIZE):
                    for sx in range(0, file_w, GRID_SIZE):
                        local_mask = template_mask[sy:sy+GRID_SIZE, sx:sx+GRID_SIZE]
                        visible_pixels = cv2.countNonZero(local_mask)
                        
                        # MASK HEALING
                        if visible_pixels >= (GRID_SIZE * GRID_SIZE * 0.85):
                            local_mask = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)
                            visible_pixels = GRID_SIZE * GRID_SIZE
                        
                        if visible_pixels < min_required_pixels:
                            continue
                            
                        scored_candidates.append({
                            'sx': sx,
                            'sy': sy,
                            'area': visible_pixels,
                            'mask': local_mask,
                            'is_fallback': True
                        })

            scored_candidates.sort(key=lambda item: item['area'], reverse=True)

            for cand in scored_candidates:
                if not cand['is_fallback']:
                    abs_x, abs_y = cand['abs_x'], cand['abs_y']
                    final_local_mask = cand['mask']
                    
                    local_visited = visited_mask[abs_y:abs_y+GRID_SIZE, abs_x:abs_x+GRID_SIZE]
                    if np.count_nonzero(local_visited) >= (GRID_SIZE * GRID_SIZE * 0.50):
                        duplicate_count += 1
                        continue

                    level_chunk_bgr = level_img_bgra[abs_y:abs_y+GRID_SIZE, abs_x:abs_x+GRID_SIZE, :3]
                    
                    pristine_bgra = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
                    pristine_bgra[:, :, :3] = level_chunk_bgr
                    pristine_bgra[:, :, 3] = final_local_mask
                    
                else:
                    sx, sy = cand['sx'], cand['sy']
                    final_local_mask = cand['mask']
                    
                    chunk_bgra = segment_bgra[sy:sy+GRID_SIZE, sx:sx+GRID_SIZE].copy()
                    
                    # Apply healed mask to fallback chunk if necessary
                    pristine_bgra = pad_to_grid_transparent(chunk_bgra, GRID_SIZE)
                    pristine_bgra[:, :, 3] = pad_to_grid_transparent(cv2.cvtColor(final_local_mask, cv2.COLOR_GRAY2BGRA), GRID_SIZE)[:,:,0]

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
                    global_unique_tiles.append(pristine_bgra)
                    saved_count += 1
                    filename = f"tile_{saved_count:05d}.png"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), pristine_bgra)
                    
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