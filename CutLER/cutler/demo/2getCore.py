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
MIN_VISIBLE_PERCENT = 0.55  

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
        
        level_h, level_w = level_img_bgra.shape[:2]
        visited_mask = np.zeros((level_h, level_w), dtype=bool)

        # ---------------------------------------------------------
        # PHASE 1: GLOBAL DICTIONARY EAT-UP
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # PHASE 2: CLAMPED OMNI-DIRECTIONAL EXTRACTION
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
            
            try:
                res = cv2.matchTemplate(level_img_bgr, template_bgr, cv2.TM_SQDIFF_NORMED, mask=template_mask)
                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            except cv2.error:
                continue
                
            if min_val > MATCH_THRESHOLD:
                continue
                
            file_x, file_y = min_loc
            
            candidates = []
            seen = set()
            
            # Helper to strictly clamp extraction bounds within the segment's actual size
            def add_bounds(cx, cy):
                ex = max(0, cx)
                ey = max(0, cy)
                ew = min(cx + GRID_SIZE, file_w) - ex
                eh = min(cy + GRID_SIZE, file_h) - ey
                
                if ew > 0 and eh > 0:
                    bounds = (ex, ey, ew, eh)
                    if bounds not in seen:
                        seen.add(bounds)
                        candidates.append(bounds)

            # Define sweep steps
            x_fwd = list(range(0, file_w, GRID_SIZE))
            y_fwd = list(range(0, file_h, GRID_SIZE))
            
            x_rev = []
            cx = file_w - GRID_SIZE
            while cx >= -GRID_SIZE + 1:
                x_rev.append(cx)
                cx -= GRID_SIZE
                
            y_rev = []
            cy = file_h - GRID_SIZE
            while cy >= -GRID_SIZE + 1:
                y_rev.append(cy)
                cy -= GRID_SIZE

            # 1. Bottom-Left Anchor
            for x in x_fwd:
                for y in y_rev: add_bounds(x, y)
            
            # 2. Top-Right Anchor
            for x in x_rev:
                for y in y_fwd: add_bounds(x, y)
                
            # 3. Bottom-Right Anchor
            for x in x_rev:
                for y in y_rev: add_bounds(x, y)
                
            # 4. Top-Left Anchor
            for x in x_fwd:
                for y in y_fwd: add_bounds(x, y)

            if not candidates:
                add_bounds(0, 0)

            for (bx, by, bw, bh) in candidates:
                abs_x = file_x + bx
                abs_y = file_y + by
                
                if abs_x < 0 or abs_y < 0 or abs_x + bw > level_w or abs_y + bh > level_h:
                    continue
                    
                # Extract exact dimensions, do not pull from outside logical boundaries
                chunk_bgra = level_img_bgra[abs_y : abs_y + bh, abs_x : abs_x + bw].copy()
                
                # Use standard blank transparent canvas to fill remaining space
                pristine_bgra = pad_to_grid_transparent(chunk_bgra, GRID_SIZE)
                pristine_bgr = pristine_bgra[:, :, :3]
                local_level_alpha = pristine_bgra[:, :, 3]
                
                visible_pixels = cv2.countNonZero(local_level_alpha)
                if visible_pixels < min_required_pixels:
                    continue
                    
                v_y_end = min(abs_y + GRID_SIZE, level_h)
                v_x_end = min(abs_x + GRID_SIZE, level_w)
                local_visited = visited_mask[abs_y:v_y_end, abs_x:v_x_end]
                
                if np.count_nonzero(local_visited) >= (GRID_SIZE * GRID_SIZE * 0.5):
                    duplicate_count += 1
                    continue

                is_unique = True
                for existing_bgra in global_unique_tiles:
                    existing_bgr = existing_bgra[:, :, :3]
                    existing_mask = existing_bgra[:, :, 3]
                    
                    combined_mask = cv2.bitwise_and(local_level_alpha, existing_mask)
                    
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
                        res = cv2.matchTemplate(level_img_bgr, pristine_bgr, cv2.TM_SQDIFF_NORMED, mask=local_level_alpha)
                        locs = np.where(res <= MATCH_THRESHOLD)
                        for y, x in zip(*locs):
                            visited_mask[y:y+GRID_SIZE, x:x+GRID_SIZE] = True
                    except cv2.error:
                        visited_mask[abs_y:v_y_end, abs_x:v_x_end] = True

    print(f"\n--- DONE ---")
    print(f"Pristine Silhouette Tiles Extracted: {saved_count}")
    print(f"Duplicates Blocked: {duplicate_count}")

if __name__ == "__main__":
    process_dataset()