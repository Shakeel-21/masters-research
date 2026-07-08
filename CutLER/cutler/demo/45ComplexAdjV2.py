import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
import shutil
import re
from collections import defaultdict, Counter

class LevelReconstructor:
    """
    Reconstructs level images using a Global Core Dictionary combined with Level-Specific Segments.
    Learns adjacency rules and saves the surviving level-specific tiles to an output folder.
    """
    
    def __init__(self, global_core_path, segment_path, level_path, output_path, match_threshold=0.04):
        self.global_core_path = Path(global_core_path)
        self.segments_base_path = Path(segment_path)
        self.level_path = Path(level_path)
        self.output_path = Path(output_path)
        self.MATCH_THRESHOLD = match_threshold
        

        self.base_core_cache = {} # Holds the global dictionary of pristine tiles
        self.template_cache = {}  # Holds global core + level-specific complex segments
        self.source_image_cache = {}
        
        self.unmatched_templates = set()
        self.used_templates = set() 
        self.dynamic_templates = {}

        self.adjacency_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self.tile_frequencies = defaultdict(float)
        self.cell_w = 0
        self.cell_h = 0

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _process_template_image(self, img_path):
        """Helper to standardize template processing for both Core and Complex tiles."""
        img_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_bgra is None: return None
        
        if img_bgra.shape[2] == 4:
            mask = img_bgra[:, :, 3] 
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = img_bgra
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _ , mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

        is_valid = cv2.countNonZero(mask) > 0
        h, w = img_bgr.shape[:2]

        erosion_kernel = np.ones((5, 5), np.uint8)
        match_mask = cv2.erode(mask, erosion_kernel, iterations=1)
        if cv2.countNonZero(match_mask) == 0:
            match_mask = mask

        OVERLAP_TOLERANCE = 4 
        kernel_size = OVERLAP_TOLERANCE * 2 + 1
        shrink_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        c_alpha = img_bgra[:, :, 3] > 127
        c_shrunk = cv2.erode((c_alpha.astype(np.uint8)*255), shrink_kernel, iterations=1) > 127
        if not np.any(c_shrunk): 
            c_shrunk = c_alpha
            
        # OPTIMIZATION: Unmasked matching is 50x faster for solid blocks
        is_solid = (cv2.countNonZero(match_mask) == (h * w))

        return {
            'img_bgr': img_bgr,
            'img_bgra': img_bgra,
            'mask': mask,
            'match_mask': None if is_solid else match_mask, 
            'w': w,
            'h': h,
            'is_valid': is_valid,
            'area': w * h,
            'filepath': img_path,
            'c_alpha': c_alpha,
            'c_shrunk': c_shrunk
        }

    def _load_source_levels(self):
        pattern = '*.png'
        image_files = list(self.level_path.glob(pattern))
        if not image_files:
            print(f"Warning: No PNG files found in {self.level_path}", file=sys.stderr)
            return
        print(f"\nFound {len(image_files)} source levels in {self.level_path}...")
        for img_path in tqdm(image_files, desc="Loading Source Levels"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED) 
            if img is not None:
                self.source_image_cache[img_path.name] = img

    def _load_global_core_templates(self):
        """Loads the massive dictionary of core tiles once to save memory and time."""
        image_files = list(self.global_core_path.glob('*.png'))
        if not image_files:
            print(f"Warning: No core tiles found in {self.global_core_path}", file=sys.stderr)
            return
            
        print(f"Loading {len(image_files)} global core tiles from {self.global_core_path}...")
        for img_path in tqdm(image_files, desc="Caching Global Core Dictionary"):
            data = self._process_template_image(img_path)
            if data:
                self.base_core_cache[img_path.name] = data

    def _load_templates_for_level(self, level_stem):
        """Combines the global core dictionary with the level's specific complex segments."""
        templates = {k: v for k, v in self.base_core_cache.items()}
        
        specific_folder = self.segments_base_path / level_stem
        if specific_folder.exists() and specific_folder.is_dir():
            image_files = list(specific_folder.glob('*.png'))
            for img_path in image_files:
                data = self._process_template_image(img_path)
                if data:
                    templates[img_path.name] = data
                    
        return templates

    def _get_nms_matches(self, img, template_data):
        if template_data['match_mask'] is not None:
            res = cv2.matchTemplate(img, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['match_mask'])
        else:
            res = cv2.matchTemplate(img, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED)
            
        locs = np.where(res <= self.MATCH_THRESHOLD)
        
        boxes = []
        scores = []
        raw_matches = []
        
        h, w = template_data['h'], template_data['w']
        
        for y, x in zip(*locs):
            score = res[y, x]
            raw_matches.append((x, y, score))
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(float(1.0 - score)) 
            
        if not boxes:
            return []
            
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.2)
        if len(indices) > 0:
            filtered = [raw_matches[i] for i in indices.flatten()]
            filtered.sort(key=lambda x: x[2]) 
            return filtered
            
        return []

    def _reconstruct_single_level_complex(self, level_name, level_img_original, level_stem, out_base):
        print(f"\n  -> Running Two-Stage Solver for {level_name}")
        h, w, _ = level_img_original.shape
        
        id_grid_A = np.full((h, w), "B", dtype=object)
        visited_mask_A = np.zeros((h, w), dtype=bool)
        placed_instances_A = []
        
        id_grid_B = np.full((h, w), "B", dtype=object)
        visited_mask_B = np.zeros((h, w), dtype=bool)
        placed_instances_B = []

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            if level_img_original.shape[2] == 4:
                # 1. Create a copy to avoid altering the original
                level_img_for_matching = level_img_original[:, :, :3].copy()
                
                # 2. Extract the alpha channel
                alpha_channel = level_img_original[:, :, 3]
                
                # 3. Find all pixels that are fully transparent
                transparent_mask = alpha_channel == 0
                
                # 4. Paint those pixels bright magenta (or another unique color)
                level_img_for_matching[transparent_mask] = [255, 0, 255] # [Blue, Green, Red]
            else:
                level_img_for_matching = level_img_original
            
            source_alpha = level_img_original[:, :, 3]
            is_occupied_source = source_alpha > 10
        else:
            level_img_for_matching = level_img_original
            bg_color = level_img_original[0, 0]
            diff = cv2.absdiff(level_img_original, bg_color)
            diff_sum = np.sum(diff, axis=2)
            is_occupied_source = diff_sum > 10

        core_templates = {k: v for k, v in self.template_cache.items() if k.startswith("tile_")}
        complex_templates = {k: v for k, v in self.template_cache.items() if not k.startswith("tile_")}
        
        core_sorted = sorted(core_templates.items(), key=lambda item: cv2.countNonZero(item[1]['mask']) if item[1]['mask'] is not None else item[1]['area'], reverse=True)
        complex_sorted = sorted(complex_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        if core_sorted:
            self.cell_w = Counter([t['w'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]
            self.cell_h = Counter([t['h'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]

        x_offsets = []
        for template_key, template_data in core_sorted[:10]: 
            if not template_data['is_valid']: continue
            matches = self._get_nms_matches(level_img_for_matching, template_data)
            for x, _, _ in matches:
                x_offsets.append(int(x) % self.cell_w)
                
        global_offset_x = Counter(x_offsets).most_common(1)[0][0] if x_offsets else 0
        print(f"  -> Auto-calibrated Grid Offset: {global_offset_x} pixels")

        # PHASE 1.1: Complex ROIs
        for comp_key, comp_data in complex_sorted:
            if not comp_data['is_valid']: continue
            matches = self._get_nms_matches(level_img_for_matching, comp_data)

            for gx, gy, _ in matches:
                gx, gy = int(gx), int(gy)
                comp_h, comp_w = comp_data['h'], comp_data['w']
                if gy + comp_h > h or gx + comp_w > w: continue

                buffer_up = self.cell_h * 2
                buffer_down = self.cell_h * 2
                buffer_side = self.cell_w * 2
                
                roi_y1 = max(0, gy - buffer_up)
                roi_y2 = min(h, gy + comp_h + buffer_down)
                roi_x1 = max(0, gx - buffer_side)
                roi_x2 = min(w, gx + comp_w + buffer_side)
                
                roi_bgr = level_img_for_matching[roi_y1:roi_y2, roi_x1:roi_x2]

                for core_key, core_data in core_sorted:
                    if not core_data['is_valid']: continue
                    if roi_bgr.shape[0] < core_data['h'] or roi_bgr.shape[1] < core_data['w']: continue

                    c_matches = self._get_nms_matches(roi_bgr, core_data)

                    for lx, ly, _ in c_matches:
                        cw, ch = core_data['w'], core_data['h']
                        
                        global_x = int(roi_x1 + lx)
                        global_y = int(roi_y1 + ly)

                        if global_y + ch > h or global_x + cw > w: continue

                        c_alpha = core_data['c_alpha']
                        

                        if np.any(visited_mask_A[global_y:global_y+ch, global_x:global_x+cw][c_alpha]):
                            continue

                        relative_x = global_x - global_offset_x
                        offset_x = relative_x % self.cell_w
                        
                        is_middle_object = False
                        if self.cell_w > 0:
                            is_middle_object = abs(offset_x - (self.cell_w // 2)) <= 1

                        if is_middle_object:
                            grid_x_left = global_x - offset_x
                            grid_x_right = global_x - offset_x + self.cell_w
                            split_pt = self.cell_w - offset_x
                            
                            bg_color_local = level_img_original[0, 0, :3] if len(level_img_original.shape) == 3 else [0,0,0]

                            left_bgra = np.zeros((ch, self.cell_w, 4), dtype=np.uint8)
                            actual_split_w = min(split_pt, cw)
                            left_bgra[:, offset_x : offset_x + actual_split_w] = core_data['img_bgra'][:, :actual_split_w]
                            
                            l_alpha = left_bgra[:, :, 3] > 127
                            is_l_pure_bg = True
                            if np.any(l_alpha):
                                mean_color = np.mean(left_bgra[l_alpha, :3], axis=0)
                                std_color = np.std(left_bgra[l_alpha, :3], axis=0)
                                if np.linalg.norm(mean_color - bg_color_local) < 15.0 and np.all(std_color < 5.0):
                                    is_l_pure_bg = True
                                else:
                                    is_l_pure_bg = False

                            right_bgra = np.zeros((ch, self.cell_w, 4), dtype=np.uint8)
                            if cw > split_pt:
                                actual_rem_w = min(cw - split_pt, self.cell_w)
                                right_bgra[:, :actual_rem_w] = core_data['img_bgra'][:, split_pt : split_pt + actual_rem_w]
                            
                            r_alpha = right_bgra[:, :, 3] > 127
                            is_r_pure_bg = True
                            if np.any(r_alpha):
                                mean_color = np.mean(right_bgra[r_alpha, :3], axis=0)
                                std_color = np.std(right_bgra[r_alpha, :3], axis=0)
                                if np.linalg.norm(mean_color - bg_color_local) < 15.0 and np.all(std_color < 5.0):
                                    is_r_pure_bg = True
                                else:
                                    is_r_pure_bg = False

                            visited_mask_A[global_y:global_y+ch, global_x:global_x+cw][c_alpha] = True
                            local_y = round((global_y - gy) / self.cell_h) if self.cell_h > 0 else 0

                            if not is_l_pure_bg:
                                local_x_left = round((grid_x_left - gx) / self.cell_w) if self.cell_w > 0 else 0
                                left_key = f"split_L_{core_key}"
                                if left_key not in self.dynamic_templates:
                                    self.dynamic_templates[left_key] = left_bgra
                                    
                                clone_id_L = f"{comp_key[:-4]}_{left_key[:-4]}_y{local_y}_x{local_x_left}"
                                l_start_x = max(0, grid_x_left)
                                l_end_x = min(w, grid_x_left + self.cell_w)
                                l_vis_w = l_end_x - l_start_x
                                
                                if l_vis_w > 0:
                                    l_mask_start = 0 if grid_x_left >= 0 else -grid_x_left
                                    l_mask_end = l_mask_start + l_vis_w
                                    chunk_l_alpha = left_bgra[:, l_mask_start:l_mask_end, 3] > 127
                                    id_grid_A[global_y:global_y+ch, l_start_x:l_end_x][chunk_l_alpha] = clone_id_L
                                    placed_instances_A.append((clone_id_L, grid_x_left, global_y, self.cell_w, ch))

                            if not is_r_pure_bg:
                                local_x_right = round((grid_x_right - gx) / self.cell_w) if self.cell_w > 0 else 0
                                right_key = f"split_R_{core_key}"
                                if right_key not in self.dynamic_templates:
                                    self.dynamic_templates[right_key] = right_bgra
                                    
                                clone_id_R = f"{comp_key[:-4]}_{right_key[:-4]}_y{local_y}_x{local_x_right}"
                                r_start_x = max(0, grid_x_right)
                                r_end_x = min(w, grid_x_right + self.cell_w)
                                r_vis_w = r_end_x - r_start_x
                                
                                if r_vis_w > 0:
                                    r_mask_start = 0 if grid_x_right >= 0 else -grid_x_right
                                    r_mask_end = r_mask_start + r_vis_w
                                    chunk_r_alpha = right_bgra[:, r_mask_start:r_mask_end, 3] > 127
                                    id_grid_A[global_y:global_y+ch, r_start_x:r_end_x][chunk_r_alpha] = clone_id_R
                                    placed_instances_A.append((clone_id_R, grid_x_right, global_y, self.cell_w, ch))
                            
                            self.used_templates.add(core_data['filepath'])
                            self.used_templates.add(comp_data['filepath'])

                        else:
                            grid_y = round((global_y - gy) / self.cell_h) if self.cell_h > 0 else 0
                            grid_x = round((global_x - gx) / self.cell_w) if self.cell_w > 0 else 0
                            
                            clone_id = f"{comp_key[:-4]}_{core_key[:-4]}_y{grid_y}_x{grid_x}"
                            
                            if clone_id not in self.dynamic_templates:
                                self.dynamic_templates[clone_id] = core_data['img_bgra']
                                
                            id_grid_A[global_y:global_y+ch, global_x:global_x+cw][c_alpha] = clone_id
                            visited_mask_A[global_y:global_y+ch, global_x:global_x+cw][c_alpha] = True
                            placed_instances_A.append((clone_id, global_x, global_y, cw, ch))
                            
                            self.used_templates.add(core_data['filepath'])
                            self.used_templates.add(comp_data['filepath'])

        # PHASE 1.2: Generic Fill & Splitting Logic (Global Dictionary Eat-Up)
        all_candidates = []
        for core_key, core_data in core_sorted:
            if not core_data['is_valid']: continue
            matches = self._get_nms_matches(level_img_for_matching, core_data)

            for x, y, score in matches:
                x, y = int(x), int(y)
                cw, ch = core_data['w'], core_data['h']
                
                if y + ch > h or x + cw > w: continue

                # --- LAPLACIAN INFORMATION DENSITY SCORING ---
                
                if core_data['mask'] is not None:
                    actual_visible_pixels = cv2.countNonZero(core_data['mask'])
                    # Create a localized bounding box of the visible area
                    visible_roi = cv2.bitwise_and(core_data['img_bgr'], core_data['img_bgr'], mask=core_data['mask'])
                else:
                    actual_visible_pixels = cw * ch
                    visible_roi = core_data['img_bgr']
                    
                # Convert to grayscale to measure structural texture
                gray_roi = cv2.cvtColor(visible_roi, cv2.COLOR_BGR2GRAY)
                
                # Calculate Laplacian variance (measures internal sharpness/texture)
                laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                
                match_quality = 1.0 - score
                
                # The score now heavily favors tiles with more opaque pixels AND more internal texture
                weighted_score = match_quality * actual_visible_pixels * (1.0 + np.log1p(laplacian_var))

                all_candidates.append({
                    "box": [x, y, cw, ch],
                    "score": float(weighted_score),
                    "key": core_key,
                    "data": core_data,
                    "x": x,
                    "y": y
                })

        if all_candidates:
            boxes = [c['box'] for c in all_candidates]
            scores = [c['score'] for c in all_candidates]
            
            # Global NMS Resolves Conflicts Based on Area & Quality
            survivor_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.15)
            
            if len(survivor_indices) > 0:
                survivors = [all_candidates[i] for i in survivor_indices.flatten()]
                
                for survivor in survivors:
                    x, y = survivor['x'], survivor['y']
                    core_key = survivor['key']
                    core_data = survivor['data']
                    cw, ch = core_data['w'], core_data['h']

                    c_alpha = core_data['c_alpha']

                    # STRICT NON-OVERLAPPING CHECK
                    if not np.any(visited_mask_B[y:y+ch, x:x+cw][c_alpha]):
                        id_grid_B[y:y+ch, x:x+cw][c_alpha] = core_key
                        visited_mask_B[y:y+ch, x:x+cw][c_alpha] = True
                        placed_instances_B.append((core_key, x, y, cw, ch))

                    if not np.any(visited_mask_A[y:y+ch, x:x+cw][c_alpha]):
                        id_grid_A[y:y+ch, x:x+cw][c_alpha] = core_key
                        visited_mask_A[y:y+ch, x:x+cw][c_alpha] = True
                        placed_instances_A.append((core_key, x, y, cw, ch))
                        self.used_templates.add(core_data['filepath'])

        # --- CLEANUP & ANALYSIS ---
        kernel = np.ones((5, 5), np.uint8)

        is_claimed_b_A = (id_grid_A == "B")
        raw_conflict_A = is_occupied_source & is_claimed_b_A
        significant_conflict_A = cv2.morphologyEx(raw_conflict_A.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        id_grid_A[significant_conflict_A] = "UNKNOWN"

        is_claimed_b_B = (id_grid_B == "B")
        raw_conflict_B = is_occupied_source & is_claimed_b_B
        significant_conflict_B = cv2.morphologyEx(raw_conflict_B.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        id_grid_B[significant_conflict_B] = "UNKNOWN"

        if self.cell_w > 0 and self.cell_h > 0:
            cell_area = self.cell_w * self.cell_h
            unique_ids, pixel_counts = np.unique(id_grid_A, return_counts=True)
            for tile_id, p_count in zip(unique_ids, pixel_counts):
                if tile_id in ["UNKNOWN", "P"]: continue
                self.tile_frequencies[tile_id] += (p_count / cell_area)

        ghost_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self._extract_neighbors_from_grid(placed_instances_B, id_grid_B, h, w, target_dict=ghost_rules)

        real_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self._extract_neighbors_from_grid(placed_instances_A, id_grid_A, h, w, target_dict=real_rules)

        clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y-?\d+_x-?\d+")

        for clone_id, rules in list(real_rules.items()):
            match = clone_pattern.match(clone_id)
            if match:
                base_name = match.group(1)       
                core_key_base = match.group(2)   
                core_key = f"{core_key_base}.png" 

                parent_core_key = core_key
                if core_key.startswith("split_L_"):
                    parent_core_key = core_key.replace("split_L_", "")
                elif core_key.startswith("split_R_"):
                    parent_core_key = core_key.replace("split_R_", "")

                if self.tile_frequencies[core_key] == 0:
                    self.tile_frequencies[core_key] = 0.05

                for d in ["top", "bottom", "left", "right"]:
                    neighbors = list(rules[d].items())
                    
                    is_strictly_internal = False
                    if len(neighbors) > 0:
                        is_strictly_internal = True
                        for n_id, _ in neighbors:
                            n_match = clone_pattern.match(n_id)
                            if not (n_match and n_match.group(1) == base_name):
                                is_strictly_internal = False
                                break
                                
                    for n_id, n_count in neighbors:
                        n_match = clone_pattern.match(n_id)
                        if n_match and n_match.group(1) == base_name:
                            n_core_key = f"{n_match.group(2)}.png"
                            real_rules[core_key][d][n_core_key] += n_count

                    if not is_strictly_internal:
                        if parent_core_key in ghost_rules:
                            for gn, gcount in ghost_rules[parent_core_key][d].items():
                                if gn == "B": continue
                                real_rules[clone_id][d][gn] += gcount
                                real_rules[core_key][d][gn] += gcount

        for clone_id, rules in real_rules.items():
            if "_y" in clone_id and "_x" in clone_id:
                for d in ["top", "bottom", "left", "right"]:
                    if clone_id in rules[d]:
                        del rules[d][clone_id]
                        
        self.adjacency_rules = real_rules

        blank_grid_A = np.zeros((h, w, 4), dtype=np.uint8)
        
        for placed_key, x, y, cw, ch in placed_instances_A:
            # Fetch the actual template image instead of the source image
            if placed_key in self.template_cache:
                tile_bgra = self.template_cache[placed_key]['img_bgra']
                tile_alpha = self.template_cache[placed_key]['c_alpha']
                
                # Paste ONLY opaque pixels to prevent bounding boxes overwriting neighbors
                blank_grid_A[y:y+ch, x:x+cw][tile_alpha] = tile_bgra[tile_alpha]
                
            elif placed_key in self.dynamic_templates:
                tile_bgra = self.dynamic_templates[placed_key]
                tile_alpha = tile_bgra[:, :, 3] > 127
                blank_grid_A[y:y+ch, x:x+cw][tile_alpha] = tile_bgra[tile_alpha]
            
        # Save to the Parent Base Directory
        output_file_path = out_base / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid_A)
        
        # EXPORT THE VISIBLE GRID ID'S FOR THE SAVER
        self.visible_reconstruction_ids = np.unique(id_grid_A)

    def _extract_neighbors_from_grid(self, placed_instances, id_grid, level_h, level_w, target_dict=None):
        if target_dict is None:
            target_dict = self.adjacency_rules
            
        TOLERANCE = 12 
        EDGE_INSET = 2 
        
        def get_smart_neighbors(r_start, r_end, c_start, c_end):
            r0 = max(0, min(r_start, level_h))
            r1 = max(0, min(r_end, level_h))
            c0 = max(0, min(c_start, level_w))
            c1 = max(0, min(c_end, level_w))
            
            if r0 >= r1 or c0 >= c1:
                return []

            slice_values = id_grid[r0:r1, c0:c1].flatten()
            unique_vals = np.unique(slice_values)

            if "UNKNOWN" in unique_vals:
                return []
            
            non_background = [v for v in unique_vals if v != "B"]
            
            if len(non_background) > 0:
                return non_background
            else:
                return ["B"] 

        for name, x, y, w, h in placed_instances:
            inset_x = min(w // 3, EDGE_INSET)
            inset_y = min(h // 3, EDGE_INSET)
            
            if y <= TOLERANCE:
                target_dict[name]["top"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y - TOLERANCE, y, 
                    x + inset_x, x + w - inset_x 
                )
                target_dict[name]["top"].update(neighbors)

            if y + h >= level_h - TOLERANCE:
                target_dict[name]["bottom"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + h, y + h + TOLERANCE, 
                    x + inset_x, x + w - inset_x
                )
                target_dict[name]["bottom"].update(neighbors)

            if x <= TOLERANCE:
                target_dict[name]["left"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x - TOLERANCE, x
                )
                target_dict[name]["left"].update(neighbors)

            if x + w >= level_w - TOLERANCE:
                target_dict[name]["right"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x + w, x + w + TOLERANCE
                )
                target_dict[name]["right"].update(neighbors)

    def save_adjacency_rules(self, filepath):
        serializable_rules = {
            k: {d: n.most_common() for d, n in v.items()}
            for k, v in self.adjacency_rules.items()
        }
        with open(filepath, 'w') as f:
            json.dump(serializable_rules, f, indent=4)
        print(f"\nAdjacency rules saved to {filepath}")

    def save_frequencies(self, filepath):
        clean_frequencies = {k: round(v, 2) for k, v in self.tile_frequencies.items()}
        with open(filepath, 'w') as f:
            json.dump(clean_frequencies, f, indent=4)
        print(f"Tile frequencies saved to {filepath}")

    def save_used_tiles(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'visible_reconstruction_ids'):
            return

        required_base_files = set()
        required_dynamic_keys = set()
        clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y-?\d+_x-?\d+")

        # 1. Parse EXACTLY what is physically visible on the final reconstruction grid
        for v_id in self.visible_reconstruction_ids:
            if v_id in ["B", "P", "UNKNOWN"]: 
                continue
                
            match = clone_pattern.match(v_id)
            if match:
                # It's a complex clone. 
                # Group 1: The complex segment file (e.g., mario_9_seg_50.png)
                comp_name = f"{match.group(1)}.png"
                required_base_files.add(comp_name)
                
                # Group 2: The core tile or split (e.g., tile_00494 or split_L_tile_00494)
                core_part = match.group(2)
                if core_part.startswith("split_"):
                    required_dynamic_keys.add(core_part)
                else:
                    required_base_files.add(f"{core_part}.png")
            else:
                # It's a standard core tile
                required_base_files.add(v_id)

        # 2. Save the Base Files (Core Tiles & Raw Complex Segments)
        for req_file in required_base_files:
            if req_file in self.template_cache:
                filepath = self.template_cache[req_file]['filepath']
                img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cv2.imwrite(str(out_dir / req_file), img)

        # 3. Save the Dynamic Templates (Split Tiles from Adapter Merges)
        for dyn_key in required_dynamic_keys:
            if dyn_key in self.dynamic_templates:
                img = self.dynamic_templates[dyn_key]
                filename = dyn_key if dyn_key.endswith('.png') else f"{dyn_key}.png"
                cv2.imwrite(str(out_dir / filename), img)

    def _enforce_rule_symmetry(self):
        print("  -> Enforcing perfect mathematical symmetry on adjacency rules...")
        
        all_tiles = list(self.adjacency_rules.keys())
        opposites = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
        
        for tile_a in all_tiles:
            for direction, opposite_direction in opposites.items():
                for tile_b, count_a in list(self.adjacency_rules[tile_a][direction].items()):
                    if tile_b == "P": 
                        continue
                        
                    count_b = self.adjacency_rules[tile_b][opposite_direction].get(tile_a, 0)
                    max_count = max(count_a, count_b)
                    
                    self.adjacency_rules[tile_a][direction][tile_b] = max_count
                    self.adjacency_rules[tile_b][opposite_direction][tile_a] = max_count

        print("  -> Pruning dead tiles with empty boundaries...")
        is_pruning = True
        
        while is_pruning:
            is_pruning = False
            tiles_to_remove = set()
            
            for tile, rules in self.adjacency_rules.items():
                if tile in ["B", "P"]: continue 
                
                for d in ["top", "bottom", "left", "right"]:
                    if len(rules[d]) == 0:
                        tiles_to_remove.add(tile)
                        break 

            if not tiles_to_remove:
                break
                
            for tile in tiles_to_remove:
                if tile in self.adjacency_rules: del self.adjacency_rules[tile]
                if tile in self.tile_frequencies: del self.tile_frequencies[tile]
                is_pruning = True
                
            for tile, rules in self.adjacency_rules.items():
                for d in ["top", "bottom", "left", "right"]:
                    for dead_tile in tiles_to_remove:
                        if dead_tile in rules[d]:
                            del rules[d][dead_tile]

    def run_reconstruction(self, base_output_dir):
        # 1. Load Source Levels
        self._load_source_levels()
        if not self.source_image_cache: return

        # 2. Cache the Global Dictionary ONCE for all levels
        self._load_global_core_templates()

        print("\n--- Starting Level Reconstruction & Analysis ---")
        out_base = Path(base_output_dir)
        
        for level_name, level_img in tqdm(self.source_image_cache.items(), desc="Processing Levels"):
            level_stem = Path(level_name).stem
            
            self.adjacency_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
            self.tile_frequencies = defaultdict(float)
            self.used_templates = set()
            self.dynamic_templates = {}
            self.cell_w = 0
            self.cell_h = 0

            # 3. Inject level-specific segments alongside the global dictionary
            self.template_cache = self._load_templates_for_level(level_stem)
            if not self.template_cache: continue

            self.unmatched_templates = set(self.template_cache.keys())
            
            
            self._reconstruct_single_level_complex(level_name, level_img, level_stem, out_base)
            

            level_output_dir = out_base / f"{level_stem}_data"
            level_output_dir.mkdir(parents=True, exist_ok=True)

            self._enforce_rule_symmetry()
            self.save_adjacency_rules(level_output_dir / "adjacency_rules.txt")
            self.save_frequencies(level_output_dir / "ratios.json")
            
            # 4. Save exactly which tiles survived NMS and were used for this specific level
            self.save_used_tiles(level_output_dir / "tiles")

if __name__ == "__main__":
    base_dir = "output_segments/coreV4"
    
    reconstructor = LevelReconstructor(
        global_core_path="output_segments/newCoreV10", # ADDED: Path to the flat folder of all core tiles
        segment_path="",   # Path to the raw level segments   output_segments/transRemake
        level_path="demo/imgs/test",                  # Path to original levels
        output_path=base_dir,
        match_threshold=0.02,
        
    )
    
    reconstructor.run_reconstruction(
        base_output_dir="Generation/coreV4"
    )