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
    Reconstructs level images using ONLY segments found in a folder matching the level's name.
    Now supports Complex Tiles via a Two-Stage Masked Solver with a Ghost Pass.
    """
    
    def __init__(self, segment_path, level_path, output_path, match_threshold=80000, use_complex_tiles=False):
        self.segments_base_path = Path(segment_path)
        self.level_path = Path(level_path)
        self.output_path = Path(output_path)
        self.MATCH_THRESHOLD = match_threshold
        self.USE_COMPLEX_TILES = use_complex_tiles

        self.template_cache = {} 
        self.source_image_cache = {}
        self.unmatched_templates = set()
        self.used_templates = set() 
        self.dynamic_templates = {}

        # --- Global Rule Dictionary ---
        self.adjacency_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})

        self.tile_frequencies = defaultdict(float)
        self.template_padding = {} 
        self.cell_w = 0
        self.cell_h = 0

        self.output_path.mkdir(parents=True, exist_ok=True)

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

    def _load_templates_for_level(self, level_stem):
        specific_folder = self.segments_base_path / level_stem
        
        if not specific_folder.exists() or not specific_folder.is_dir():
            return None

        image_files = list(specific_folder.glob('*.png'))
        if not image_files: 
            return None
        
        print(f"Loading {len(image_files)} templates from {specific_folder}...")
        
        templates = {}
        for img_path in image_files:
            img_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_bgra is None: continue
            
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
            template_key = img_path.name 

            erosion_kernel = np.ones((5, 5), np.uint8)
            match_mask = cv2.erode(mask, erosion_kernel, iterations=1)
            if cv2.countNonZero(match_mask) == 0:
                match_mask = mask

            templates[template_key] = {
                'img_bgr': img_bgr,
                'img_bgra': img_bgra,
                'mask': mask,
                'match_mask': match_mask,
                'w': w,
                'h': h,
                'is_valid': is_valid,
                'area': w * h,
                'filepath': img_path 
            }
        return templates

    def _reconstruct_single_level_complex(self, level_name, level_img_original, level_stem):
        print(f"\n  -> Running Two-Stage Masked Solver (Ghost Pass Enabled) for {level_name}")
        h, w, _ = level_img_original.shape
        
        # --- INIT GRIDS FOR BOTH PASSES ---
        id_grid_A = np.full((h, w), "B", dtype=object)
        visited_mask_A = np.zeros((h, w), dtype=bool)
        placed_instances_A = []
        
        id_grid_B = np.full((h, w), "B", dtype=object)
        visited_mask_B = np.zeros((h, w), dtype=bool)
        placed_instances_B = []

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            level_img_for_matching = cv2.cvtColor(level_img_original, cv2.COLOR_BGRA2BGR)
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
        
        core_sorted = sorted(core_templates.items(), key=lambda item: item[1]['area'], reverse=True)
        complex_sorted = sorted(complex_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        if core_sorted:
            self.cell_w = Counter([t['w'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]
            self.cell_h = Counter([t['h'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]

        OVERLAP_TOLERANCE = 4 
        kernel_size = OVERLAP_TOLERANCE * 2 + 1
        shrink_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        x_offsets = []
        for template_key, template_data in core_sorted[:5]: 
            if not template_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_for_matching, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['match_mask'])
                locs = np.where(res <= self.MATCH_THRESHOLD)
                for start_x in locs[1]:
                    x_offsets.append(start_x % self.cell_w)
            except cv2.error:
                continue

        global_offset_x = Counter(x_offsets).most_common(1)[0][0] if x_offsets else 0

        # ---------------------------------------------------------
        # PHASE 1.1: Core Injection inside Complex ROIs (PASS A)
        # ---------------------------------------------------------
        for comp_key, comp_data in complex_sorted:
            if not comp_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_for_matching, comp_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=comp_data['match_mask'])
                locs = np.where(res <= self.MATCH_THRESHOLD)
                matches = [(x, y, res[y, x]) for y, x in zip(*locs)]
                matches.sort(key=lambda x: x[2])

                for gx, gy, _ in matches:
                    comp_h, comp_w = comp_data['h'], comp_data['w']
                    if gy + comp_h > h or gx + comp_w > w: continue

                    buffer = 2
                    roi_y1 = max(0, gy - buffer)
                    roi_y2 = min(h, gy + comp_h + buffer)
                    roi_x1 = max(0, gx - buffer)
                    roi_x2 = min(w, gx + comp_w + buffer)
                    
                    roi_bgr = level_img_for_matching[roi_y1:roi_y2, roi_x1:roi_x2]

                    for core_key, core_data in core_sorted:
                        if not core_data['is_valid']: continue
                        if roi_bgr.shape[0] < core_data['h'] or roi_bgr.shape[1] < core_data['w']: continue

                        c_res = cv2.matchTemplate(roi_bgr, core_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=core_data['match_mask'])
                        c_locs = np.where(c_res <= self.MATCH_THRESHOLD)
                        c_matches = [(lx, ly, c_res[ly, lx]) for ly, lx in zip(*c_locs)]
                        c_matches.sort(key=lambda x: x[2])

                        for lx, ly, _ in c_matches:
                            cw, ch = core_data['w'], core_data['h']
                            
                            global_x = roi_x1 + lx
                            global_y = roi_y1 + ly

                            if global_y + ch > h or global_x + cw > w: continue

                            c_alpha = core_data['img_bgra'][:, :, 3] > 127
                            c_shrunk = cv2.erode((c_alpha.astype(np.uint8)*255), shrink_kernel, iterations=1) > 127
                            if not np.any(c_shrunk): c_shrunk = c_alpha

                            if np.any(visited_mask_A[global_y:global_y+ch, global_x:global_x+cw][c_shrunk]):
                                continue

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
            except cv2.error:
                continue

        # ---------------------------------------------------------
        # PHASE 1.2: Generic Fill & Splitting Logic (PASS A & B)
        # ---------------------------------------------------------
        for core_key, core_data in core_sorted:
            if not core_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_for_matching, core_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=core_data['match_mask'])
                locs = np.where(res <= self.MATCH_THRESHOLD)
                matches = [(x, y, res[y, x]) for y, x in zip(*locs)]
                matches.sort(key=lambda x: x[2])

                for x, y, _ in matches:
                    cw, ch = core_data['w'], core_data['h']
                    if y + ch > h or x + cw > w: continue

                    c_alpha = core_data['img_bgra'][:, :, 3] > 127
                    c_shrunk = cv2.erode((c_alpha.astype(np.uint8)*255), shrink_kernel, iterations=1) > 127
                    if not np.any(c_shrunk): c_shrunk = c_alpha

                    # -- GHOST PASS (B) --
                    # Ghost pass runs completely independent of complex clones
                    if not np.any(visited_mask_B[y:y+ch, x:x+cw][c_shrunk]):
                        id_grid_B[y:y+ch, x:x+cw][c_alpha] = core_key
                        visited_mask_B[y:y+ch, x:x+cw][c_alpha] = True
                        placed_instances_B.append((core_key, x, y, cw, ch))

                    # -- REAL PASS (A) --
                    if not np.any(visited_mask_A[y:y+ch, x:x+cw][c_shrunk]):
                        relative_x = x - global_offset_x
                        offset_x = relative_x % self.cell_w
                        
                        is_middle_object = False
                        if self.cell_w > 0:
                            is_middle_object = abs(offset_x - (self.cell_w // 2)) <= 1

                        if is_middle_object:
                            grid_x_left = x - offset_x
                            grid_x_right = x - offset_x + self.cell_w
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

                            visited_mask_A[y:y+ch, x:x+cw][c_alpha] = True

                            if not is_l_pure_bg:
                                left_key = f"split_L_{core_key}"
                                if left_key not in self.dynamic_templates:
                                    self.dynamic_templates[left_key] = left_bgra
                                    
                                l_start_x = max(0, grid_x_left)
                                l_end_x = min(w, grid_x_left + self.cell_w)
                                l_vis_w = l_end_x - l_start_x
                                
                                if l_vis_w > 0:
                                    l_mask_start = 0 if grid_x_left >= 0 else -grid_x_left
                                    l_mask_end = l_mask_start + l_vis_w
                                    chunk_l_alpha = left_bgra[:, l_mask_start:l_mask_end, 3] > 127
                                    id_grid_A[y:y+ch, l_start_x:l_end_x][chunk_l_alpha] = left_key
                                    placed_instances_A.append((left_key, grid_x_left, y, self.cell_w, ch))

                            if not is_r_pure_bg:
                                right_key = f"split_R_{core_key}"
                                if right_key not in self.dynamic_templates:
                                    self.dynamic_templates[right_key] = right_bgra
                                    
                                r_start_x = max(0, grid_x_right)
                                r_end_x = min(w, grid_x_right + self.cell_w)
                                r_vis_w = r_end_x - r_start_x
                                
                                if r_vis_w > 0:
                                    r_mask_start = 0 if grid_x_right >= 0 else -grid_x_right
                                    r_mask_end = r_mask_start + r_vis_w
                                    chunk_r_alpha = right_bgra[:, r_mask_start:r_mask_end, 3] > 127
                                    id_grid_A[y:y+ch, r_start_x:r_end_x][chunk_r_alpha] = right_key
                                    placed_instances_A.append((right_key, grid_x_right, y, self.cell_w, ch))
                        else:
                            id_grid_A[y:y+ch, x:x+cw][c_alpha] = core_key
                            visited_mask_A[y:y+ch, x:x+cw][c_alpha] = True
                            placed_instances_A.append((core_key, x, y, cw, ch))
                            self.used_templates.add(core_data['filepath'])

            except cv2.error:
                continue

        # --- UNKNOWN CLEANUP & FREQUENCY COUNTING ---
        is_claimed_b = (id_grid_A == "B")
        raw_conflict = is_occupied_source & is_claimed_b
        kernel = np.ones((5, 5), np.uint8)
        significant_conflict = cv2.morphologyEx(raw_conflict.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        significant_conflict = significant_conflict.astype(bool)
        id_grid_A[significant_conflict] = "UNKNOWN"

        if self.cell_w > 0 and self.cell_h > 0:
            cell_area = self.cell_w * self.cell_h
            unique_ids, pixel_counts = np.unique(id_grid_A, return_counts=True)
            for tile_id, p_count in zip(unique_ids, pixel_counts):
                if tile_id in ["UNKNOWN", "B", "P"]: continue
                
                # 1. Keep the exact clone ID frequency for precise WFC pacing
                self.tile_frequencies[tile_id] += (p_count / cell_area)
                
                # 2. Extract and assign frequency to the base generic core as well
                match = re.match(r"(.+?)_(tile_.+)_y\d+_x\d+", str(tile_id))
                if match:
                    base_tile = f"{match.group(2)}.png"
                    self.tile_frequencies[base_tile] += (p_count / cell_area)

        # ---------------------------------------------------------
        # PHASE 2: GHOST PASS RULES EXTRACTION
        # ---------------------------------------------------------
        ghost_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self._extract_neighbors_from_grid(placed_instances_B, id_grid_B, h, w, target_dict=ghost_rules)

        # ---------------------------------------------------------
        # PHASE 3: THE ADAPTER MERGE
        # ---------------------------------------------------------
        real_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self._extract_neighbors_from_grid(placed_instances_A, id_grid_A, h, w, target_dict=real_rules)

        clone_pattern = re.compile(r"(.+?)_(tile_.+)_y\d+_x\d+")

        for clone_id, rules in list(real_rules.items()):
            match = clone_pattern.match(clone_id)
            if match:
                base_name = match.group(1)       
                core_key_base = match.group(2)   
                core_key = f"{core_key_base}.png" 

                # 1. Populate the generic core tile with its Ghost Pass rules
                if core_key in ghost_rules:
                    for d in ["top", "bottom", "left", "right"]:
                        for gn, gcount in ghost_rules[core_key][d].items():
                            real_rules[core_key][d][gn] += gcount
                            
                # Give generic cores a baseline frequency
                if self.tile_frequencies[core_key] == 0:
                    self.tile_frequencies[core_key] = 0.05

                # 2. Dynamic Boundary Edge Detection
                for d in ["top", "bottom", "left", "right"]:
                    neighbors = list(rules[d].items())
                    
                    # A face is strictly internal ONLY if all its connections are siblings
                    is_strictly_internal = False
                    if len(neighbors) > 0:
                        is_strictly_internal = True
                        for n_id, _ in neighbors:
                            n_match = clone_pattern.match(n_id)
                            if not (n_match and n_match.group(1) == base_name):
                                is_strictly_internal = False
                                break
                                
                    # Learn generic internal structure for the base core tiles
                    for n_id, n_count in neighbors:
                        n_match = clone_pattern.match(n_id)
                        if n_match and n_match.group(1) == base_name:
                            n_core_key = f"{n_match.group(2)}.png"
                            real_rules[core_key][d][n_core_key] += n_count

                    # Apply Ghost rules ONLY to external boundaries
                    if not is_strictly_internal:
                        if core_key in ghost_rules:
                            for gn, gcount in ghost_rules[core_key][d].items():
                                real_rules[clone_id][d][gn] += gcount

        # for core_key, core_data in core_templates.items():
        #     # 1. If the core tile was never placed standalone, give it its Ghost Pass rules
        #     if core_key not in real_rules:
        #         if core_key in ghost_rules:
        #             real_rules[core_key] = {
        #                 d: Counter(ghost_rules[core_key][d]) for d in ["top", "bottom", "left", "right"]
        #             }
            
        #     # 2. Enforce a small baseline frequency so the generator can use it to start/fill lines
        #     if self.tile_frequencies[core_key] < 0.05:
        #         self.tile_frequencies[core_key] = 0.05
                
        #     # 3. Ensure the image asset gets copied to the output tiles directory
        #     self.used_templates.add(core_data['filepath'])
        for clone_id, rules in real_rules.items():
            if "_y" in clone_id and "_x" in clone_id:
                for d in ["top", "bottom", "left", "right"]:
                    if clone_id in rules[d]:
                        del rules[d][clone_id]
                        
        self.adjacency_rules = real_rules

        # Create output visualization
        blank_grid_A = np.zeros((h, w, 4), dtype=np.uint8)
        for _, x, y, cw, ch in placed_instances_A:
            roi_original = level_img_original[y:y+ch, x:x+cw]
            blank_grid_A[y:y+ch, x:x+cw] = roi_original
        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid_A)

    def _reconstruct_single_level(self, level_name, level_img_original, level_stem):
        """
        Original logic for USE_COMPLEX_TILES = False
        """
        h, w, _ = level_img_original.shape
        blank_grid = np.zeros((h, w, 4), dtype=np.uint8)
        id_grid = np.full((h, w), "B", dtype=object)
        visited_mask = np.zeros((h, w), dtype=bool)
        overlap_heatmap = np.zeros((h, w), dtype=np.uint16)

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            level_img_for_matching = cv2.cvtColor(level_img_original, cv2.COLOR_BGRA2BGR)
        else:
            level_img_for_matching = level_img_original

        level_templates = {k: v for k, v in self.template_cache.items()} 
        placed_instances = []

        sorted_templates = sorted(level_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        valid_widths = [t['w'] for t in self.template_cache.values() if t['is_valid']]
        valid_heights = [t['h'] for t in self.template_cache.values() if t['is_valid']]
        
        if valid_widths and valid_heights:
            self.cell_w = Counter(valid_widths).most_common(1)[0][0]
            self.cell_h = Counter(valid_heights).most_common(1)[0][0]
        else:
            return 
            
        print(f"  -> Unsupervised Grid Detection: {self.cell_w}x{self.cell_h}")

        x_offsets = []
        for template_key, template_data in sorted_templates[:5]: 
            if not template_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_for_matching, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['match_mask'])
                locs = np.where(res <= self.MATCH_THRESHOLD)
                for start_x in locs[1]:
                    x_offsets.append(start_x % self.cell_w)
            except cv2.error:
                continue

        global_offset_x = Counter(x_offsets).most_common(1)[0][0] if x_offsets else 0
        print(f"  -> Auto-calibrated Grid Offset: {global_offset_x} pixels")

        pass_thresholds = [self.MATCH_THRESHOLD, self.MATCH_THRESHOLD * 2.0]

        for pass_num, current_threshold in enumerate(pass_thresholds):
            for template_key, template_data in sorted_templates:
                if not template_data['is_valid']: continue
                
                t_h, t_w = template_data['h'], template_data['w']
                if t_h > h or t_w > w: continue

                try:
                    result = cv2.matchTemplate(level_img_for_matching, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['match_mask'])
                    locations = np.where(result <= self.MATCH_THRESHOLD)
                    
                    matches = []
                    for y, x in zip(*locations):
                        score = result[y, x]
                        matches.append((x, y, score))
                    
                    matches.sort(key=lambda x: x[2])

                    if len(matches) > 0 and template_key in self.unmatched_templates:
                        self.unmatched_templates.remove(template_key)

                    for x, y, score in matches:
                        alpha_mask = template_data['img_bgra'][:, :, 3] > 127

                        OVERLAP_TOLERANCE = 4 
                        kernel_size = OVERLAP_TOLERANCE * 2 + 1
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        
                        alpha_uint8 = alpha_mask.astype(np.uint8) * 255
                        shrunk_mask_uint8 = cv2.erode(alpha_uint8, kernel, iterations=1)
                        shrunk_mask = shrunk_mask_uint8 > 127
                        
                        if not np.any(shrunk_mask):
                            shrunk_mask = alpha_mask

                        roi_visited = visited_mask[y:y + t_h, x:x + t_w]
                        if np.any(roi_visited[shrunk_mask]):
                            continue
                            
                        relative_x = x - global_offset_x
                        offset_x = relative_x % self.cell_w
                        
                        is_middle_object = False
                        if self.cell_w > 0:
                            is_middle_object = abs(offset_x - (self.cell_w // 2)) <= 1

                        if is_middle_object:
                            grid_x_left = x - offset_x
                            grid_x_right = x - offset_x + self.cell_w
                            split_pt = self.cell_w - offset_x
                            
                            bg_color = level_img_original[0, 0, :3]

                            left_bgra = np.zeros((t_h, self.cell_w, 4), dtype=np.uint8)
                            actual_split_w = min(split_pt, t_w)
                            left_bgra[:, offset_x : offset_x + actual_split_w] = template_data['img_bgra'][:, :actual_split_w]
                            
                            l_alpha = left_bgra[:, :, 3] > 127
                            is_l_pure_bg = True
                            if np.any(l_alpha):
                                mean_color = np.mean(left_bgra[l_alpha, :3], axis=0)
                                std_color = np.std(left_bgra[l_alpha, :3], axis=0)
                                if np.linalg.norm(mean_color - bg_color) < 15.0 and np.all(std_color < 5.0):
                                    is_l_pure_bg = True
                                else:
                                    is_l_pure_bg = False

                            right_bgra = np.zeros((t_h, self.cell_w, 4), dtype=np.uint8)
                            if t_w > split_pt:
                                actual_rem_w = min(t_w - split_pt, self.cell_w)
                                right_bgra[:, :actual_rem_w] = template_data['img_bgra'][:, split_pt : split_pt + actual_rem_w]
                            
                            r_alpha = right_bgra[:, :, 3] > 127
                            is_r_pure_bg = True
                            if np.any(r_alpha):
                                mean_color = np.mean(right_bgra[r_alpha, :3], axis=0)
                                std_color = np.std(right_bgra[r_alpha, :3], axis=0)
                                if np.linalg.norm(mean_color - bg_color) < 15.0 and np.all(std_color < 5.0):
                                    is_r_pure_bg = True
                                else:
                                    is_r_pure_bg = False

                            overlap_heatmap[y:y + t_h, x:x + t_w][alpha_mask] += 1
                            roi_grid = blank_grid[y:y + t_h, x:x + t_w]
                            roi_grid[alpha_mask] = template_data['img_bgra'][alpha_mask]
                            blank_grid[y:y + t_h, x:x + t_w] = roi_grid
                            visited_mask[y:y + t_h, x:x + t_w][alpha_mask] = True

                            if not is_l_pure_bg:
                                left_key = f"split_L_{template_key}"
                                if left_key not in self.dynamic_templates:
                                    self.dynamic_templates[left_key] = left_bgra
                                    
                                l_start_x = max(0, grid_x_left)
                                l_end_x = min(w, grid_x_left + self.cell_w)
                                l_vis_w = l_end_x - l_start_x
                                
                                if l_vis_w > 0:
                                    l_mask_start = 0 if grid_x_left >= 0 else -grid_x_left
                                    l_mask_end = l_mask_start + l_vis_w
                                    chunk_l_alpha = left_bgra[:, l_mask_start:l_mask_end, 3] > 127
                                    id_grid[y:y + t_h, l_start_x:l_end_x][chunk_l_alpha] = left_key
                                    placed_instances.append((left_key, grid_x_left, y, self.cell_w, t_h))

                            if not is_r_pure_bg:
                                right_key = f"split_R_{template_key}"
                                if right_key not in self.dynamic_templates:
                                    self.dynamic_templates[right_key] = right_bgra
                                    
                                r_start_x = max(0, grid_x_right)
                                r_end_x = min(w, grid_x_right + self.cell_w)
                                r_vis_w = r_end_x - r_start_x
                                
                                if r_vis_w > 0:
                                    r_mask_start = 0 if grid_x_right >= 0 else -grid_x_right
                                    r_mask_end = r_mask_start + r_vis_w
                                    chunk_r_alpha = right_bgra[:, r_mask_start:r_mask_end, 3] > 127
                                    id_grid[y:y + t_h, r_start_x:r_end_x][chunk_r_alpha] = right_key
                                    placed_instances.append((right_key, grid_x_right, y, self.cell_w, t_h))
                                    
                        else:
                            overlap_heatmap[y:y + t_h, x:x + t_w][alpha_mask] += 1
                            roi_grid = blank_grid[y:y + t_h, x:x + t_w]
                            roi_grid[alpha_mask] = template_data['img_bgra'][alpha_mask]
                            blank_grid[y:y + t_h, x:x + t_w] = roi_grid
                            
                            roi_ids = id_grid[y:y + t_h, x:x + t_w]
                            roi_ids[alpha_mask] = template_key
                            id_grid[y:y + t_h, x:x + t_w] = roi_ids

                            visited_mask[y:y + t_h, x:x + t_w][alpha_mask] = True
                            placed_instances.append((template_key, x, y, t_w, t_h))
                            
                            self.used_templates.add(template_data['filepath'])
                        
                except cv2.error:
                    continue

        if level_img_original.shape[2] == 4:
            source_alpha = level_img_original[:, :, 3]
            is_occupied_source = source_alpha > 10
        else:
            bg_color = level_img_original[0, 0]
            diff = cv2.absdiff(level_img_original, bg_color)
            diff_sum = np.sum(diff, axis=2)
            is_occupied_source = diff_sum > 10

        is_claimed_b = (id_grid == "B")
        raw_conflict = is_occupied_source & is_claimed_b
        kernel = np.ones((5, 5), np.uint8)
        significant_conflict = cv2.morphologyEx(raw_conflict.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        significant_conflict = significant_conflict.astype(bool)
        id_grid[significant_conflict] = "UNKNOWN"

        if self.cell_w > 0 and self.cell_h > 0:
            cell_area = self.cell_w * self.cell_h
            unique_ids, pixel_counts = np.unique(id_grid, return_counts=True)
            for tile_id, p_count in zip(unique_ids, pixel_counts):
                if tile_id == "UNKNOWN": continue
                self.tile_frequencies[tile_id] += (p_count / cell_area)

        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid)
        
        self._extract_neighbors_from_grid(placed_instances, id_grid, h, w)

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
        clone_pattern = re.compile(r"(.+?)_(tile_.+)_y\d+_x\d+")

        required_base_files = set()

        # 1. Determine exactly which files are required by the final JSON rules
        for rule_key in self.adjacency_rules.keys():
            if rule_key in ["B", "P", "UNKNOWN"]: continue

            match = clone_pattern.match(rule_key)
            if match:
                # If a complex clone is required, extract its core tile name
                required_base_files.add(f"{match.group(2)}.png")
            else:
                # Add standard tiles and split tiles
                required_base_files.add(rule_key)

        # 2. Save standard templates safely from the cache
        for req_file in required_base_files:
            if req_file in self.template_cache:
                filepath = self.template_cache[req_file]['filepath']
                img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cv2.imwrite(str(out_dir / req_file), img)

        # 3. Save dynamic split templates
        for name, img in self.dynamic_templates.items():
            # The dynamic template keys for splits already include .png (e.g., split_L_tile_00008.png)
            if name in required_base_files:
                cv2.imwrite(str(out_dir / name), img)

    def _enforce_rule_symmetry(self):
        print("  -> Enforcing perfect mathematical symmetry on adjacency rules...")
        
        all_tiles = list(self.adjacency_rules.keys())
        opposites = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left"
        }
        
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
                if tile in self.adjacency_rules:
                    del self.adjacency_rules[tile]
                if tile in self.tile_frequencies:
                    del self.tile_frequencies[tile]
                
                # Dynamic templates check for splits
                if tile in self.dynamic_templates:
                    del self.dynamic_templates[tile]
                    
                is_pruning = True
                
            for tile, rules in self.adjacency_rules.items():
                for d in ["top", "bottom", "left", "right"]:
                    for dead_tile in tiles_to_remove:
                        if dead_tile in rules[d]:
                            del rules[d][dead_tile]


    def run_reconstruction(self, base_output_dir):
        self._load_source_levels()

        if not self.source_image_cache:
            return

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

            self.template_cache = self._load_templates_for_level(level_stem)
            
            if not self.template_cache:
                continue

            valid_t = [t for t in self.template_cache.values() if t['is_valid']]
            if valid_t:
                self.cell_w = min(t['w'] for t in valid_t)
                self.cell_h = min(t['h'] for t in valid_t)

            self.unmatched_templates = set(self.template_cache.keys())
            
            # Switch between normal and ghost pass logic based on the flag
            if self.USE_COMPLEX_TILES:
                self._reconstruct_single_level_complex(level_name, level_img, level_stem)
            else:
                self._reconstruct_single_level(level_name, level_img, level_stem)

            level_output_dir = out_base / f"{level_stem}_data"
            level_output_dir.mkdir(parents=True, exist_ok=True)

            self._enforce_rule_symmetry()
            self.save_adjacency_rules(level_output_dir / "adjacency_rules.txt")
            self.save_frequencies(level_output_dir / "ratios.json")
            self.save_used_tiles(level_output_dir / "tiles")


if __name__ == "__main__":
    base_dir = "output_segments/mixedSizes"
    
    reconstructor = LevelReconstructor(
        segment_path=base_dir,
        level_path="demo/imgs/test",
        output_path=base_dir,
        match_threshold=0.05,
        use_complex_tiles=True 
    )
    

    reconstructor.run_reconstruction(
        base_output_dir="Generation/mixedSizesV3"
    )