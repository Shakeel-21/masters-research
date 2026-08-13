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
    Supports Complex Tiles via a Two-Stage Masked Solver with a Ghost Pass.
    """
    
    def __init__(self, segment_path, level_path, output_path, match_threshold=0.08, use_complex_tiles=False):
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

        self.adjacency_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})
        self.tile_frequencies = defaultdict(float)
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
            img_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED) 
            if img_bgra is not None:
                self.source_image_cache[img_path.name] = {
                    'bgra': img_bgra
                }

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

            active_pixels = cv2.countNonZero(mask)
            bounding_area = w * h
            solidity = active_pixels / bounding_area if bounding_area > 0 else 0

            templates[template_key] = {
                'img_bgr': img_bgr,
                'img_bgra': img_bgra,
                'mask': mask,
                'w': w,
                'h': h,
                'is_valid': is_valid,
                'area': active_pixels,
                'solidity': solidity,
                'filepath': img_path 
            }
        return templates

    def _reconstruct_single_level_complex(self, level_name, level_img_original, level_stem):
        print(f"\n  -> Running Spatial Geometric Solver (Global NMS & Mask) for {level_name}")
        h, w, _ = level_img_original.shape
        
        id_grid_A = np.full((h, w), "B", dtype=object)
        visited_mask_A = np.zeros((h, w), dtype=bool)
        placed_instances_A = []
        
        id_grid_B = np.full((h, w), "B", dtype=object)
        visited_mask_B = np.zeros((h, w), dtype=bool)
        placed_instances_B = []

        level_img_bgr = level_img_original[:, :, :3]

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            source_alpha = level_img_original[:, :, 3]
            is_occupied_source = source_alpha > 10
        else:
            bg_color = level_img_original[0, 0]
            diff = cv2.absdiff(level_img_original, bg_color)
            diff_sum = np.sum(diff, axis=2)
            is_occupied_source = diff_sum > 10

        core_templates = {k: v for k, v in self.template_cache.items() if k.startswith("tile_")}
        complex_templates = {k: v for k, v in self.template_cache.items() if not k.startswith("tile_")}
        
        core_sorted = sorted(
            core_templates.items(), 
            key=lambda item: (item[1]['solidity'], item[1]['area']), 
            reverse=True
        )
        complex_sorted = sorted(complex_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        if core_sorted:
            self.cell_w = Counter([t['w'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]
            self.cell_h = Counter([t['h'] for _, t in core_sorted if t['is_valid']]).most_common(1)[0][0]

        OVERLAP_TOLERANCE = 3 
        shrink_kernel = np.ones((OVERLAP_TOLERANCE * 2 + 1, OVERLAP_TOLERANCE * 2 + 1), np.uint8)

        # ---------------------------------------------------------
        # PHASE 1.1: Complex ROIs (PASS A)
        # ---------------------------------------------------------
        for comp_key, comp_data in complex_sorted:
            if not comp_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_bgr, comp_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=comp_data['mask'])
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
                    
                    roi_bgr = level_img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

                    roi_candidates = []
                    for core_key, core_data in core_sorted:
                        if not core_data['is_valid']: continue
                        if roi_bgr.shape[0] < core_data['h'] or roi_bgr.shape[1] < core_data['w']: continue

                        c_res = cv2.matchTemplate(roi_bgr, core_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=core_data['mask'])
                        c_locs = np.where(c_res <= self.MATCH_THRESHOLD)
                        
                        for ly, lx in zip(*c_locs):
                            error = c_res[ly, lx]
                            weighted_score = (1.0 - error) * core_data['area']
                            roi_candidates.append({
                                "score": float(weighted_score),
                                "x": int(lx), "y": int(ly),
                                "core_key": core_key, "core_data": core_data
                            })
                            
                    roi_candidates.sort(key=lambda c: c["score"], reverse=True)

                    for cand in roi_candidates:
                        core_key, core_data = cand["core_key"], cand["core_data"]
                        lx, ly = cand["x"], cand["y"]
                        cw, ch = core_data['w'], core_data['h']
                        
                        global_x, global_y = roi_x1 + lx, roi_y1 + ly
                        if global_y + ch > h or global_x + cw > w: continue

                        c_alpha = core_data['img_bgra'][:, :, 3] > 127
                        c_shrunk = cv2.erode((c_alpha.astype(np.uint8)*255), shrink_kernel, iterations=1) > 127
                        if not np.any(c_shrunk): c_shrunk = c_alpha

                        if np.any(visited_mask_A[global_y:global_y+ch, global_x:global_x+cw][c_shrunk]):
                            continue

                        # Generate unique clone keys for complex rules
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
        # PHASE 1.2: Generic Fill (Global Area-Weighted NMS Voting)
        # ---------------------------------------------------------
        global_candidates = []
        
        for core_key, core_data in core_sorted:
            if not core_data['is_valid']: continue
            try:
                res = cv2.matchTemplate(level_img_bgr, core_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=core_data['mask'])
                locs = np.where(res <= self.MATCH_THRESHOLD)
                
                for y, x in zip(*locs):
                    error = res[y, x]
                    weighted_score = (1.0 - error) * core_data['area']
                    
                    global_candidates.append({
                        "box": [int(x), int(y), int(core_data['w']), int(core_data['h'])],
                        "score": float(weighted_score),
                        "x": int(x), "y": int(y),
                        "core_key": core_key,
                        "core_data": core_data
                    })
            except cv2.error:
                continue

        if global_candidates:
            boxes = [c["box"] for c in global_candidates]
            scores = [c["score"] for c in global_candidates]
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.30)
            
            if len(indices) > 0:
                surviving = [global_candidates[i] for i in indices.flatten()]
                surviving.sort(key=lambda c: c["score"], reverse=True)

                for cand in surviving:
                    core_key = cand["core_key"]
                    core_data = cand["core_data"]
                    x, y = cand["x"], cand["y"]
                    cw, ch = core_data['w'], core_data['h']
                    
                    if y + ch > h or x + cw > w: continue

                    c_alpha = core_data['img_bgra'][:, :, 3] > 127
                    c_shrunk = cv2.erode((c_alpha.astype(np.uint8)*255), shrink_kernel, iterations=1) > 127
                    if not np.any(c_shrunk): c_shrunk = c_alpha

                    total_pixels = np.sum(c_shrunk)
                    if total_pixels == 0: continue

                    target_B = visited_mask_B[y:y+ch, x:x+cw]
                    overlap_B = np.sum(target_B[c_shrunk])
                    
                    if (overlap_B / total_pixels) < 0.30:
                        id_grid_B[y:y+ch, x:x+cw][c_alpha] = core_key
                        visited_mask_B[y:y+ch, x:x+cw][c_alpha] = True
                        placed_instances_B.append((core_key, x, y, cw, ch))

                    target_A = visited_mask_A[y:y+ch, x:x+cw]
                    overlap_A = np.sum(target_A[c_shrunk])

                    if (overlap_A / total_pixels) < 0.30:
                        id_grid_A[y:y+ch, x:x+cw][c_alpha] = core_key
                        visited_mask_A[y:y+ch, x:x+cw][c_alpha] = True
                        placed_instances_A.append((core_key, x, y, cw, ch))
                        self.used_templates.add(core_data['filepath'])

        # --- UNKNOWN CLEANUP (4-Pixel Buffer Pruning) ---
        pruning_buffer_kernel = np.ones((4, 4), np.uint8)

        is_claimed_b_A = (id_grid_A == "B")
        raw_conflict_A = is_occupied_source & is_claimed_b_A
        # Erosion requires the entire 4x4 buffer to be True (conflicting) to trigger UNKNOWN
        significant_conflict_A = cv2.erode(raw_conflict_A.astype(np.uint8), pruning_buffer_kernel, iterations=1).astype(bool)
        id_grid_A[significant_conflict_A] = "UNKNOWN"

        is_claimed_b_B = (id_grid_B == "B")
        raw_conflict_B = is_occupied_source & is_claimed_b_B
        significant_conflict_B = cv2.erode(raw_conflict_B.astype(np.uint8), pruning_buffer_kernel, iterations=1).astype(bool)
        id_grid_B[significant_conflict_B] = "UNKNOWN"

        if self.cell_w > 0 and self.cell_h > 0:
            cell_area = self.cell_w * self.cell_h
            unique_ids, pixel_counts = np.unique(id_grid_A, return_counts=True)
            for tile_id, p_count in zip(unique_ids, pixel_counts):
                if tile_id in ["UNKNOWN", "P"]: continue
                self.tile_frequencies[tile_id] += (p_count / cell_area)

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

                if core_key in ghost_rules:
                    for d in ["top", "bottom", "left", "right"]:
                        for gn, gcount in ghost_rules[core_key][d].items():
                            real_rules[core_key][d][gn] += gcount
                            
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
                        if core_key in ghost_rules:
                            for gn, gcount in ghost_rules[core_key][d].items():
                                if gn == "B":
                                    continue
                                real_rules[clone_id][d][gn] += gcount

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
        self.placed_instances = placed_instances_A
        self.level_h = h
        self.level_w = w

    def _extract_neighbors_from_grid(self, placed_instances, id_grid, level_h, level_w, target_dict=None):
        if target_dict is None:
            target_dict = self.adjacency_rules
            
        # Calculate 60% thickness based on the largest dimension
        thickness_threshold = max(1, int(max(self.cell_w, self.cell_h) * 0.60))

        # The raycast only goes as deep as the longest check requires
        TOLERANCE = max(6, thickness_threshold) 
        EDGE_INSET = 2 
        
        def get_smart_neighbors(r_start, r_end, c_start, c_end, direction, origin_id):
            r0 = max(0, min(r_start, level_h))
            r1 = max(0, min(r_end, level_h))
            c0 = max(0, min(c_start, level_w))
            c1 = max(0, min(c_end, level_w))
            
            if r0 >= r1 or c0 >= c1:
                return []

            slice_values = id_grid[r0:r1, c0:c1]
            
            if "UNKNOWN" in slice_values:
                return []

            if direction == "left":
                layers = [slice_values[:, i] for i in range(slice_values.shape[1]-1, -1, -1)]
            elif direction == "right":
                layers = [slice_values[:, i] for i in range(slice_values.shape[1])]
            elif direction == "top":
                layers = [slice_values[i, :] for i in range(slice_values.shape[0]-1, -1, -1)]
            elif direction == "bottom":
                layers = [slice_values[i, :] for i in range(slice_values.shape[0])]
                
            found_neighbors = set()
            consecutive_b_layers = 0
            self_tile_thickness = 0 # Add this tracker
            
            for layer in layers:
                unique_in_layer = np.unique(layer)
                non_b_p = [v for v in unique_in_layer if v not in ["B", "P"]]
                
                if len(non_b_p) > 0:
                    consecutive_b_layers = 0 
                    
                    # Verify if it's just an overlap glitch of itself
                    if len(non_b_p) == 1 and non_b_p[0] == origin_id:
                        self_tile_thickness += 1
                        if self_tile_thickness >= thickness_threshold:
                            found_neighbors.add(origin_id)
                            return list(found_neighbors)
                        continue 
                    else:
                        valid_tiles = [v for v in non_b_p if v != origin_id]
                        if valid_tiles:
                            found_neighbors.update(valid_tiles)
                            return list(found_neighbors)
                        
                elif "B" in unique_in_layer:
                    consecutive_b_layers += 1
                    if consecutive_b_layers == 6:
                        found_neighbors.add("B")
                        return list(found_neighbors)
                        
            if consecutive_b_layers > 0 and len(found_neighbors) == 0:
                found_neighbors.add("B")
                
            return list(found_neighbors)
        

        for name, x, y, w, h in placed_instances:
            inset_x = min(w // 3, EDGE_INSET)
            inset_y = min(h // 3, EDGE_INSET)
            
            if y <= TOLERANCE:
                target_dict[name]["top"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y - TOLERANCE, y, 
                    x + inset_x, x + w - inset_x, "top", name 
                )
                target_dict[name]["top"].update(neighbors)

            if y + h >= level_h - TOLERANCE:
                target_dict[name]["bottom"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + h, y + h + TOLERANCE, 
                    x + inset_x, x + w - inset_x, "bottom", name
                )
                target_dict[name]["bottom"].update(neighbors)

            if x <= TOLERANCE:
                target_dict[name]["left"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x - TOLERANCE, x, "left", name
                )
                target_dict[name]["left"].update(neighbors)

            if x + w >= level_w - TOLERANCE:
                target_dict[name]["right"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x + w, x + w + TOLERANCE, "right", name
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

        for rule_key in self.adjacency_rules.keys():
            if rule_key in ["B", "P", "UNKNOWN"]: continue

            match = clone_pattern.match(rule_key)
            if match:
                required_base_files.add(f"{match.group(2)}.png")
            else:
                required_base_files.add(rule_key)

        for req_file in required_base_files:
            if req_file in self.template_cache:
                filepath = self.template_cache[req_file]['filepath']
                img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cv2.imwrite(str(out_dir / req_file), img)

        for name, img in self.dynamic_templates.items():
            if name in required_base_files:
                cv2.imwrite(str(out_dir / name), img)


    def _save_pruned_reconstruction(self, level_img_original, level_stem, out_dir):
        blank_grid = np.zeros((self.level_h, self.level_w, 4), dtype=np.uint8)
        
        for instance_id, x, y, w, h in self.placed_instances:
            if instance_id in self.adjacency_rules:
                if instance_id in self.dynamic_templates:
                    template_bgra = self.dynamic_templates[instance_id]
                elif instance_id in self.template_cache:
                    template_bgra = self.template_cache[instance_id]['img_bgra']
                else:
                    match = re.match(r"(.+?)_(tile_.+)_y\d+_x\d+", instance_id)
                    if match:
                        core_file = f"{match.group(2)}.png"
                        if core_file in self.template_cache:
                            template_bgra = self.template_cache[core_file]['img_bgra']
                        else:
                            continue
                    else:
                        continue
                        
                start_x = max(0, x)
                start_y = max(0, y)
                end_x = min(self.level_w, x + w)
                end_y = min(self.level_h, y + h)

                vis_w = end_x - start_x
                vis_h = end_y - start_y

                if vis_w <= 0 or vis_h <= 0:
                    continue

                mask_start_x = 0 if x < 0 else 0
                mask_start_y = 0 if y < 0 else 0
                
                if x < 0: mask_start_x = -x
                if y < 0: mask_start_y = -y
                    
                mask_end_x = mask_start_x + vis_w
                mask_end_y = mask_start_y + vis_h

                t_bgra_clipped = template_bgra[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
                alpha = t_bgra_clipped[:, :, 3] > 0

                roi = blank_grid[start_y:end_y, start_x:end_x]
                roi[alpha] = t_bgra_clipped[alpha]
                blank_grid[start_y:end_y, start_x:end_x] = roi
                    
        output_file_path = out_dir / f"{level_stem}_reconstructed_pruned.png"
        cv2.imwrite(str(output_file_path), blank_grid)


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
        
        for level_name, level_data in tqdm(self.source_image_cache.items(), desc="Processing Levels"):
            level_stem = Path(level_name).stem
            level_img_bgra = level_data['bgra']
            
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
            
            self._reconstruct_single_level_complex(level_name, level_img_bgra, level_stem)
            
            level_output_dir = out_base / f"{level_stem}_data"
            level_output_dir.mkdir(parents=True, exist_ok=True)

            self._enforce_rule_symmetry()
            self._save_pruned_reconstruction(level_img_bgra, level_stem, out_base)   
            self.save_adjacency_rules(level_output_dir / "adjacency_rules.txt")
            self.save_frequencies(level_output_dir / "ratios.json")
            self.save_used_tiles(level_output_dir / "tiles")


if __name__ == "__main__":
    base_dir = "output_segments/1 corePerLevel Final"
    
    reconstructor = LevelReconstructor(
        segment_path=base_dir,
        level_path="demo/imgs/test",
        output_path=base_dir,
        match_threshold=0.08,
        use_complex_tiles=True 
    )

    reconstructor.run_reconstruction(
        base_output_dir="Generation/1 corePerLevel Final"
    )