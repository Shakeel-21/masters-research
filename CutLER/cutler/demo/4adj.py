import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
import shutil
from collections import defaultdict, Counter

class LevelReconstructor:
    """
    Reconstructs level images using ONLY segments found in a folder matching the level's name.
    """
    
    def __init__(self, segment_path, level_path, output_path, match_threshold=80000):
        self.segments_base_path = Path(segment_path)
        self.level_path = Path(level_path)
        self.output_path = Path(output_path)
        self.MATCH_THRESHOLD = match_threshold

        self.template_cache = {} 
        self.source_image_cache = {}
        self.unmatched_templates = set()
        self.used_templates = set() # NEW: Tracks successfully placed tiles

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
        """
        Loads templates ONLY from the specific folder: segments_base_path / level_stem
        """
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
                'filepath': img_path # Store original path for copying later
            }
        return templates

    def _reconstruct_single_level(self, level_name, level_img_original, level_stem):
        h, w, _ = level_img_original.shape
        blank_grid = np.zeros((h, w, 4), dtype=np.uint8)
        id_grid = np.full((h, w), "B", dtype=object)
        visited_mask = np.zeros((h, w), dtype=bool)
        
        # NEW: Matrix to track how many times a solid pixel tries to be placed
        overlap_heatmap = np.zeros((h, w), dtype=np.uint16)

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            level_img_for_matching = cv2.cvtColor(level_img_original, cv2.COLOR_BGRA2BGR)
        else:
            level_img_for_matching = level_img_original

        level_templates = {k: v for k, v in self.template_cache.items()} 
        placed_instances = []

        # Sort templates large to small
        sorted_templates = sorted(level_templates.items(), key=lambda item: item[1]['area'], reverse=True)

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
                        
                        # Shrink the collision mask
                        alpha_uint8 = alpha_mask.astype(np.uint8) * 255
                        shrunk_mask_uint8 = cv2.erode(alpha_uint8, kernel, iterations=1)
                        shrunk_mask = shrunk_mask_uint8 > 127
                        
                        # Safeguard: If the tile is so thin that erosion destroyed it, use the original mask
                        if not np.any(shrunk_mask):
                            shrunk_mask = alpha_mask

                        # Check for collisions ONLY using the shrunk core mask
                        roi_visited = visited_mask[y:y + t_h, x:x + t_w]
                        if np.any(roi_visited[shrunk_mask]):
                            continue

                        overlap_heatmap[y:y + t_h, x:x + t_w][alpha_mask] += 1
                        # Apply the tile
                        roi_grid = blank_grid[y:y + t_h, x:x + t_w]
                        roi_grid[alpha_mask] = template_data['img_bgra'][alpha_mask]
                        blank_grid[y:y + t_h, x:x + t_w] = roi_grid
                        
                        roi_ids = id_grid[y:y + t_h, x:x + t_w]
                        roi_ids[alpha_mask] = template_key
                        id_grid[y:y + t_h, x:x + t_w] = roi_ids

                        visited_mask[y:y + t_h, x:x + t_w][alpha_mask] = True
                        placed_instances.append((template_key, x, y, t_w, t_h))
                        
                        # Mark tile as used for our minimal set extraction
                        self.used_templates.add(template_data['filepath'])
                        
                except cv2.error:
                    continue

        # Generate output outputs
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

        # Output original reconstruction
        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid)

        # NEW: Output Overlap Visualization (Red Overlay)
        overlap_overlay = blank_grid.copy()
        red_mask = overlap_heatmap > 1
        overlap_overlay[red_mask] = [0, 0, 255, 255] # Red with full alpha
        overlap_file_path = self.output_path / f"{level_stem}_overlaps.png"
        cv2.imwrite(str(overlap_file_path), overlap_overlay)
        
        self._extract_neighbors_from_grid(placed_instances, id_grid, h, w)

    def _extract_neighbors_from_grid(self, placed_instances, id_grid, level_h, level_w):
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
            
            # --- TOP NEIGHBORS ---
            if y <= TOLERANCE:
                self.adjacency_rules[name]["top"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y - TOLERANCE, y, 
                    x + inset_x, x + w - inset_x 
                )
                self.adjacency_rules[name]["top"].update(neighbors)

            # --- BOTTOM NEIGHBORS ---
            if y + h >= level_h - TOLERANCE:
                self.adjacency_rules[name]["bottom"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + h, y + h + TOLERANCE, 
                    x + inset_x, x + w - inset_x
                )
                self.adjacency_rules[name]["bottom"].update(neighbors)

            # --- LEFT NEIGHBORS ---
            if x <= TOLERANCE:
                self.adjacency_rules[name]["left"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x - TOLERANCE, x
                )
                self.adjacency_rules[name]["left"].update(neighbors)

            # --- RIGHT NEIGHBORS ---
            if x + w >= level_w - TOLERANCE:
                self.adjacency_rules[name]["right"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y,
                    x + w, x + w + TOLERANCE
                )
                self.adjacency_rules[name]["right"].update(neighbors)

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

    # NEW: Function to extract and save only the used segments
    def save_used_tiles(self, output_dir):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving {len(self.used_templates)} unique used tiles to {output_dir}...")
        
        for src_path in self.used_templates:
            # src_path is now directly the Path object we saved earlier
            if src_path and src_path.exists():
                dest_path = out_path / src_path.name
                shutil.copy(src_path, dest_path)

    def run_reconstruction(self, unmatched_output_path=None, rules_output_path=None, freq_output_path=None, used_tiles_output_path=None):
        self._load_source_levels()

        if not self.source_image_cache:
            return

        print("\n--- Starting Level Reconstruction & Analysis ---")
        
        for level_name, level_img in tqdm(self.source_image_cache.items(), desc="Processing Levels"):
            level_stem = Path(level_name).stem
            self.template_cache = self._load_templates_for_level(level_stem)
            
            if not self.template_cache:
                continue

            if self.cell_w == 0:
                valid_t = [t for t in self.template_cache.values() if t['is_valid']]
                if valid_t:
                    self.cell_w = min(t['w'] for t in valid_t)
                    self.cell_h = min(t['h'] for t in valid_t)

            self.unmatched_templates = set(self.template_cache.keys())
            self._reconstruct_single_level(level_name, level_img, level_stem)

        if rules_output_path:
            self.save_adjacency_rules(rules_output_path)

        if freq_output_path:
            self.save_frequencies(freq_output_path)
            
        if used_tiles_output_path:
            Path(used_tiles_output_path).parent.mkdir(parents=True, exist_ok=True)
            self.save_used_tiles(used_tiles_output_path)


if __name__ == "__main__":
    reconstructor = LevelReconstructor(
        segment_path="Generation/mario1t ratios",
        level_path="demo/imgs/test",
        output_path="Generation/mario1t ratios",
        match_threshold=0.05
    )
    
    # Notice the new 'used_tiles_output_path' argument added below
    reconstructor.run_reconstruction(
        used_tiles_output_path="Generation/mario1t ratios/mario_1t_used",
        unmatched_output_path="output_segments/unmatched",
        rules_output_path="Generation/mario1t ratios/mario_1t_used/adjacency_rules.txt",
        freq_output_path="Generation/mario1t ratios/mario_1t_used/ratios.json"
        
    )