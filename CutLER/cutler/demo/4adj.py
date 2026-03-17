import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
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

        # We no longer load a global cache. This will change per level.
        self.template_cache = {} 
        self.source_image_cache = {}
        self.unmatched_templates = set()

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
        
        # Check if the specific folder exists
        if not specific_folder.exists() or not specific_folder.is_dir():
            return None

        # Look for PNGs inside that specific folder (not recursive)
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

            templates[template_key] = {
                'img_bgr': img_bgr,
                'img_bgra': img_bgra,
                'mask': mask,
                'w': w,
                'h': h,
                'is_valid': is_valid,
                'area': w * h
            }
        return templates

    def _reconstruct_single_level(self, level_name, level_img_original, level_stem):
        h, w, _ = level_img_original.shape
        blank_grid = np.zeros((h, w, 4), dtype=np.uint8)
        id_grid = np.full((h, w), "B", dtype=object)
        visited_mask = np.zeros((h, w), dtype=bool)

        if len(level_img_original.shape) == 3 and level_img_original.shape[2] == 4:
            level_img_for_matching = cv2.cvtColor(level_img_original, cv2.COLOR_BGRA2BGR)
        else:
            level_img_for_matching = level_img_original

        level_templates = {k: v for k, v in self.template_cache.items()} 
        placed_instances = []

        # Sort templates large to small
        sorted_templates = sorted(level_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        for template_key, template_data in sorted_templates:
            if not template_data['is_valid']: continue
            
            t_h, t_w = template_data['h'], template_data['w']
            if t_h > h or t_w > w: continue

            try:
                result = cv2.matchTemplate(level_img_for_matching, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['mask'])
                locations = np.where(result <= self.MATCH_THRESHOLD)
                
                matches = []
                for y, x in zip(*locations):
                    score = result[y, x]
                    matches.append((x, y, score))
                
                matches.sort(key=lambda x: x[2])

                if len(matches) > 0 and template_key in self.unmatched_templates:
                    self.unmatched_templates.remove(template_key)

                for x, y, score in matches:
                    cy, cx = y + t_h // 2, x + t_w // 2
                    if visited_mask[cy, cx]:
                        continue

                    roi_grid = blank_grid[y:y + t_h, x:x + t_w]
                    alpha_mask = template_data['img_bgra'][:, :, 3] > 127
                    
                    roi_grid[alpha_mask] = template_data['img_bgra'][alpha_mask]
                    blank_grid[y:y + t_h, x:x + t_w] = roi_grid
                    
                    roi_ids = id_grid[y:y + t_h, x:x + t_w]
                    roi_ids[alpha_mask] = template_key
                    id_grid[y:y + t_h, x:x + t_w] = roi_ids

                    visited_mask[y:y + t_h, x:x + t_w] = True
                    placed_instances.append((template_key, x, y, t_w, t_h))
                    
            except cv2.error:
                continue

        if level_img_original.shape[2] == 4:
            # If Alpha channel exists, anything with Alpha > 10 is an object
            source_alpha = level_img_original[:, :, 3]
            is_occupied_source = source_alpha > 10
        else:
            # If no alpha (e.g. JPG), assume Top-Left pixel is the background color
            bg_color = level_img_original[0, 0]
            # Calculate difference from bg_color
            diff = cv2.absdiff(level_img_original, bg_color)
            diff_sum = np.sum(diff, axis=2)
            is_occupied_source = diff_sum > 10

            # Identify where Reconstructor thinks it is "B"
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
                
                # Convert raw pixels into WFC grid units 
                # (e.g. 1600 'B' pixels / 256 cell area = 6.25 'B' tiles)
                self.tile_frequencies[tile_id] += (p_count / cell_area)

        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid)
        
        # Analyze neighbors (Rules are still global and accumulate across levels)
        self._extract_neighbors_from_grid(placed_instances, id_grid, h, w)

    def _extract_neighbors_from_grid(self, placed_instances, id_grid, level_h, level_w):
        """
        Scans for neighbors using a tolerance buffer and an EDGE INSET.
        The INSET prevents corner overlaps (like a pipe on the floor above) 
        from being registered as a side neighbor.
        """
        TOLERANCE = 12   # How far out to scan
        EDGE_INSET = 2   # How many pixels to ignore at the corners of the scan
        
        # Helper to get unique IDs, filtering out 'B' if other things exist
        def get_smart_neighbors(r_start, r_end, c_start, c_end):
            # Clip to image bounds
            r0 = max(0, min(r_start, level_h))
            r1 = max(0, min(r_end, level_h))
            c0 = max(0, min(c_start, level_w))
            c1 = max(0, min(c_end, level_w))
            
            if r0 >= r1 or c0 >= c1:
                return []

            # Get all IDs in this tolerance zone
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
            
            # Calculate dynamic insets 
            # (Use min() to ensure we don't cross over if a tile is tiny)
            inset_x = min(w // 3, EDGE_INSET)
            inset_y = min(h // 3, EDGE_INSET)
            
            # --- TOP NEIGHBORS ---
            # Scan X range: padded by inset_x to avoid left/right corner overlaps
            if y <= TOLERANCE:
                self.adjacency_rules[name]["top"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y - TOLERANCE, y, 
                    x + inset_x, x + w - inset_x  # <--- Apply Inset Here
                )
                self.adjacency_rules[name]["top"].update(neighbors)

            # --- BOTTOM NEIGHBORS ---
            # Scan X range: padded by inset_x
            if y + h >= level_h - TOLERANCE:
                self.adjacency_rules[name]["bottom"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + h, y + h + TOLERANCE, 
                    x + inset_x, x + w - inset_x  # <--- Apply Inset Here
                )
                self.adjacency_rules[name]["bottom"].update(neighbors)

            # --- LEFT NEIGHBORS ---
            # Scan Y range: padded by inset_y to avoid top/bottom corner overlaps
            if x <= TOLERANCE:
                self.adjacency_rules[name]["left"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y, # <--- Apply Inset Here
                    x - TOLERANCE, x
                )
                self.adjacency_rules[name]["left"].update(neighbors)

            # --- RIGHT NEIGHBORS ---
            # Scan Y range: padded by inset_y
            if x + w >= level_w - TOLERANCE:
                self.adjacency_rules[name]["right"].update(["P"])
            else:
                neighbors = get_smart_neighbors(
                    y + inset_y, y + h - inset_y, # <--- Apply Inset Here
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
        # Round the floats to make the JSON cleaner
        clean_frequencies = {k: round(v, 2) for k, v in self.tile_frequencies.items()}
        
        with open(filepath, 'w') as f:
            json.dump(clean_frequencies, f, indent=4)
        print(f"Tile frequencies saved to {filepath}")

    def run_reconstruction(self, unmatched_output_path=None, rules_output_path=None, freq_output_path=None):
        self._load_source_levels()

        if not self.source_image_cache:
            return

        print("\n--- Starting Level Reconstruction & Analysis ---")
        
        for level_name, level_img in tqdm(self.source_image_cache.items(), desc="Processing Levels"):
            level_stem = Path(level_name).stem
            
            # 1. Try to load templates specifically for this level
            self.template_cache = self._load_templates_for_level(level_stem)
            
            # 2. Skip if no folder/segments found
            if not self.template_cache:
                continue

            if self.cell_w == 0:
                valid_t = [t for t in self.template_cache.values() if t['is_valid']]
                if valid_t:
                    self.cell_w = min(t['w'] for t in valid_t)
                    self.cell_h = min(t['h'] for t in valid_t)

            # 3. Reset unmatched tracking for this level
            self.unmatched_templates = set(self.template_cache.keys())

            # 4. Reconstruct
            self._reconstruct_single_level(level_name, level_img, level_stem)

        if rules_output_path:
            self.save_adjacency_rules(rules_output_path)

        if freq_output_path:
            self.save_frequencies(freq_output_path)

if __name__ == "__main__":
    reconstructor = LevelReconstructor(
        segment_path="Generation/smb41 ratios",
        level_path="demo/imgs/test",
        output_path="Generation/smb41 ratios",
        match_threshold=0.05
    )
    
    reconstructor.run_reconstruction(
        unmatched_output_path="output_segments/unmatched",
        rules_output_path= "Generation/smb41 ratios/SuperMarioBros2(J)-World4-1/adjacency_rules.txt" ,
        freq_output_path="Generation/smb41 ratios/SuperMarioBros2(J)-World4-1/ratios.json"
    )