import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
from collections import defaultdict, Counter

class LevelReconstructor:
    """
    Reconstructs level images from a folder of segments using
    masked template matching and extracts WFC adjacency rules.
    """
    
    def __init__(self, segment_path, level_path, output_path, match_threshold=80000):
        self.segments_base_path = Path(segment_path)
        self.level_path = Path(level_path)
        self.output_path = Path(output_path)
        self.MATCH_THRESHOLD = match_threshold

        self.template_cache = {} 
        self.source_image_cache = {}
        self.unmatched_templates = set()

        # --- Global Rule Dictionary ---
        # Structure: {template_name: { "top": [names], "bottom": [names], "left": [names], "right": [names] }}
        self.adjacency_rules = defaultdict(lambda: {"top": Counter(), "bottom": Counter(), "left": Counter(), "right": Counter()})

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _load_source_levels(self):
        pattern = '*.png'
        image_files = list(self.level_path.glob(pattern))
        if not image_files:
            print(f"Warning: No PNG files found in {self.level_path}", file=sys.stderr)
            return
        print(f"\nLoading {len(image_files)} source levels from {self.level_path}...")
        for img_path in tqdm(image_files, desc="Loading Source Levels"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) 
            if img is not None:
                self.source_image_cache[img_path.name] = img

    def _load_all_templates(self):
        image_files = list(self.segments_base_path.glob('**/*.png'))
        if not image_files: return
        
        print(f"\nLoading {len(image_files)} templates...")
        for img_path in tqdm(image_files, desc="Loading Templates"):
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

            # Use simple filename as key, or relative path if you prefer
            template_key = img_path.name 

            self.template_cache[template_key] = {
                'img_bgr': img_bgr,
                'img_bgra': img_bgra,
                'mask': mask,
                'w': w,
                'h': h,
                'is_valid': is_valid,
                'area': w * h
            }
        self.unmatched_templates = set(self.template_cache.keys())

    def _reconstruct_single_level(self, level_name, level_img_original, level_stem):
        h, w, _ = level_img_original.shape
        blank_grid = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Grid to store names (for rules)
        id_grid = np.full((h, w), "B", dtype=object)

        # --- ⭐️ NEW: Occupancy Grid (The Fix) ---
        # Keeps track of pixels we have already claimed.
        visited_mask = np.zeros((h, w), dtype=bool)

        level_templates = {k: v for k, v in self.template_cache.items()} 
        
        placed_instances = []

        # Sort templates large to small
        sorted_templates = sorted(level_templates.items(), key=lambda item: item[1]['area'], reverse=True)

        for template_key, template_data in sorted_templates:
            if not template_data['is_valid']: continue
            
            t_h, t_w = template_data['h'], template_data['w']
            if t_h > h or t_w > w: continue

            try:
                result = cv2.matchTemplate(level_img_original, template_data['img_bgr'], cv2.TM_SQDIFF_NORMED, mask=template_data['mask'])
                locations = np.where(result <= self.MATCH_THRESHOLD)
                
                # --- ⭐️ NEW: Sort Matches by Quality ---
                # We want to process the BEST matches first.
                # Zip the y, x, and score together.
                matches = []
                for y, x in zip(*locations):
                    score = result[y, x]
                    matches.append((x, y, score))
                
                # Sort by score (lowest is best for SQDIFF)
                matches.sort(key=lambda x: x[2])

                if len(matches) > 0 and template_key in self.unmatched_templates:
                    self.unmatched_templates.remove(template_key)

                for x, y, score in matches:
                    
                    # --- ⭐️ NEW: NMS Logic ---
                    # Check the center point of where we want to place this.
                    # If the center is already occupied, skip this placement.
                    # This prevents placing the same enemy 1 pixel away from itself.
                    cy, cx = y + t_h // 2, x + t_w // 2
                    if visited_mask[cy, cx]:
                        continue

                    # --- Visual Reconstruction ---
                    roi_grid = blank_grid[y:y + t_h, x:x + t_w]
                    
                    # --- ⭐️ FIX: Transparency Threshold ---
                    # Increase from 0 to 127 to stop invisible noise erasing floors
                    alpha_mask = template_data['img_bgra'][:, :, 3] > 127
                    
                    roi_grid[alpha_mask] = template_data['img_bgra'][alpha_mask]
                    blank_grid[y:y + t_h, x:x + t_w] = roi_grid
                    
                    # --- Update ID Grid ---
                    roi_ids = id_grid[y:y + t_h, x:x + t_w]
                    roi_ids[alpha_mask] = template_key
                    id_grid[y:y + t_h, x:x + t_w] = roi_ids

                    # --- ⭐️ NEW: Mark Area as Visited ---
                    # Mark the rectangle on our NMS mask so we don't overlap here again
                    visited_mask[y:y + t_h, x:x + t_w] = True

                    placed_instances.append((template_key, x, y, t_w, t_h))
                    
            except cv2.error:
                continue

        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid)

        # --- Analyze Neighbors ---
        self._extract_neighbors_from_grid(placed_instances, id_grid, h, w)

    def _extract_neighbors_from_grid(self, placed_instances, id_grid, level_h, level_w):
        """
        Scans the entire edge of every placed instance to record ALL touching neighbors.
        Robust to irregular sizes and misalignments.
        """
        # Define a helper to safely get a slice from the grid
        def get_slice(r_start, r_end, c_start, c_end):
            # Clip coordinates to image bounds to avoid errors
            r0 = max(0, min(r_start, level_h))
            r1 = max(0, min(r_end, level_h))
            c0 = max(0, min(c_start, level_w))
            c1 = max(0, min(c_end, level_w))
            
            # If our slice is invalid (e.g. completely off-screen), return empty
            if r0 >= r1 or c0 >= c1:
                return []
                
            return id_grid[r0:r1, c0:c1].flatten()

        for name, x, y, w, h in placed_instances:
            
            # --- TOP NEIGHBORS ---
            # Look at the row immediately above the template (y-1)
            # We span the full width (x to x+w)
            if y > 0:
                top_slice = get_slice(y-1, y, x, x+w)
                unique_top = np.unique(top_slice)
                self.adjacency_rules[name]["top"].update(unique_top)
            else:
                self.adjacency_rules[name]["top"].update("P") # Top of screen

            # --- BOTTOM NEIGHBORS ---
            # Look at the row immediately below (y+h)
            if y + h < level_h:
                bottom_slice = get_slice(y+h, y+h+1, x, x+w)
                unique_bottom = np.unique(bottom_slice)
                self.adjacency_rules[name]["bottom"].update(unique_bottom)
            else:
                self.adjacency_rules[name]["bottom"].update("P") # Padding

            # --- LEFT NEIGHBORS ---
            # Look at the column to the left (x-1)
            # Span full height (y to y+h)
            if x > 0:
                left_slice = get_slice(y, y+h, x-1, x)
                unique_left = np.unique(left_slice)
                self.adjacency_rules[name]["left"].update(unique_left)
            else:
                self.adjacency_rules[name]["left"].update("P")

            # --- RIGHT NEIGHBORS ---
            # Look at the column to the right (x+w)
            if x + w < level_w:
                right_slice = get_slice(y, y+h, x+w, x+w+1)
                unique_right = np.unique(right_slice)
                self.adjacency_rules[name]["right"].update(unique_right)
            else:
                self.adjacency_rules[name]["right"].update("P")

    def _get_id_at(self, point, grid, max_h, max_w, direction):
        """
        Safely retrieves the ID from the grid, handling bounds.
        """
        px, py = point
        
        # Check Vertical Bounds
        if py < 0:
            return "B" # Above screen -> Background
        if py >= max_h:
            return "P" # Below screen -> Padding (as requested)
            
        # Check Horizontal Bounds
        if px < 0 or px >= max_w:
            return "P" # Sides -> Padding (or use "B" if preferred)

        return grid[py, px]

    def save_adjacency_rules(self, filepath):
        """
        Saves the learned rules to a text file (JSON format).
        """
        # Convert sets to lists for JSON serialization
        serializable_rules = {
            k: {
                direction: neighbors.most_common() 
                for direction, neighbors in v.items()
            }
            for k, v in self.adjacency_rules.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_rules, f, indent=4)
        print(f"\nAdjacency rules saved to {filepath}")

    def run_reconstruction(self, unmatched_output_path=None, rules_output_path=None):
        self._load_all_templates()
        self._load_source_levels()

        if not self.source_image_cache or not self.template_cache:
            return

        print("\n--- Starting Level Reconstruction & Analysis ---")
        for level_name, level_img in tqdm(self.source_image_cache.items(), desc="Processing Levels"):
            level_stem = Path(level_name).stem
            self._reconstruct_single_level(level_name, level_img, level_stem)

        
        # --- ⭐️ ADJACENCY UPDATE 5: Save Rules ---
        if rules_output_path:
            self.save_adjacency_rules(rules_output_path)


if __name__ == "__main__":
    reconstructor = LevelReconstructor(
        segment_path="output_segments/TestNorm05",
        level_path="demo/imgs/test",
        output_path="output_segments/AdjacencyLearningTestNorm05NMSRecon",
        match_threshold=0.05
    )
    
    reconstructor.run_reconstruction(
        unmatched_output_path="output_segments/unmatched",
        rules_output_path= "output_segments/AdjacencyLearningTestNorm05NMSRecon/adjacency_rules.txt" # <--- Output file
    )