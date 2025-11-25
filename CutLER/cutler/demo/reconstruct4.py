import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

class LevelReconstructor:
    """
    Reconstructs level images from a folder of segments using
    masked template matching, preserving transparency and
    allowing for overlapping segments.
    """
    
    def __init__(self, segment_path, level_path, output_path, match_threshold=1000):
        self.segments_base_path = Path(segment_path)
        self.level_path = Path(level_path)
        self.output_path = Path(output_path)
        self.MATCH_THRESHOLD = match_threshold

        # Caches to hold image data
        # This will hold {name: {img_bgr, img_bgra, mask, w, h, is_valid, area}}
        self.template_cache = {} 
        self.source_image_cache = {} # Will hold {name: img}
        
        # A set to track which templates are never used
        self.unmatched_templates = set()

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _load_source_levels(self):
        """
        Loads all source level images into the cache (as 3-channel BGR).
        """
        pattern = '*.png'
        image_files = list(self.level_path.glob(pattern))
        
        if not image_files:
            print(f"Warning: No PNG files found in {self.level_path} with pattern '{pattern}'", file=sys.stderr)
            return

        print(f"\nLoading {len(image_files)} source levels from {self.level_path}...")
        pbar = tqdm(image_files, desc="Loading Source Levels", unit="img")
        for img_path in pbar:
            # Source images are loaded as BGR (3-channel) for matching
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) 
            if img is not None:
                self.source_image_cache[img_path.name] = img
            else:
                print(f"Warning: Failed to load image {img_path}", file=sys.stderr)

    def _load_all_templates(self):
        """
        Loads and processes ALL templates from all subfolders,
        preserving transparency.
        """
        image_files = list(self.segments_base_path.glob('**/*.png'))
        
        if not image_files:
            print(f"Warning: No templates found in {self.segments_base_path}", file=sys.stderr)
            return
        
        print(f"\nLoading {len(image_files)} templates...")
        pbar = tqdm(image_files, desc="Loading Templates", unit="img")
        for img_path in pbar:
            # --- ⭐️ TRANSPARENCY FIX 1: Load with alpha channel ---
            img_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            
            if img_bgra is None:
                print(f"Warning: Failed to load template {img_path}", file=sys.stderr)
                continue
            
            # --- ⭐️ TRANSPARENCY FIX 2: Process 4-channel image ---
            if img_bgra.shape[2] == 4:
                # Use the 4th channel (alpha) as the mask
                mask = img_bgra[:, :, 3] 
                # Create the 3-channel BGR version for matching
                img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            else:
                # Fallback for 3-channel templates (e.g., old files)
                print(f"Warning: Template {img_path.name} is not 4-channel. Creating mask from threshold.", file=sys.stderr)
                img_bgr = img_bgra
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _ , mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                # Create a 4-channel version for pasting
                img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

            # Check validity
            is_valid = cv2.countNonZero(mask) > 0
            h, w = img_bgr.shape[:2] # Use BGR shape for matching dimensions

            # Store all computed data in the cache
            template_key = f"{img_path.parent.name}/{img_path.name}" # e.g., "mario-1-1/mario-1-1_seg_0.png"
            self.template_cache[template_key] = {
                'img_bgr': img_bgr,     # 3-ch for matching
                'img_bgra': img_bgra,   # 4-ch for pasting
                'mask': mask,         # 1-ch alpha mask
                'w': w,
                'h': h,
                'is_valid': is_valid,
                'area': w * h
            }
        
        # Initialize the set of all loaded templates (will remove as they are matched)
        self.unmatched_templates = set(self.template_cache.keys())

    def _reconstruct_single_level(self, level_name, level_img_original, level_stem):
        """
        Performs the "find all, paste all" reconstruction for one level.
        """
        
        # --- ⭐️ TRANSPARENCY FIX 3: Create 4-channel (BGRA) canvas ---
        h, w, _ = level_img_original.shape
        blank_grid = np.zeros((h, w, 4), dtype=np.uint8) # 4 channels for BGRA

        # Filter templates that belong to this level
        level_templates = {
            k: v for k, v in self.template_cache.items() 
            if k.startswith(level_stem)
        }
        
        if not level_templates:
            print(f"No templates found for {level_stem}, saving blank image.", file=sys.stderr)
            return # Will save the blank grid at the end

        # Sort templates by area (largest first) to handle z-layering
        sorted_templates = sorted(
            level_templates.items(), 
            key=lambda item: item[1]['area'], 
            reverse=True
        )

        # Iterate through each template (largest to smallest)
        for template_key, template_data in sorted_templates:
            
            if not template_data['is_valid']:
                continue
                
            t_h, t_w = template_data['h'], template_data['w']
            if t_h > h or t_w > w:
                continue

            template_bgr = template_data['img_bgr']
            template_bgra = template_data['img_bgra'] # 4-channel image for pasting
            mask = template_data['mask']
            
            # --- ⭐️ OVERLAP FIX: Search on ORIGINAL image, not a copy ---
            try:
                # Search on the original, unmodified level image
                result = cv2.matchTemplate(level_img_original, template_bgr, cv2.TM_SQDIFF, mask=mask)
                
                # Find ALL locations below the threshold
                locations = np.where(result <= self.MATCH_THRESHOLD)
                
                if locations[0].size > 0:
                    # If we found at least one match, mark this template as "used"
                    if template_key in self.unmatched_templates:
                        self.unmatched_templates.remove(template_key)
                
                # Iterate over all found locations and paste
                for pt in zip(*locations[::-1]): # pt is (x, y)
                    # --- ⭐️ TRANSPARENCY FIX 4: Alpha-aware paste ---
                    
                    # Get the 4-channel region of interest from the canvas
                    roi_grid = blank_grid[pt[1]:pt[1] + t_h, pt[0]:pt[0] + t_w]
                    
                    # Create a boolean mask from the template's alpha
                    # (i.e., "where is this template not fully transparent?")
                    alpha_mask = template_bgra[:, :, 3] > 0
                    
                    # Where the mask is true, copy the template's pixels (BGRA)
                    # This overwrites anything underneath (e.g., pipe overwrites floor)
                    roi_grid[alpha_mask] = template_bgra[alpha_mask]
                    
                    # Place the modified ROI back onto the canvas
                    blank_grid[pt[1]:pt[1] + t_h, pt[0]:pt[0] + t_w] = roi_grid
                    
            except cv2.error:
                # Error (e.g., template was larger than image, though we checked)
                continue

        # --- ⭐️ TRANSPARENCY FIX 5: Save the 4-channel grid ---
        output_file_path = self.output_path / f"{level_stem}_reconstructed.png"
        cv2.imwrite(str(output_file_path), blank_grid)

    def _save_unmatched_segments(self, unmatched_path):
        """
        Saves all templates that were never matched to any level.
        """
        unmatched_dir = Path(unmatched_path)
        unmatched_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(self.unmatched_templates)} unmatched segments to {unmatched_dir}...")
        pbar_save = tqdm(self.unmatched_templates, desc="Saving Unmatched", unit="img")
        
        for template_key in pbar_save:
            # --- ⭐️ TRANSPARENCY FIX 6: Save the original 4-channel image ---
            template_img = self.template_cache[template_key].get('img_bgra')
            
            if template_img is not None:
                # Use template_key (e.g., "mario-1-1/mario-1-1_seg_0.png")
                # and create subdirectories as needed
                output_path = unmatched_dir / template_key
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), template_img)
            else:
                print(f"Warning: Could not find image data for {template_key} in cache.", file=sys.stderr)

    def run_reconstruction(self, unmatched_output_path=None):
        """
        Public method to run the entire reconstruction process.
        """
        # Load all templates *first* to populate the cache
        self._load_all_templates()
        
        # Load all source images
        self._load_source_levels()

        if not self.source_image_cache:
            print("Error: Could not load source levels. Exiting.", file=sys.stderr)
            return
            
        if not self.template_cache:
            print("Error: Could not load templates. Exiting.", file=sys.stderr)
            return

        print("\n--- Starting Level Reconstruction ---")
        pbar_levels = tqdm(self.source_image_cache.items(), desc="Reconstructing Levels", unit="level")

        for level_name, level_img in pbar_levels:
            level_stem = Path(level_name).stem
            pbar_levels.set_postfix_str(f"Building {level_stem}...")

            # Reconstruct the single level
            self._reconstruct_single_level(level_name, level_img, level_stem)

        print("\n--- Reconstruction Finished ---")

        if unmatched_output_path:
            self._save_unmatched_segments(unmatched_output_path)
            
        print("\n--- Script Finished ---")


# --- This makes the script runnable ---
if __name__ == "__main__":
    
    # --- Configuration ---
    SEGMENTS_BASE_PATH = Path("output_segments/core3Remake")
    ORIGINAL_PATH = Path("demo/imgs/test")
    RECONSTRUCTION_PATH = Path("output_segments/core3Recon")
    UNMATCHED_PATH = Path("output_segments/unmatched")
    MATCH_THRESHOLD = 1000 # Error threshold for TM_SQDIFF

    # 1. Create an instance of the class
    reconstructor = LevelReconstructor(
        segment_path=SEGMENTS_BASE_PATH,
        level_path=ORIGINAL_PATH,
        output_path=RECONSTRUCTION_PATH,
        match_threshold=MATCH_THRESHOLD
    )
    
    # 2. Run the process
    reconstructor.run_reconstruction(
        unmatched_output_path=UNMATCHED_PATH
    )