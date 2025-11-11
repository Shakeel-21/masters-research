import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

SEGMENT_PATH = Path("output_segments/results/trans")
ORIGINAL_PATH = Path("demo/imgs/test")
BLOCKS_PATH = Path("output_segments/remakeTrans")

MATCH_THRESHOLD = 1000

def get_source_name(template_name: str) -> str:
    """
    Extracts the source image name from the template filename.
    e.g., 'mario-6-2_seg_8.png' -> 'mario-6-2'
    """
    if "_seg_" not in template_name:
        print(f"Warning: Template '{template_name}' may have an invalid name format.", file=sys.stderr)
        return template_name.split('.')[0] # Best guess
        
    return template_name.split("_seg_")[0]

# --- MODIFIED FUNCTION ---
def load_images(base_path: Path, recursive: bool = False, description: str = "Loading", read_flag: int = cv2.IMREAD_COLOR):
    """
    Loads all images from a directory into a dictionary {filename: image_object}.
    
    Args:
        read_flag: The OpenCV flag to use for reading images (e.g.,
                   cv2.IMREAD_COLOR or cv2.IMREAD_UNCHANGED).
    """
    pattern = '**/*.png' if recursive else '*.png'
    image_files = list(base_path.glob(pattern))
    image_cache = {}
    
    if not image_files:
        print(f"Warning: No PNG files found in {base_path} with pattern '{pattern}'", file=sys.stderr)
        return image_cache

    print(f"\nLoading {len(image_files)} images from {base_path}...")
    pbar = tqdm(image_files, desc=description, unit="img")
    for img_path in pbar:
        # --- Use the specified read_flag ---
        img = cv2.imread(str(img_path), read_flag)
        
        if img is not None:
            image_cache[img_path.name] = img
        else:
            print(f"Warning: Failed to load image {img_path}", file=sys.stderr)
            
    return image_cache

def main():
    # --- MODIFIED CALLS ---
    # Load templates WITH alpha channel (4-channel)
    template_cache = load_images(
        SEGMENT_PATH, 
        recursive=True, 
        description="Loading Templates", 
        read_flag=cv2.IMREAD_UNCHANGED
    )
    # Load source images WITHOUT alpha channel (3-channel)
    source_image_cache = load_images(
        ORIGINAL_PATH, 
        recursive=False, 
        description="Loading Source Images", 
        read_flag=cv2.IMREAD_COLOR
    )
    # --- END MODIFIED CALLS ---

    level_match_report = {level_name: [] for level_name in source_image_cache.keys()}

    pbar_levels = tqdm(source_image_cache.items(), desc="Recreate level", unit="level")

    for level_name, level_img in pbar_levels:
        pbar_levels.set_postfix_str(f"Scanning {level_name}...")
        
        # Renamed for clarity
        for template_name, template_img_bgra in template_cache.items():
            
            # --- START: TRANSPARENCY FIX ---
            
            # 1. Skip if template isn't 4-channel (BGRA)
            if template_img_bgra.shape[2] != 4:
                print(f"Warning: Template {template_name} is not 4-channel, skipping.", file=sys.stderr)
                continue
                
            # 2. Extract the alpha channel (index 3) to use as the mask
            mask = template_img_bgra[:, :, 3]

            # 3. Get the 3-channel BGR part for the template matching
            template_bgr = cv2.cvtColor(template_img_bgra, cv2.COLOR_BGRA2BGR)
            
            # 4. Check if mask is all-zero (fully transparent)
            if cv2.countNonZero(mask) == 0:
                continue # No visible pixels to match

            # --- END: TRANSPARENCY FIX ---
            
            try:
                # 5. Match using the 3-channel template and its *real* alpha mask
                result = cv2.matchTemplate(level_img, template_bgr, cv2.TM_SQDIFF, mask=mask)
                min_val, _max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
                
                # 6. Check threshold
                if min_val <= MATCH_THRESHOLD:
                    level_match_report[level_name].append(template_name)
                    
            except cv2.error as e:
                # print(f"OpenCV error matching {template_name}: {e}", file=sys.stderr)
                continue

    print("\n--- Level Match Report ---")
    for level_name, templates in level_match_report.items():
        if templates:
            level_folder_name = Path(level_name).stem
            level_output_dir = BLOCKS_PATH / level_folder_name
            level_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✅ Level: {level_name} ({len(templates)} segments found)")
            
            for i, tpl_name in enumerate(templates):
                print(f"   {i+1}. {tpl_name}")
                
                # --- START: SAVE FIX ---
                # 7. Get the original 4-CHANNEL (BGRA) image from the cache
                template_img_to_save = template_cache.get(tpl_name)
                
                if template_img_to_save is not None:
                    output_file_path = level_output_dir / tpl_name
                    # 8. Save the 4-channel image, preserving transparency
                    cv2.imwrite(str(output_file_path), template_img_to_save)
                # --- END: SAVE FIX ---
        else:
            print(f"\n❌ Level: {level_name} (0 segments found)")

if __name__ == "__main__":
    main()