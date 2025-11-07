import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

SEGMENT_PATH = Path("output_segments/results/point1_15")
ORIGINAL_PATH = Path("demo/imgs/all_levels")
UNMATCHED_PATH = Path("output_segments/unmatched") # <-- Destination for unmatched images

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


def load_images(base_path: Path, recursive: bool = False, description: str = "Loading"):
    """
    Loads all images from a directory into a dictionary {filename: image_object}.
    
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
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            # Note: This assumes all filenames (img_path.name) are unique,
            # even if in different subfolders, due to recursive=True.
            image_cache[img_path.name] = img
        else:
            print(f"Warning: Failed to load image {img_path}", file=sys.stderr)
            
    return image_cache


def main():
    template_cache = load_images(SEGMENT_PATH, recursive=True, description="Loading Templates")
    source_image_cache = load_images(ORIGINAL_PATH, recursive=False, description="Loading Source Images")

    if not template_cache or not source_image_cache:
        print("Error: Could not load templates or source images. Exiting.", file=sys.stderr)
        return

    # --- GOAL 1: Match templates to their specific original image ---
    print("\n--- Starting Goal 1: Specific Source Matching ---")
    
    specific_failures = []
    global_match_status = {tpl_name: False for tpl_name in template_cache.keys()}

    pbar_goal1 = tqdm(template_cache.items(), desc="Goal 1: Specific Match", unit="template")
    
    for template_name, template_img in pbar_goal1:
        source_name_base = get_source_name(template_name)
        source_image_name = f"{source_name_base}.png"
        source_img = source_image_cache.get(source_image_name)
        
        is_match = False
        
        if source_img is None:
            # Source image not found in cache
            pass # Failure is handled by is_match=False
        elif template_img.shape[0] > source_img.shape[0] or template_img.shape[1] > source_img.shape[1]:
            # Template is bigger than the image, cannot match
            print(f"Warning: Template {template_name} is larger than its source {source_image_name}", file=sys.stderr)
        else:
            # Perform the template matching
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            _ , mask = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY)

            result = cv2.matchTemplate(source_img, template_img, cv2.TM_SQDIFF, mask=mask)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            
            if _min_val <= MATCH_THRESHOLD:
                is_match = True
                global_match_status[template_name] = True # Mark as matched for Goal 2
        
        if not is_match:
            specific_failures.append(template_name)

    # --- GOAL 1: Results ---
    print("\n--- Goal 1 Results ---")
    print(f"Total templates processed: {len(template_cache)}")
    print(f"Templates that FAILED to match their original source: {len(specific_failures)}")
    if specific_failures:
        print("Failed templates:")
        for i, failure in enumerate(specific_failures):
            print(f"  {i+1}. {failure}")

    # --- GOAL 2: Find templates that match NO image at all ---
    print("\n--- Starting Goal 2: Global Matching (for unmatched templates) ---")
    
    
    # Check only templates that haven't already been matched in Goal 1
    templates_to_check_globally = [
        tpl_name for tpl_name, matched in global_match_status.items() 
        if not matched
    ]
    
    if not templates_to_check_globally:
        print("All templates found a match with their specific source. Skipping global check.")
    else:
        print(f"Checking {len(templates_to_check_globally)} templates against all {len(source_image_cache)} source images...")

    pbar_goal2 = tqdm(templates_to_check_globally, desc="Goal 2: Global Match", unit="template")
    
    for template_name in pbar_goal2:
        template_img = template_cache[template_name]
        
        # We already know it didn't match its *specific* source,
        # so we check all other sources.
        for image_name, source_img in source_image_cache.items():
            # Skip if template is larger than image
            if template_img.shape[0] > source_img.shape[0] or template_img.shape[1] > source_img.shape[1]:
                continue
                
            # Perform matching
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            _ , mask = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY)

            result = cv2.matchTemplate(source_img, template_img, cv2.TM_SQDIFF, mask=mask)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            
            if _min_val <= MATCH_THRESHOLD:
                global_match_status[template_name] = True # Mark as matched
                break # Found a match, no need to check other images
    
    # --- GOAL 2: Results ---
    
    # ***MODIFICATION***: Added 'if not matched' to correctly filter failures.
    global_failures = [
        tpl_name for tpl_name, matched in global_match_status.items()
        if not matched
    ]
    
    print("\n--- Goal 2 Results ---")
    print(f"Total templates that failed to match ANY image: {len(global_failures)}")
    if global_failures:
        print("Unmatched templates:")
        for i, failure in enumerate(global_failures):
            print(f"  {i+1}. {failure}")

    # --- ***NEW SECTION***: Save unmatched segments ---
    if global_failures:
        print("\n--- Starting: Saving Unmatched Segments ---")
        UNMATCHED_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(global_failures)} unmatched images to {UNMATCHED_PATH}...")
        
        pbar_save = tqdm(global_failures, desc="Saving Unmatched", unit="img")
        for template_name in pbar_save:
            # Get the image data from the cache
            template_img = template_cache.get(template_name)
            
            if template_img is not None:
                # Construct the full output path
                output_path = UNMATCHED_PATH / template_name
                
                # Save the image
                # Note: This will save all images flat in the UNMATCHED_PATH.
                # If your segments have duplicate names in subfolders,
                # you may need to modify load_images to store relative paths.
                cv2.imwrite(str(output_path), template_img)
            else:
                print(f"Warning: Could not find image data for {template_name} in cache.", file=sys.stderr)

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()