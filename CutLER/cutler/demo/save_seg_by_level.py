import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

SEGMENT_PATH = Path("output_segments/results/point1_15")
ORIGINAL_PATH = Path("demo/imgs/all_levels")
BLOCKS_PATH = Path("output_segments/remake")

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

    level_match_report = {level_name: [] for level_name in source_image_cache.keys()}

    pbar_levels = tqdm(source_image_cache.items(), desc="Recreate level", unit="level")

    for level_name, level_img in pbar_levels:
        pbar_levels.set_postfix_str(f"Scanning {level_name}...")
        for template_name, template_img in template_cache.items():
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            _ , mask = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY)
            try:
                result = cv2.matchTemplate(level_img, template_img, cv2.TM_SQDIFF, mask=mask)
                min_val, _max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
                
                # 5. Check threshold
                if min_val <= MATCH_THRESHOLD:
                    
                    level_match_report[level_name].append(template_name)
            except cv2.error:
                # Skip if OpenCV fails for any reason
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
                template_img = template_cache.get(tpl_name)
                output_file_path = level_output_dir / tpl_name
                cv2.imwrite(str(output_file_path), template_img)
        else:
            print(f"\n❌ Level: {level_name} (0 segments found)")

if __name__ == "__main__":
    main()