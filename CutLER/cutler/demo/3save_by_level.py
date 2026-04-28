import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

SEGMENT_PATH = Path("output_segments/core3")
ORIGINAL_PATH = Path("demo/imgs/test")
BLOCKS_PATH = Path("output_segments/TestNorm05")

MATCH_THRESHOLD = 0.05
# NMS_THRESHOLD: Controls how much overlap is allowed. 
# 0.3 means if two boxes overlap by more than 30%, the one with the lower score is deleted.
NMS_OVERLAP_THRESHOLD = 0.30

def get_source_name(template_name: str) -> str:
    if "_seg_" not in template_name:
        # print(f"Warning: Template '{template_name}' may have an invalid name format.", file=sys.stderr)
        return template_name.split('.')[0] 
    return template_name.split("_seg_")[0]

def load_images(base_path: Path, recursive: bool = False, description: str = "Loading", read_flag: int = cv2.IMREAD_COLOR):
    pattern = '**/*.png' if recursive else '*.png'
    image_files = list(base_path.glob(pattern))
    image_cache = {}
    
    if not image_files:
        print(f"Warning: No PNG files found in {base_path} with pattern '{pattern}'", file=sys.stderr)
        return image_cache

    pbar = tqdm(image_files, desc=description, unit="img")
    for img_path in pbar:
        img = cv2.imread(str(img_path), read_flag)
        if img is not None:
            image_cache[img_path.name] = img
    return image_cache

def main():
    template_cache = load_images(SEGMENT_PATH, recursive=True, description="Loading Templates", read_flag=cv2.IMREAD_UNCHANGED)
    source_image_cache = load_images(ORIGINAL_PATH, recursive=False, description="Loading Source Images", read_flag=cv2.IMREAD_COLOR)

    level_match_report = {level_name: [] for level_name in source_image_cache.keys()}

    pbar_levels = tqdm(source_image_cache.items(), desc="Recreate level", unit="level")

    for level_name, level_img in pbar_levels:
        pbar_levels.set_postfix_str(f"Scanning {level_name}...")
        
        # --- CHANGE 1: Create a list to hold ALL potential matches for this level ---
        # We cannot save them yet. We must let them "fight" first.
        candidates = [] 
        
        for template_name, template_img_bgra in template_cache.items():
            
            # --- Transparency Setup (Same as before) ---
            if template_img_bgra.shape[2] != 4: continue
            mask = template_img_bgra[:, :, 3]
            template_bgr = cv2.cvtColor(template_img_bgra, cv2.COLOR_BGRA2BGR)
            if cv2.countNonZero(mask) == 0: continue
            
            try:
                # Match
                result = cv2.matchTemplate(level_img, template_bgr, cv2.TM_SQDIFF_NORMED, mask=mask)
                min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
                
                # Check threshold
                if min_val <= MATCH_THRESHOLD:
                    
                    # --- THE FIX FOR "FULL BLOCK vs MIDDLE PART" ---
                    
                    # A. Calculate Base Quality (0.0 to 1.0)
                    # 1.0 is a perfect pixel match, 0.0 is barely passing the threshold
                    match_quality = 1.0 - min_val 
                    
                    # B. Get the Area (Size) of the template
                    h, w = template_bgr.shape[:2]
                    area = h * w
                    
                    # C. Calculate Final Score: Quality * Area
                    # This ensures a 16x16 block always beats an 8x8 block
                    # even if the 8x8 block has a slightly "cleaner" pixel match.
                    weighted_score = match_quality * area
                    
                    # Store candidate with the NEW weighted_score
                    candidates.append({
                        "box": [min_loc[0], min_loc[1], w, h],
                        "score": weighted_score, 
                        "name": template_name
                    })
                    
            except cv2.error as e:
                continue

        # --- CHANGE 3: Apply Non-Maximum Suppression (NMS) ---
        if candidates:
            # Extract lists for OpenCV NMS function
            boxes = [c["box"] for c in candidates]
            scores = [c["score"] for c in candidates]
            
            # Run NMS
            # score_threshold=0 because we already filtered by MATCH_THRESHOLD manually above
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=NMS_OVERLAP_THRESHOLD)
            
            # indices returns a list of integers representing the winners
            if len(indices) > 0:
                for i in indices.flatten():
                    winner = candidates[i]
                    level_match_report[level_name].append(winner["name"])

    # --- Report and Save (Mostly unchanged, just iterates the winners) ---
    print("\n--- Level Match Report (After NMS Cleaning) ---")
    for level_name, templates in level_match_report.items():
        if templates:
            level_folder_name = Path(level_name).stem
            level_output_dir = BLOCKS_PATH / level_folder_name
            level_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✅ Level: {level_name} ({len(templates)} segments saved)")
            
            for i, tpl_name in enumerate(templates):
                print(f"   {i+1}. {tpl_name}")
                template_img_to_save = template_cache.get(tpl_name)
                if template_img_to_save is not None:
                    output_file_path = level_output_dir / tpl_name
                    cv2.imwrite(str(output_file_path), template_img_to_save)
        else:
            print(f"\n❌ Level: {level_name} (0 segments found)")

if __name__ == "__main__":
    main()