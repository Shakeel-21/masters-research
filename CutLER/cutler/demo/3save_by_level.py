import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
SEGMENT_PATH = Path("output_segments/test4")
ORIGINAL_PATH = Path("demo/imgs/test")
BLOCKS_PATH = Path("output_segments/test4Levels")

MATCH_THRESHOLD = 0.05
NMS_OVERLAP_THRESHOLD = 0.30

def load_images(base_path: Path, recursive: bool = False, description: str = "Loading", read_flag: int = cv2.IMREAD_COLOR):
    pattern = '**/*.png' if recursive else '*.png'
    image_files = list(base_path.glob(pattern))
    image_cache = {}
    
    if not image_files:
        print(f"Warning: No PNG files found in {base_path}", file=sys.stderr)
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

    print("\nPrecomputing template masks and color spaces...")
    processed_templates = {}
    for tpl_name, img_bgra in template_cache.items():
        if img_bgra.shape[2] != 4: 
            continue
        
        mask = img_bgra[:, :, 3]
        if cv2.countNonZero(mask) == 0: 
            continue
            
        bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        
        processed_templates[tpl_name] = {
            "bgra": img_bgra,
            "bgr": bgr,
            "mask": mask,
            "w": w,
            "h": h
        }

    print("\n--- Starting Level Processing ---")
    pbar_levels = tqdm(source_image_cache.items(), desc="Processing Levels", unit="level")

    for level_name, level_img in pbar_levels:
        pbar_levels.set_postfix_str(f"Scanning {level_name}...")
        
        candidates = [] 
        
        for template_name, tpl_data in processed_templates.items():
            try:
                result = cv2.matchTemplate(level_img, tpl_data["bgr"], cv2.TM_SQDIFF_NORMED, mask=tpl_data["mask"])
                
                # Find EVERY location where this tile matches below the threshold
                locs = np.where(result <= MATCH_THRESHOLD)
                
                for y, x in zip(*locs):
                    error = result[y, x]
                    score = 1.0 - error # Convert error to a score (NMS wants higher = better)
                    
                    candidates.append({
                        "box": [int(x), int(y), int(tpl_data["w"]), int(tpl_data["h"])],
                        "score": float(score),
                        "name": template_name
                    })
                    
            except cv2.error:
                continue

        # --- NMS REFEREE ---
        if candidates:
            boxes = [c["box"] for c in candidates]
            scores = [c["score"] for c in candidates]
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=NMS_OVERLAP_THRESHOLD)
            
            if len(indices) > 0:
                surviving_templates = set()
                
                for i in indices.flatten():
                    surviving_templates.add(candidates[i]["name"])
                    
                level_folder_name = Path(level_name).stem
                level_output_dir = BLOCKS_PATH / level_folder_name
                level_output_dir.mkdir(parents=True, exist_ok=True)
                
                for tpl_name in surviving_templates:
                    cv2.imwrite(str(level_output_dir / tpl_name), template_cache[tpl_name])
                
                tqdm.write(f"✅ Level: {level_name} ({len(surviving_templates)} unique tiles saved)")
            else:
                tqdm.write(f"❌ Level: {level_name} (0 tiles survived NMS)")
        else:
            tqdm.write(f"❌ Level: {level_name} (0 candidate tiles found)")

    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()