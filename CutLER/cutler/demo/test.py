import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_ROOT = 'output_segments/transRemake'
OUTPUT_FOLDER = 'output_segments/core3'

SIMILARITY_THRESHOLD = 1000 
MIN_VISIBLE_PIXELS = 128
COLOR_ROUNDING = 20
BLUR_KERNEL = (5, 5)
BUCKET_SEARCH_RANGE = 20

def load_images_from_folder(folder):
    path_obj = Path(folder).resolve()
    if not path_obj.exists():
        print(f"Error: The folder {path_obj} does not exist.")
        return []
    return list(path_obj.rglob('*.png'))

def clean_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[img[:, :, 3] == 0] = [0, 0, 0, 0]
    return img

def preprocess_for_matching(img):
    processed = img.copy()
    bgr = processed[:, :, :3]
    blurred_bgr = cv2.GaussianBlur(bgr, BLUR_KERNEL, 0)
    quantized_bgr = (blurred_bgr // COLOR_ROUNDING) * COLOR_ROUNDING + COLOR_ROUNDING // 2
    processed[:, :, :3] = quantized_bgr
    return processed.astype(np.uint8)

def get_mean_brightness(img):
    return int(np.mean(img[:, :, :3]))

# --- NEW FUNCTION: THE RESIZE FIX ---
def pad_to_16x16_transparent(img):
    """
    Pastes the image into a 16x16 transparent canvas.
    Does NOT stretch or distort pixels.
    """
    h, w = img.shape[:2]
    
    # If it's already perfect, return it
    if h == 16 and w == 16:
        return img.copy()

    # Create a blank 16x16 canvas filled with ZEROS (Transparent Black)
    # Zeros in the Alpha channel mean "Ignore this" to matchTemplate
    canvas = np.zeros((16, 16, 4), dtype=np.uint8)
    
    # Paste the image in the top-left corner
    # You can change this to center it if you prefer, but top-left is standard
    copy_h = min(h, 16)
    copy_w = min(w, 16)
    
    canvas[0:copy_h, 0:copy_w] = img[0:copy_h, 0:copy_w]
    
    return canvas

def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = load_images_from_folder(input_dir)
    print(f"Found {len(image_files)} files. Processing...")

    tile_buckets = defaultdict(list)
    saved_count = 0
    duplicate_count = 0
    image_files.sort()

    for idx, file_path in enumerate(image_files):
        if idx % 100 == 0:
            print(f"Scanning {idx}/{len(image_files)}... (Saved: {saved_count} | Dups: {duplicate_count})")

        img_raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None: continue

        img = clean_image(img_raw)
        h, w = img.shape[:2]

        candidates = []

        # --- CANDIDATE EXTRACTION (UPDATED) ---
        # We use pad_to_16x16_transparent instead of cv2.resize
        
        # CASE A: Small loose tiles
        if 15 <= h <= 17 and 15 <= w <= 17:
            # If > 16, we crop. If < 16, we pad transparently.
            if h > 16 or w > 16:
                # Basic center crop for slightly too large items
                candidates.append(img[0:16, 0:16]) 
            else:
                candidates.append(pad_to_16x16_transparent(img))

        # CASE B: Vertical Strips
        elif 15 <= w <= 17 and h >= 15:
            current_y = h
            while current_y > 0:
                bottom = current_y
                top = current_y - 16
                
                # Handle the top chunk
                if top < 0:
                    if current_y >= 15:
                        chunk = img[0:current_y, 0:w]
                        candidates.append(pad_to_16x16_transparent(chunk))
                    break 
                
                # Handle standard chunks
                chunk = img[top:bottom, 0:w]
                candidates.append(pad_to_16x16_transparent(chunk))
                
                current_y -= 16
                
        # CASE C: Perfect Grid (Unchanged)
        elif h >= 16 and w >= 16:
            if h % 16 == 0 and w % 16 == 0:
                for y in range(0, h, 16):
                    for x in range(0, w, 16):
                        tile = img[y:y+16, x:x+16]
                        candidates.append(tile.copy())

        # --- MATCHING LOGIC ---
        for cand in candidates:
            if np.sum(cand[:, :, 3]) < (MIN_VISIBLE_PIXELS * 255):
                continue

            cand_fuzzy = preprocess_for_matching(cand)
            cand_mean = get_mean_brightness(cand_fuzzy)
            
            potential_duplicates = []
            start = max(0, cand_mean - BUCKET_SEARCH_RANGE)
            end = min(256, cand_mean + BUCKET_SEARCH_RANGE + 1)
            
            for b_key in range(start, end):
                if b_key in tile_buckets:
                    potential_duplicates.extend(tile_buckets[b_key])

            is_unique = True
            cand_bgr = cand_fuzzy[:, :, :3]
            cand_mask = cand_fuzzy[:, :, 3] # <--- THE MAGIC IS HERE

            if potential_duplicates:
                for (exist_orig, exist_fuzzy) in potential_duplicates:
                    
                    # NOTE: We reverted to constant black border for the duplicate check
                    # because we are now relying on the MASK to handle the edges.
                    padded_existing = cv2.copyMakeBorder(
                        exist_fuzzy[:, :, :3], 1, 1, 1, 1, 
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    
                    try:
                        # Because cand_mask has 0s in the padded row, 
                        # matchTemplate effectively ignores that row entirely.
                        res = cv2.matchTemplate(padded_existing, cand_bgr, cv2.TM_SQDIFF_NORMED, mask=cand_mask)
                        min_val, _, _, _ = cv2.minMaxLoc(res)
                        
                        if min_val < SIMILARITY_THRESHOLD:
                            is_unique = False
                            duplicate_count += 1
                            break
                    except Exception:
                        continue
            
            if is_unique:
                tile_buckets[cand_mean].append((cand, cand_fuzzy))
                saved_count += 1
                filename = f"tile_{saved_count:05d}.png"
                cv2.imwrite(os.path.join(output_dir, filename), cand)

    print(f"--- DONE ---")
    print(f"Saved: {saved_count}")
    print(f"Discarded Duplicates: {duplicate_count}")

if __name__ == "__main__":
    process_dataset(INPUT_ROOT, OUTPUT_FOLDER)