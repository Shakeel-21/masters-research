import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_ROOT = 'output_segments/transRemake'
OUTPUT_FOLDER = 'output_segments/core3'

# THRESHOLD: 750 is very loose. Combined with blur, this will merge many things.
SIMILARITY_THRESHOLD = 1000.0 

# FILTER: Minimum non-transparent pixels
MIN_VISIBLE_PIXELS = 128

# COLOR ROUNDING: 50 is very high (blocks colors into groups of 20%)
COLOR_ROUNDING = 50

# BLUR SETTING: 
# (3, 3) is standard for 16x16 images. (5, 5) might be too blurry and lose details.
BLUR_KERNEL = (5, 5)

# BUCKET SEARCH
BUCKET_SEARCH_RANGE = 20

def load_images_from_folder(folder):
    """Recursively find all PNG images."""
    path_obj = Path(folder).resolve()
    if not path_obj.exists():
        print(f"Error: The folder {path_obj} does not exist.")
        return []
    return list(path_obj.rglob('*.png'))

def clean_image(img):
    """Standardizes to 4-channel BGRA and cleans invisible pixels."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Force invisible pixels to pure black
    img[img[:, :, 3] == 0] = [0, 0, 0, 0]
    return img

def preprocess_for_matching(img):
    """
    Applies BLUR and QUANTIZATION to create a 'fuzzy' version for comparison.
    """
    # 1. BLUR (Only blur the colors, NOT the alpha channel)
    # We copy the image so we don't ruin the original
    processed = img.copy()
    
    # Extract BGR (Colors) and Alpha
    bgr = processed[:, :, :3]
    alpha = processed[:, :, 3]
    
    # Apply Gaussian Blur to colors
    # This smooths out noise/dithering patterns
    blurred_bgr = cv2.GaussianBlur(bgr, BLUR_KERNEL, 0)
    
    # 2. QUANTIZE (Round colors)
    # Integer division trick
    quantized_bgr = (blurred_bgr // COLOR_ROUNDING) * COLOR_ROUNDING + COLOR_ROUNDING // 2
    
    # Recombine
    processed[:, :, :3] = quantized_bgr
    
    # Return as uint8
    return processed.astype(np.uint8)
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
def get_mean_brightness(img):
    return int(np.mean(img[:, :, :3]))

def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = load_images_from_folder(input_dir)
    print(f"Found {len(image_files)} files. Processing...")

    tile_buckets = defaultdict(list)
    saved_count = 0
    duplicate_count = 0

    for idx, file_path in enumerate(image_files):
        if idx % 100 == 0:
            print(f"Scanning {idx}/{len(image_files)}... (Saved: {saved_count} | Dups: {duplicate_count})")

        img_raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None: continue

        img = clean_image(img_raw)
        h, w = img.shape[:2]

        candidates = []

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

        # 2. PROCESS
        for cand in candidates:
            # A. Trash Check (Empty Air)
            if np.sum(cand[:, :, 3]) < (MIN_VISIBLE_PIXELS * 255):
                continue

            # B. CREATE FUZZY MATCH VERSION (Blur + Quantize)
            # cand_fuzzy is used ONLY for math. cand is used for saving.
            cand_fuzzy = preprocess_for_matching(cand)
            
            # C. Bucketing (Use fuzzy version for bucket key)
            cand_mean = get_mean_brightness(cand_fuzzy)
            
            potential_duplicates = []
            start = max(0, cand_mean - BUCKET_SEARCH_RANGE)
            end = min(256, cand_mean + BUCKET_SEARCH_RANGE + 1)
            
            for b_key in range(start, end):
                if b_key in tile_buckets:
                    potential_duplicates.extend(tile_buckets[b_key])

            # D. The Check
            is_unique = True
            
            if potential_duplicates:
                cand_bgr = cand_fuzzy[:, :, :3]
                cand_mask = cand_fuzzy[:, :, 3]

                for (exist_orig, exist_fuzzy) in potential_duplicates:
                    # Pad existing FUZZY tile
                    padded_existing = cv2.copyMakeBorder(
                        exist_fuzzy[:, :, :3], 1, 1, 1, 1, 
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    
                    try:
                        res = cv2.matchTemplate(padded_existing, cand_bgr, cv2.TM_SQDIFF, mask=cand_mask)
                        min_val, _, _, _ = cv2.minMaxLoc(res)
                        error = min_val / 256.0

                        if error < SIMILARITY_THRESHOLD:
                            is_unique = False
                            duplicate_count += 1
                            break
                    except:
                        continue
            
            if is_unique:
                # Add (Original, Fuzzy) to bucket
                tile_buckets[cand_mean].append((cand, cand_fuzzy))
                saved_count += 1
                filename = f"tile_{saved_count:05d}.png"
                cv2.imwrite(os.path.join(output_dir, filename), cand)

    print(f"--- DONE ---")
    print(f"Saved: {saved_count}")
    print(f"Discarded Duplicates: {duplicate_count}")

if __name__ == "__main__":
    process_dataset(INPUT_ROOT, OUTPUT_FOLDER)