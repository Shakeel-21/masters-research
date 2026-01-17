import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_levels(level_path):
    """
    Loads all source level images into the cache (as 3-channel BGR).
    """
    pattern = '*.png'
    image_files = list(level_path.glob(pattern))
    image_cache = {}

    print(f"\nLoading {len(image_files)} levels from {level_path}...")
    pbar = tqdm(image_files, desc="Loading Levels", unit="img")
    for img_path in pbar:
        # Load image, preserving alpha channel if it exists
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED) 
        if img is not None:
            image_cache[img_path.name] = img
            
    return image_cache
    
def compare_images(img1, img2):
    """
    Compares two images (loaded as numpy arrays) and returns pixel identity %.
    Handles potential 4-channel (alpha) images by comparing BGR only.
    Handles dimension mismatches by resizing.
    """
    # --- Pre-processing ---
    # Handle potential 4th (alpha) channel by keeping only BGR
    if img1.shape[2] == 4:
        img1 = img1[:, :, :3]
    if img2.shape[2] == 4:
        img2 = img2[:, :, :3]
    
    # Handle dimension mismatch (e.g., if reconstruction is diff size)
    h1, w1, _ = img1.shape
    if img1.shape != img2.shape:
        # Resize img2 (reconstruction) to match img1 (original)
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

    # --- Comparison Logic ---
    # Calculate the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Count identical pixels (where difference is 0 across all channels)
    identical_pixels = np.sum(diff == 0) // img1.shape[2] # Divide by number of channels (3)

    # Calculate total number of pixels
    total_pixels = img1.shape[0] * img1.shape[1]

    # Calculate percentage
    percentage_identical = (identical_pixels / total_pixels) * 100

    return percentage_identical


if __name__ == "__main__":
    
    RECONSTRUCTION_PATH = Path("output_segments/core3Recon")
    ORIGINAL_PATH = Path("demo/imgs/test")

    # 1. Load both sets of images into memory
    print("--- Loading Original Images ---")
    original_images = load_levels(ORIGINAL_PATH)
    
    print("\n--- Loading Reconstructed Images ---")
    reconstructed_images = load_levels(RECONSTRUCTION_PATH)
    
    print("\n--- 🔎 Starting Comparison ---")
    
    if not original_images:
        print(f"Error: No images found in {ORIGINAL_PATH}. Exiting.")
        exit()
        
    total_identity = 0
    num_compared = 0
    
    # 2. Iterate through the original images (sorted for clean output)
    for filename in sorted(original_images.keys()):
        original_img = original_images[filename]
        
        # 3. Find the matching reconstructed image by filename
        original_stem = Path(filename).stem
        reconstructed_filename = f"{original_stem}_reconstructed.png"
        
        reconstructed_img = reconstructed_images.get(reconstructed_filename)
        
        print(f"Processing: {filename}")
        
        if reconstructed_img is not None:
            # 4. A match is found, run the comparison
            identity_pct = compare_images(original_img, reconstructed_img)
            
            # 5. Print the result for this pair
            print(f"  ✅ Match Found. Pixel Identity: {identity_pct:.2f}%")
            
            total_identity += identity_pct
            num_compared += 1
        else:
            # 6. No matching file was found
            print(f"  ❌ No match found for '{reconstructed_filename}' in {RECONSTRUCTION_PATH}.")
            
    # 7. Print a final summary
    if num_compared > 0:
        average_identity = total_identity / num_compared
        print("\n--- 📊 Summary ---")
        print(f"Compared {num_compared} matching image pairs.")
        print(f"Average Pixel Identity: {average_identity:.2f}%")
    else:
        print("\n--- 📊 Summary ---")
        print("No matching image pairs were found to compare.")