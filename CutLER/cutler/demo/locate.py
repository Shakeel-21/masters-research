import cv2
import numpy as np

# --- Paths ---
segment_path = "output_segments/unmatched/mario-2-1_seg_8.png"
original_path = "demo/imgs/all_levels/mario-2-1.png"
# Read the main image
img_rgb = cv2.imread(original_path)

# 1. Load the template in COLOR first
template_bgr = cv2.imread(segment_path, cv2.IMREAD_COLOR)

if img_rgb is None or template_bgr is None:
    print(f"Error: Could not load images. Check paths.")
    print(f"Original: {original_path}")
    print(f"Segment: {segment_path}")
    exit()

# 2. Convert template to grayscale
template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

# 3. Create a binary mask
_ , mask = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY)


w, h = template_bgr.shape[:2]

# Perform match operations using the new mask
res = cv2.matchTemplate(img_rgb, template_bgr, cv2.TM_SQDIFF, mask=mask)

# Specify a threshold
threshold = 1000 

# Store the coordinates of matched area in a numpy array
loc = np.where(res <= threshold)

if loc[0].size == 0:
    print(f"No matches found with threshold {threshold}.")
    
    # --- Debugging: Print max value ---
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    print(f"Debug: The highest match value found was: {max_val}")
    print("If this value is close (e.g., 0.85), try lowering the threshold.")
    
else:
    print(f"Found {len(loc[0])} matches.")
    
    # Draw a rectangle around the matched region.
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    # Show the final image with the matched area.
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# Add a final message in case the 'else' block was skipped
print("--- Script finished ---")