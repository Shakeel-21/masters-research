# Python program to illustrate
# template matching
import cv2
import numpy as np


segment_path="output_segments/results/point1_15/cluster_6/mario-6-2_seg_8.png"
original_path="demo/imgs/all_levels/mario-6-2.png"
# Read the main image
img_rgb = cv2.imread(original_path)

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread(segment_path, 0)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.8

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0) 

# ---- And this is good practice to add after ----
cv2.destroyAllWindows()