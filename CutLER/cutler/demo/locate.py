import cv2
import os
import numpy as np

folder  = "demo/imgs/test"

for filename in os.listdir(folder):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(folder, filename)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    cv2.imshow("img",img)
    cv2.waitKey(0)