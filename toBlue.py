import cv2
import numpy as np
import time
import random

img_path = r"C:\Users\kashz\AI Life\AI Projects - IAIP, PTs (Web +"

img = cv2.imread(img_path)
img = np.array(Image.open(img_path))
# img = cv2.resize(img, (1280, 720))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Add white pixels
img_copy = img.copy()
img_copy[0][0] = np.array([255, 255, 255])

cv2.imshow("Image", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()