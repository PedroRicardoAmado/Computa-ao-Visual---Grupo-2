#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep01b_ImgPatch.py   (na pasta do PL04.zip)
# Cópias, imagens vazias, recortes e cv.resize.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

image_path = str(PASTA / "imgs" / "astronaut.png")

# Read image in Color format (default)
img_orig = cv.imread(image_path, 1)
print("Image Dimensions: ", img_orig.shape)

# Create an image copy
img_copy = img_orig.copy()
print("Image Dimensions: ", img_copy.shape)

# Create an empty image, size M x N
img_empty = 255 * np.ones((256, 256, 3), dtype="uint8")
print("Image Dimensions: ", img_empty.shape)

# Create an empty image with the size of the original image
img_empty_orig = np.zeros_like(img_orig)
print("Image Dimensions: ", img_empty_orig.shape)

# Crop an image patch ("RoI") Example:
# x coordinates = 170 to 320; y coordinates = 40 to 200
patch = img_orig[40:201, 170:321]  # (y,x)
print("Image Dimensions: ", patch.shape)

# Find height and width of the patch
patch_h, patch_w = patch.shape[:2]

# Copy to left and right
img_copy[40 : 40 + patch_h, 10 : 10 + patch_w] = patch
img_copy[40 : 40 + patch_h, 300 : 300 + patch_w] = patch

cv.imshow("Image Copy", img_copy)

# Resize
# Set rows and columns
resize_down_w = 300
resize_down_h = 200
img_downscaled = cv.resize(
    img_copy, (resize_down_w, resize_down_h), interpolation=cv.INTER_AREA
)
cv.imshow("Downscaled", img_downscaled)

# Mess up the aspect ratio
resize_down_w = 600
resize_down_h = 900
img_upscaled = cv.resize(
    img_copy, (resize_down_w, resize_down_h), interpolation=cv.INTER_LANCZOS4
)
cv.imshow("Upscaled", img_upscaled)
cv.waitKey(0)
