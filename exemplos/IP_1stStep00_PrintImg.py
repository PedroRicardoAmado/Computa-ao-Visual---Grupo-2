#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep00_PrintImg.py   (na pasta do PL04.zip)
# A imagem como matriz de números.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

image_path = str(PASTA / "imgs" / "astronaut.png")

# Read image in Grayscale format
test_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

print(test_img)
print(f"\nData type = {test_img.dtype}\n")
print(f"Object type = {type(test_img)}\n")
print(f"Image Dimensions = {test_img.shape}\n")

print(test_img[0, 0])  # Remember: [y,x] <=> [row,col]

test_img[0, 2] = 0
test_img[2, 0] = 255

print(test_img)  # (y,x)

test_patch = test_img[0:2, 0:4]
print(test_patch)  # (y,x)

test_img[0:2, 0:4] = 0
print(test_img)  # (y,x)
