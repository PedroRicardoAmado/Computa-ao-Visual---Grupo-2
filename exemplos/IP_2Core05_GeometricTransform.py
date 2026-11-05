#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core05_GeometricTransform.py   (na pasta do PL04.zip)
# Escala, translação, rotação e perspetiva.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img_orig = cv.imread(str(PASTA / "imgs" / "astronaut.png"))

# ----- Scaling -----
print("Scaling")
res1 = cv.resize(img_orig, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# OR

height, width = img_orig.shape[:2]  # rows,cols,channels = img_orig2.shape
res2 = cv.resize(img_orig, (2 * width, 2 * height), interpolation=cv.INTER_LANCZOS4)
res3 = cv.resize(img_orig, (int(width / 5), 3 * height), interpolation=cv.INTER_AREA)

cv.imshow("Original", img_orig)
cv.imshow("Scaling 1", res1)
cv.imshow("Scaling 2", res2)
cv.imshow("Scaling 3", res3)

cv.waitKey(0)
cv.destroyAllWindows()

# ----- Translation -----
print("Translation")

rows, cols = img_orig.shape[:2]  # rows,cols,channels = img_orig2.shape

M = np.float32([[1, 0, 200], [0, 1, 50]])
img_out = cv.warpAffine(img_orig, M, (cols, rows))

cv.imshow("Original", img_orig)
cv.imshow("Translation", img_out)
cv.waitKey(0)
cv.destroyAllWindows()

# ----- Rotation -----
print("Rotation")

# cv.getRotationMatrix2D(center, angle, scale)
Mrot = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
img_out = cv.warpAffine(img_orig, Mrot, (cols, rows))

Mrot2 = cv.getRotationMatrix2D((0, 0), 45, 0.5)
img_out2 = cv.warpAffine(img_orig, Mrot2, (cols, rows))

cv.imshow("Original", img_orig)
cv.imshow("Rotation 1", img_out)
cv.imshow("Rotation 2", img_out2)
cv.waitKey(0)
cv.destroyAllWindows()

# ----- PrespectiveTransform -----
print("Prespective Transform")

rows, cols, ch = img_orig.shape
print(rows, cols)

pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
pts2 = np.float32([[100, 150], [300, 180], [150, 250], [350, 350]])

M = cv.getPerspectiveTransform(pts1, pts2)
img_out = cv.warpPerspective(img_orig, M, (cols, rows))

cv.imshow("Original", img_orig)
cv.imshow("Homography Transform", img_out)
cv.waitKey(0)
cv.destroyAllWindows()
