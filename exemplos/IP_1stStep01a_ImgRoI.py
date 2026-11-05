#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep01a_ImgRoI.py   (na pasta do PL04.zip)
# Copiar uma região de interesse (RoI) para outro sítio.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv.imread(str(PASTA / "imgs" / "astronaut.png"), 0)
cv.imshow("Original", img)

# [line, column] == [y, x]
patch = img[50:150, 100:150]
img[100:200, 400:450] = patch

cv.imshow("RoI modified", img)

# Wait for ESC
while cv.waitKey(0) != 27:
    pass

cv.destroyAllWindows()
