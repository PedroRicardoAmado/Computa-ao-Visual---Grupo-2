#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep01_LoadSaveImg.py   (na pasta do PL04.zip)
# Mostrar e gravar, a cores e em cinzento.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load a color image (for grayscale use cv.IMREAD_GRAYSCALE)
img = cv.imread(str(PASTA / "imgs" / "astronaut.png"), cv.IMREAD_COLOR)

# Show the image in a window
cv.imshow("Image", img)

# If you want to resize the window
# cv.namedWindow('Image', cv.WINDOW_NORMAL)

# Save the color image
cv.imwrite(str(PASTA / "imgs" / "astronautSave.png"), img)

# Save the image in grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite(str(PASTA / "imgs" / "astronautSaveGray.png"), img_gray)

# Wait to end the process
cv.waitKey(0)
cv.destroyAllWindows()
