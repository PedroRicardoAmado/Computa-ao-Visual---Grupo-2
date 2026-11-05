#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_01_LoadSaveImg.py   (na pasta do PL04.zip)

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv.imread(str(PASTA / "assets" / "esplanada.png0001.png"), cv.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("imread devolveu None: verifica o caminho imgs/astronaut.png")

cv.imshow("Imagem", img)

cv.imwrite(str(PASTA / "assets" / "esplanada.png0001.png"), img)

cv.imwrite(str(PASTA / "assets" / "esplanada.png0001.png"), img)

cv.waitKey(0)          # espera por uma tecla
cv.destroyAllWindows()
