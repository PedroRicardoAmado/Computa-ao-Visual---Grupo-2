#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_00_PrintImg.py   (na pasta do PL04.zip)

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv.imread(str(PASTA / "assets" / "esplanada.png0001.png"), cv.IMREAD_GRAYSCALE)

if img is None:
    raise SystemExit("imread devolveu None: verifica o caminho imgs/astronaut.png")

print(img)
print("tipo:", img.dtype)        # uint8
print("dimensoes:", img.shape)   # (512, 512)

print(img[0, 0])       # [linha, coluna], nao [x, y]
img[0, 2] = 0          # um pixel a preto
img[2, 0] = 255        # um pixel a branco

patch = img[0:2, 0:4]  # 2 linhas, 4 colunas
print(patch)
