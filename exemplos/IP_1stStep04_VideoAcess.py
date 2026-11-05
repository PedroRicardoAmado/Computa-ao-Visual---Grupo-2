#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep04_VideoAcess.py   (na pasta do PL04.zip)
# Ler um vídeo de um ficheiro, à velocidade dele.

import cv2 as cv
import time
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

cap = cv.VideoCapture(str(PASTA / "videos" / "cadeira-animacao.mp4"))

# Calculate the inter-frame delay from the video's FPS
frame_delay = int(1000 / cap.get(cv.CAP_PROP_FPS))

while cap.isOpened():
    # Time the loop start
    start_time = time.perf_counter()

    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow("Video", frame)

    # Subtract the processing time of the loop from the remaining waiting time (for waitKey)
    elapsed_time = time.perf_counter() - start_time
    remaining_time = max(0, frame_delay - int(elapsed_time * 1000))

    if cv.waitKey(remaining_time) == 27:  # ESC
        break

cv.destroyAllWindows()
cap.release()
