#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_02_CamAccess.py   (na pasta do PL04.zip; ESC para sair)

import cv2 as cv

cap = cv.VideoCapture(0)   # 0 = camara local; experimentar 1 se nao abrir
# telemovel como camara (app DroidCam, na mesma rede Wi-Fi):
# cap = cv.VideoCapture("http://IP:4747/video")

if not cap.isOpened():
    raise SystemExit("Nao abriu a camara")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Cor", frame)
    cv.imshow("Cinzento", gray)
    if cv.waitKey(1) == 27:   # ESC
        break

cap.release()
cv.destroyAllWindows()
