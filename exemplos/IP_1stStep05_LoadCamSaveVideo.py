#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep05_LoadCamSaveVideo.py   (na pasta do PL04.zip)
# Gravar a câmara num ficheiro de vídeo.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Open a video capture connection
cap = cv.VideoCapture(0)  # Local camera
# cap = cv.VideoCapture("http://<IP do telemóvel>:4747/video")  # IP (remote) camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)  # exit with an error code; exit (0) --> exit ok

# Get the frame resolution, which is system dependent
# Cast the resolutions from float to integer
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_fps = cap.get(cv.CAP_PROP_FPS) or 30  # some cameras report 0 FPS

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*"mp4v")

out = cv.VideoWriter(str(PASTA / "videos" / "output.mp4"), fourcc, frame_fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the video, if necessary
    #frame = cv.flip(frame, 0)
    
    # Write the frame
    out.write(frame)

    cv.imshow("frame", frame)
    if cv.waitKey(1) == 27:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
