#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep03_CamListed.py   (na pasta do PL04.zip)
# Quantas câmaras há, com a resolução e os FPS de cada uma.

import cv2 as cv
import numpy as np
import time

# Select the camera source
camera_src = 0
# camera_src = "http://<IP do telemóvel>:4747/video"

while True:
    # Open the video feed
    cap = cv.VideoCapture(camera_src)

    # Check if the camera opened successfully
    if not cap.isOpened():
        if type(camera_src) == int and camera_src > 0:  # no more cameras: keep the last one that opened
            camera_src -= 1
            break
        print("Cannot open camera")
        exit(1)

    # Get the width, height and fps of the video feed
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv.CAP_PROP_FPS)

    # Number of frames to capture
    num_frames = 120

    # Get the start time
    # start = time.time() # Less accurate
    start = time.perf_counter()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = cap.read()

    # Get the end time
    end = time.perf_counter()

    # Time elapsed
    seconds = end - start

    # Calculate frames per second
    fps = num_frames / seconds

    print(
        f"Camera: {camera_src} width: {frame_width} height: {frame_height} fps: {int(fps)}"
    )

    cap.release()
    match type(camera_src) == int:
        case True:
            camera_src += 1
        case False:
            break


##################################################

# Sleep for 2 seconds to allow the camera to be released properly
time.sleep(2)

# Select the camera source
# camera_src = 0
# camera_src = "http://<IP do telemóvel>:4747/video"

# Open the video feed
cap = cv.VideoCapture(camera_src)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

while cap.isOpened():
    # ret, img = cap.read()
    # _, img = cap.read() # Uses a dummy variable: _ (yes, that is a valid variable name)
    img = cap.read()[1]

    cv.imshow("frame", img)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
