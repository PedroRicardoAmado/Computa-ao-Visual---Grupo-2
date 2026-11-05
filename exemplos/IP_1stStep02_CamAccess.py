#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep02_CamAccess.py   (na pasta do PL04.zip)
# A câmara em tempo real, a cores e em cinzento.

import cv2 as cv

# Open a video capture connection
cap = cv.VideoCapture(0)  # Local camera
# cap = cv.VideoCapture("http://<IP do telemóvel>:4747/video")  # IP (remote) camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)  # exit with an error code; exit (0) --> exit ok

# Get the video width and height as integers
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Process the video feed
while cap.isOpened():
    # Capture frame-by-frame: (bool:True/False, frame)
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # --- Frame processing ---
    # Rotate the video 90 degrees clockwise
    # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)  # só para o telemóvel na vertical
    # Create a gray frame from the feed
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frames
    cv.imshow("frameGray", gray)
    cv.imshow("frameColor", frame)

    # Break the loop on ESC
    if cv.waitKey(1) == 27:  # ESC or select a key --> ord('q'):
        break

# Close the video feed
cap.release()
cv.destroyAllWindows()
