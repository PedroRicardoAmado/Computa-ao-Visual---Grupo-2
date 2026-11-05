#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
# ]
# ///
import sys

import cv2 as cv
import numpy as np

print("Python", sys.version.split()[0])
print("OpenCV", cv.__version__)
print("NumPy", np.__version__)
