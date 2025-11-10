import numpy as np
import cv2 as cv

img = cv.imread("actual_path.png", cv.IMREAD_COLOR)

rows, cols, channels = img.shape

black = np.array([0, 0, 0], dtype = np.uint8)
white = np.array([255, 255, 255], dtype = np.uint8)
red = np.array([0, 0, 255], dtype = np.uint8)

for r in range(rows):
    for c in range(cols):
        color = img[r, c]
        if color[0] == 0 and color[1] == 0 and color[2] == 0:
            img[r, c] = white
        elif color[0] == 255 and color[1] == 255 and color[2] == 255:
            img[r, c] = black
        elif color[0] == 0 and color[1] == 255 and color[2] == 0:
            pass
            #cv.circle(img,(c, r), 5, (0, 0, 0), 1)

for r in range(rows):
    for c in range(cols):

        if (r%2 != 0) or (c%2 != 0):
            continue

        color = img[r, c]
        if color[0] == 0 and color[1] == 255 and color[2] == 0:
            cv.circle(img,(c, r), 3, (0, 0, 0), 1)

cv.imwrite("path.png", img)