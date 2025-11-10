import numpy as np
import cv2 as cv
from math import sin, sqrt, pi

rWall = np.zeros(shape = (700, 700), dtype = np.uint8)
lWall = np.zeros(shape = (700, 700), dtype = np.uint8)

a = 50
b = 3*(pi/700)

Rfile = open('R.txt', 'wt')
Lfile = open('L.txt', 'wt')
d_path = np.zeros(shape =(700, 700), dtype = np.uint8)

for y in range(-349, 351):
    x = int(a*sin(b*y) - a)

    i = 350 - y
    j = x + 350
    Rfile.write(str(j + 55) + ',' + str(i) +'\n')
    Lfile.write(str(j - 55) + ',' + str(i) +'\n')
    rWall[i, (j + 55):(j + 60)] = 255
    lWall[i, (j - 60):(j - 55)] = 255
    d_path[i, j] = 255

cv.imwrite('R.png', rWall)
cv.imwrite('L.png', lWall)
cv.imwrite('desired_path.png', d_path)

Rfile.close()
Lfile.close()